import Mathlib

namespace parabola_equation_l450_45068

-- Define the parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - 4 * y - 24 = 0

-- Theorem statement
theorem parabola_equation (p : Parabola) :
  (∀ x y, p.equation x y ↔ x^2 = 2 * y) →  -- Standard form of parabola with vertex at origin and y-axis as axis of symmetry
  (∃ x y, p.equation x y ∧ line x y) →     -- Focus lies on the given line
  (∀ x y, p.equation x y ↔ x^2 = -24 * y)  -- Conclusion: The standard equation is x² = -24y
  := by sorry

end parabola_equation_l450_45068


namespace classics_section_books_l450_45064

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := 6

/-- The number of books per author -/
def books_per_author : ℕ := 33

/-- The total number of books in Jack's classics section -/
def total_books : ℕ := num_authors * books_per_author

theorem classics_section_books :
  total_books = 198 :=
by sorry

end classics_section_books_l450_45064


namespace quiz_correct_answers_l450_45093

theorem quiz_correct_answers (total : ℕ) (difference : ℕ) (sang_hyeon : ℕ) : 
  total = sang_hyeon + (sang_hyeon + difference) → 
  difference = 5 → 
  total = 43 → 
  sang_hyeon = 19 := by sorry

end quiz_correct_answers_l450_45093


namespace legos_lost_l450_45043

def initial_legos : ℕ := 380
def given_to_sister : ℕ := 24
def current_legos : ℕ := 299

theorem legos_lost : initial_legos - given_to_sister - current_legos = 57 := by
  sorry

end legos_lost_l450_45043


namespace cube_plane_difference_l450_45058

/-- Represents a cube with points placed on each face -/
structure MarkedCube where
  -- Add necessary fields

/-- Represents a plane intersecting the cube -/
structure IntersectingPlane where
  -- Add necessary fields

/-- Represents a segment on the surface of the cube -/
structure SurfaceSegment where
  -- Add necessary fields

/-- The maximum number of planes required to create all possible segments -/
def max_planes (cube : MarkedCube) : ℕ := sorry

/-- The minimum number of planes required to create all possible segments -/
def min_planes (cube : MarkedCube) : ℕ := sorry

/-- All possible segments on the surface of the cube -/
def all_segments (cube : MarkedCube) : Set SurfaceSegment := sorry

/-- The set of segments created by a given set of planes -/
def segments_from_planes (cube : MarkedCube) (planes : Set IntersectingPlane) : Set SurfaceSegment := sorry

theorem cube_plane_difference (cube : MarkedCube) :
  max_planes cube - min_planes cube = 24 :=
sorry

end cube_plane_difference_l450_45058


namespace unique_factorization_l450_45072

/-- A factorization of 2210 into a two-digit and a three-digit number -/
structure Factorization :=
  (a : ℕ) (b : ℕ)
  (h1 : 10 ≤ a ∧ a ≤ 99)
  (h2 : 100 ≤ b ∧ b ≤ 999)
  (h3 : a * b = 2210)

/-- Two factorizations are considered equal if they have the same factors (regardless of order) -/
def factorization_eq (f1 f2 : Factorization) : Prop :=
  (f1.a = f2.a ∧ f1.b = f2.b) ∨ (f1.a = f2.b ∧ f1.b = f2.a)

/-- The set of all valid factorizations of 2210 -/
def factorizations : Set Factorization :=
  {f : Factorization | true}

theorem unique_factorization : ∃! (f : Factorization), f ∈ factorizations :=
sorry

end unique_factorization_l450_45072


namespace fog_sum_l450_45077

theorem fog_sum (f o g : Nat) : 
  f < 10 → o < 10 → g < 10 →
  (100 * f + 10 * o + g) * 4 = 1464 →
  f + o + g = 15 := by
sorry

end fog_sum_l450_45077


namespace rectangle_area_l450_45090

theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 →
  length = 2 * width →
  area = length * width →
  area = 50 := by
sorry

end rectangle_area_l450_45090


namespace coin_count_theorem_l450_45040

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | Dollar

/-- The value of each coin type in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .Dollar => 100

/-- The set of all coin types --/
def allCoinTypes : List CoinType := [.Penny, .Nickel, .Dime, .Quarter, .Dollar]

theorem coin_count_theorem (n : ℕ) 
    (h1 : n > 0)
    (h2 : (List.sum (List.map (fun c => coinValue c * n) allCoinTypes)) = 351) :
    List.length allCoinTypes * n = 15 := by
  sorry

end coin_count_theorem_l450_45040


namespace possible_values_of_C_l450_45024

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The first number in the problem -/
def number1 (A B : Digit) : ℕ := 9100000 + 10000 * A.val + 300 + 10 * B.val + 2

/-- The second number in the problem -/
def number2 (A B C : Digit) : ℕ := 6000000 + 100000 * A.val + 10000 * B.val + 400 + 50 + 10 * C.val + 2

/-- Theorem stating the possible values of C -/
theorem possible_values_of_C :
  ∀ (A B C : Digit),
    (∃ k : ℕ, number1 A B = 3 * k) →
    (∃ m : ℕ, number2 A B C = 5 * m) →
    C.val = 0 ∨ C.val = 5 := by
  sorry

end possible_values_of_C_l450_45024


namespace hat_shoppe_pricing_l450_45049

theorem hat_shoppe_pricing (x : ℝ) (h : x > 0) : 
  0.75 * (1.3 * x) = 0.975 * x := by
  sorry

end hat_shoppe_pricing_l450_45049


namespace min_value_of_f_on_interval_l450_45015

def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3

theorem min_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 2 → f y ≥ f x) ∧
  f x = -37 :=
sorry

end min_value_of_f_on_interval_l450_45015


namespace three_m_minus_n_l450_45047

theorem three_m_minus_n (m n : ℝ) (h : m + 1 = (n - 2) / 3) : 3 * m - n = -5 := by
  sorry

end three_m_minus_n_l450_45047


namespace prime_factor_sum_l450_45075

theorem prime_factor_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 2450 → 3*w + 2*x + 7*y + 5*z = 27 := by
  sorry

end prime_factor_sum_l450_45075


namespace function_inequality_l450_45013

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x : ℝ, (deriv^[2] f) x < 2 * f x) : 
  (Real.exp 4034 * f (-2017) > f 0) ∧ (f 2017 < Real.exp 4034 * f 0) := by
  sorry

end function_inequality_l450_45013


namespace equation_solution_l450_45030

theorem equation_solution :
  ∃ x : ℤ, 45 - (x - (37 - (15 - 18))) = 57 ∧ x = 28 := by sorry

end equation_solution_l450_45030


namespace choose_team_with_smaller_variance_l450_45080

-- Define the teams
inductive Team
  | A
  | B

-- Define the properties of the teams
def average_height : ℝ := 1.72
def variance (t : Team) : ℝ :=
  match t with
  | Team.A => 1.2
  | Team.B => 5.6

-- Define a function to determine which team has more uniform heights
def more_uniform_heights (t1 t2 : Team) : Prop :=
  variance t1 < variance t2

-- Theorem statement
theorem choose_team_with_smaller_variance :
  more_uniform_heights Team.A Team.B :=
sorry

end choose_team_with_smaller_variance_l450_45080


namespace cubic_sum_given_sum_and_product_l450_45048

theorem cubic_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 16) : x^3 + y^3 = 520 := by
  sorry

end cubic_sum_given_sum_and_product_l450_45048


namespace equation_solution_l450_45022

theorem equation_solution (a : ℝ) (ha : a < 0) :
  ∃! x : ℝ, x * |x| + |x| - x - a = 0 ∧ x = -1 - Real.sqrt (1 - a) := by
  sorry

end equation_solution_l450_45022


namespace quadratic_factorization_l450_45069

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end quadratic_factorization_l450_45069


namespace michaels_book_purchase_l450_45045

theorem michaels_book_purchase (m : ℚ) : 
  (∃ (n : ℚ), (1 / 3 : ℚ) * m = (1 / 2 : ℚ) * n * ((1 / 3 : ℚ) * m / ((1 / 2 : ℚ) * n))) →
  (5 : ℚ) = (1 / 15 : ℚ) * m →
  m - ((2 / 3 : ℚ) * m + (1 / 15 : ℚ) * m) = (4 / 15 : ℚ) * m :=
by sorry

end michaels_book_purchase_l450_45045


namespace age_problem_solution_l450_45085

/-- Represents the ages of James and Joe -/
structure Ages where
  james : ℕ
  joe : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.joe = ages.james + 10 ∧
  2 * (ages.joe + 8) = 3 * (ages.james + 8)

/-- The theorem to prove -/
theorem age_problem_solution :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.james = 12 ∧ ages.joe = 22 := by
  sorry


end age_problem_solution_l450_45085


namespace equation_equivalence_l450_45083

theorem equation_equivalence : ∀ x y : ℝ, (5 * x - y = 6) ↔ (y = 5 * x - 6) := by sorry

end equation_equivalence_l450_45083


namespace exam_score_problem_l450_45009

theorem exam_score_problem (correct_score : ℕ) (incorrect_score : ℤ) 
  (total_score : ℕ) (correct_answers : ℕ) :
  correct_score = 4 →
  incorrect_score = -1 →
  total_score = 150 →
  correct_answers = 42 →
  ∃ (total_questions : ℕ), 
    total_questions = correct_answers + (correct_score * correct_answers - total_score) := by
  sorry

end exam_score_problem_l450_45009


namespace geometric_sequence_ratio_l450_45065

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →  -- sum formula
  q = 1/2 →  -- given ratio
  S 4 / a 4 = 15 := by
sorry

end geometric_sequence_ratio_l450_45065


namespace equation_solution_l450_45092

theorem equation_solution (x y A : ℝ) : 
  (x + y)^3 - x*y*(x + y) = (x + y) * A → A = x^2 + x*y + y^2 := by
sorry

end equation_solution_l450_45092


namespace intersection_when_a_is_three_possible_values_of_a_l450_45027

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 3 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x < 1 ∨ x > 6}

-- Theorem 1
theorem intersection_when_a_is_three :
  A 3 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem possible_values_of_a (a : ℝ) :
  a > 0 ∧ A a ∩ B = ∅ → 0 < a ∧ a ≤ 2 := by sorry

end intersection_when_a_is_three_possible_values_of_a_l450_45027


namespace alice_score_l450_45010

theorem alice_score (total_score : ℝ) (other_players : ℕ) (avg_score : ℝ) : 
  total_score = 72 ∧ 
  other_players = 7 ∧ 
  avg_score = 4.7 → 
  total_score - (other_players : ℝ) * avg_score = 39.1 := by
sorry

end alice_score_l450_45010


namespace charlie_metal_purchase_l450_45073

/-- Given that Charlie needs a total amount of metal and has some in storage,
    this function calculates the additional amount he needs to buy. -/
def additional_metal_needed (total_needed : ℕ) (in_storage : ℕ) : ℕ :=
  total_needed - in_storage

/-- Theorem stating that given Charlie's specific situation, 
    he needs to buy 359 lbs of additional metal. -/
theorem charlie_metal_purchase : 
  additional_metal_needed 635 276 = 359 := by sorry

end charlie_metal_purchase_l450_45073


namespace regular_polygon_with_140_degree_interior_angles_is_nonagon_l450_45055

theorem regular_polygon_with_140_degree_interior_angles_is_nonagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 140 →
    (n - 2) * 180 = n * interior_angle →
    n = 9 :=
by sorry

end regular_polygon_with_140_degree_interior_angles_is_nonagon_l450_45055


namespace parabola_focus_focus_of_specific_parabola_l450_45088

/-- The focus of a parabola y = ax^2 + k is at (0, 1/(4a) + k) -/
theorem parabola_focus (a k : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1/(4*a) + k)
  ∀ x y : ℝ, y = a * x^2 + k → (x - f.1)^2 + (y - f.2)^2 = (y - k + 1/(4*a))^2 :=
sorry

/-- The focus of the parabola y = 9x^2 + 6 is at (0, 217/36) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, 217/36)
  ∀ x y : ℝ, y = 9 * x^2 + 6 → (x - f.1)^2 + (y - f.2)^2 = (y - 6 + 1/36)^2 :=
sorry

end parabola_focus_focus_of_specific_parabola_l450_45088


namespace triangle_area_change_l450_45039

theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = 1.10 * base) 
  (h2 : height_new = 0.95 * height) 
  (h3 : area = (base * height) / 2) 
  (h4 : area_new = (base_new * height_new) / 2) :
  area_new = 1.045 * area := by
  sorry

#check triangle_area_change

end triangle_area_change_l450_45039


namespace basic_astrophysics_degrees_l450_45054

/-- Represents the allocation of a research and development budget in a circle graph -/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ

/-- The theorem stating that the remaining sector (basic astrophysics) occupies 90 degrees of the circle -/
theorem basic_astrophysics_degrees (budget : BudgetAllocation) : 
  budget.microphotonics = 14 ∧ 
  budget.home_electronics = 19 ∧ 
  budget.food_additives = 10 ∧ 
  budget.genetically_modified_microorganisms = 24 ∧ 
  budget.industrial_lubricants = 8 → 
  (100 - (budget.microphotonics + budget.home_electronics + budget.food_additives + 
          budget.genetically_modified_microorganisms + budget.industrial_lubricants)) / 100 * 360 = 90 :=
by sorry

end basic_astrophysics_degrees_l450_45054


namespace age_difference_l450_45020

theorem age_difference (father_age : ℕ) (son_age_5_years_ago : ℕ) :
  father_age = 38 →
  son_age_5_years_ago = 14 →
  father_age - (son_age_5_years_ago + 5) = 19 := by
  sorry

end age_difference_l450_45020


namespace min_value_of_expression_l450_45081

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a/(a-1) + 4*b/(b-1) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ a₀/(a₀-1) + 4*b₀/(b₀-1) = 9 :=
sorry

end min_value_of_expression_l450_45081


namespace cary_needs_14_weekends_l450_45078

/-- Calculates the number of weekends Cary needs to mow lawns to afford discounted shoes --/
def weekends_needed (
  normal_cost : ℚ
  ) (discount_percent : ℚ
  ) (saved : ℚ
  ) (bus_expense : ℚ
  ) (earnings_per_lawn : ℚ
  ) (lawns_per_weekend : ℕ
  ) : ℕ :=
  sorry

/-- Theorem stating that Cary needs 14 weekends to afford the discounted shoes --/
theorem cary_needs_14_weekends :
  weekends_needed 120 20 30 10 5 3 = 14 :=
  sorry

end cary_needs_14_weekends_l450_45078


namespace nucleic_acid_test_is_comprehensive_l450_45060

/-- Represents a survey method -/
inductive SurveyMethod
| BallpointPenRefills
| FoodProducts
| CarCrashResistance
| NucleicAcidTest

/-- Predicate to determine if a survey method destroys its subjects -/
def destroysSubjects (method : SurveyMethod) : Prop :=
  match method with
  | SurveyMethod.BallpointPenRefills => true
  | SurveyMethod.FoodProducts => true
  | SurveyMethod.CarCrashResistance => true
  | SurveyMethod.NucleicAcidTest => false

/-- Definition of a comprehensive survey -/
def isComprehensiveSurvey (method : SurveyMethod) : Prop :=
  ¬(destroysSubjects method)

/-- Theorem: Nucleic Acid Test is suitable for a comprehensive survey -/
theorem nucleic_acid_test_is_comprehensive :
  isComprehensiveSurvey SurveyMethod.NucleicAcidTest :=
by
  sorry

#check nucleic_acid_test_is_comprehensive

end nucleic_acid_test_is_comprehensive_l450_45060


namespace total_profit_is_840_l450_45061

/-- Represents the investment and profit details of a business partnership --/
structure BusinessPartnership where
  initial_investment_A : ℕ
  initial_investment_B : ℕ
  withdrawal_A : ℕ
  addition_B : ℕ
  months_before_change : ℕ
  total_months : ℕ
  profit_share_A : ℕ

/-- Calculates the total profit given the business partnership details --/
def calculate_total_profit (bp : BusinessPartnership) : ℕ :=
  sorry

/-- Theorem stating that given the specific investment pattern, if A's profit share is 320,
    then the total profit is 840 --/
theorem total_profit_is_840 (bp : BusinessPartnership)
  (h1 : bp.initial_investment_A = 3000)
  (h2 : bp.initial_investment_B = 4000)
  (h3 : bp.withdrawal_A = 1000)
  (h4 : bp.addition_B = 1000)
  (h5 : bp.months_before_change = 8)
  (h6 : bp.total_months = 12)
  (h7 : bp.profit_share_A = 320) :
  calculate_total_profit bp = 840 :=
  sorry

end total_profit_is_840_l450_45061


namespace clock_coincidences_l450_45097

/-- Represents a clock with minute and hour hands -/
structure Clock :=
  (minuteRotations : ℕ) -- Number of full rotations of minute hand in 12 hours
  (hourRotations : ℕ)   -- Number of full rotations of hour hand in 12 hours

/-- The standard 12-hour clock -/
def standardClock : Clock :=
  { minuteRotations := 12,
    hourRotations := 1 }

/-- Number of coincidences between minute and hour hands in 12 hours -/
def coincidences (c : Clock) : ℕ :=
  c.minuteRotations - c.hourRotations

/-- Interval between coincidences in minutes -/
def coincidenceInterval (c : Clock) : ℚ :=
  (12 * 60) / (coincidences c)

theorem clock_coincidences (c : Clock) :
  c = standardClock →
  coincidences c = 11 ∧
  coincidenceInterval c = 65 + 5/11 :=
sorry

end clock_coincidences_l450_45097


namespace sufficient_not_necessary_condition_l450_45082

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 3 → x^2 > 4) ∧ 
  (∃ x : ℝ, x^2 > 4 ∧ ¬(x > 3)) :=
by sorry

end sufficient_not_necessary_condition_l450_45082


namespace first_customer_headphones_l450_45018

/-- The cost of one MP3 player -/
def mp3_cost : ℕ := 120

/-- The cost of one set of headphones -/
def headphone_cost : ℕ := 30

/-- The number of MP3 players bought by the first customer -/
def first_customer_mp3 : ℕ := 5

/-- The total cost of the first customer's purchase -/
def first_customer_total : ℕ := 840

/-- The number of MP3 players bought by the second customer -/
def second_customer_mp3 : ℕ := 3

/-- The number of headphone sets bought by the second customer -/
def second_customer_headphones : ℕ := 4

/-- The total cost of the second customer's purchase -/
def second_customer_total : ℕ := 480

theorem first_customer_headphones :
  ∃ h : ℕ, first_customer_mp3 * mp3_cost + h * headphone_cost = first_customer_total ∧
          h = 8 :=
by sorry

end first_customer_headphones_l450_45018


namespace rug_overlap_problem_l450_45076

/-- Given three rugs with total area A, overlapped to cover floor area F,
    with S2 area covered by exactly two layers, prove the area S3
    covered by three layers. -/
theorem rug_overlap_problem (A F S2 : ℝ) (hA : A = 200) (hF : F = 138) (hS2 : S2 = 24) :
  ∃ S1 S3 : ℝ,
    S1 + S2 + S3 = F ∧
    S1 + 2 * S2 + 3 * S3 = A ∧
    S3 = 19 := by
  sorry

end rug_overlap_problem_l450_45076


namespace max_area_right_quadrilateral_in_circle_l450_45046

/-- 
Given a circle with radius r, prove that the area of a right quadrilateral inscribed in the circle 
with one side tangent to the circle and one side a chord of the circle is maximized when the 
distance from the center of the circle to the midpoint of the chord is r/2.
-/
theorem max_area_right_quadrilateral_in_circle (r : ℝ) (h : r > 0) :
  ∃ (x y : ℝ),
    x^2 + y^2 = r^2 ∧  -- Pythagorean theorem for right triangle OCE
    (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 → (x + r) * y ≥ (x' + r) * y') ∧  -- Area maximization condition
    x = r / 2  -- The distance that maximizes the area
  := by sorry

end max_area_right_quadrilateral_in_circle_l450_45046


namespace second_largest_is_seven_l450_45066

def numbers : Finset ℕ := {5, 8, 4, 3, 7}

theorem second_largest_is_seven :
  ∃ (x : ℕ), x ∈ numbers ∧ x > 7 ∧ ∀ y ∈ numbers, y ≠ x → y ≤ 7 :=
by sorry

end second_largest_is_seven_l450_45066


namespace marbles_left_l450_45023

def initial_marbles : ℕ := 350
def marbles_given : ℕ := 175

theorem marbles_left : initial_marbles - marbles_given = 175 := by
  sorry

end marbles_left_l450_45023


namespace quadratic_roots_sum_l450_45036

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 - 3*α - 1 = 0) → 
  (β^2 - 3*β - 1 = 0) → 
  7 * α^4 + 10 * β^3 = 1093 := by
  sorry

end quadratic_roots_sum_l450_45036


namespace apple_pie_price_per_pound_l450_45008

/-- Given the following conditions for an apple pie:
  - The pie serves 8 people
  - 2 pounds of apples are needed
  - Pre-made pie crust costs $2.00
  - Lemon costs $0.50
  - Butter costs $1.50
  - Each serving of pie costs $1
  Prove that the price per pound of apples is $2.00 -/
theorem apple_pie_price_per_pound (servings : ℕ) (apple_pounds : ℝ) 
  (crust_cost lemon_cost butter_cost serving_cost : ℝ) :
  servings = 8 → 
  apple_pounds = 2 → 
  crust_cost = 2 → 
  lemon_cost = 0.5 → 
  butter_cost = 1.5 → 
  serving_cost = 1 → 
  (servings * serving_cost - (crust_cost + lemon_cost + butter_cost)) / apple_pounds = 2 :=
by sorry


end apple_pie_price_per_pound_l450_45008


namespace candy_probability_l450_45021

/-- The number of red candies initially in the jar -/
def red_candies : ℕ := 15

/-- The number of blue candies initially in the jar -/
def blue_candies : ℕ := 15

/-- The total number of candies initially in the jar -/
def total_candies : ℕ := red_candies + blue_candies

/-- The number of candies Terry picks -/
def terry_picks : ℕ := 3

/-- The number of candies Mary picks -/
def mary_picks : ℕ := 2

/-- The probability that Terry and Mary pick candies of the same color -/
def same_color_probability : ℚ := 8008 / 142221

theorem candy_probability : 
  same_color_probability = 
    (Nat.choose red_candies terry_picks * Nat.choose (red_candies - terry_picks) mary_picks + 
     Nat.choose blue_candies terry_picks * Nat.choose (blue_candies - terry_picks) mary_picks) / 
    (Nat.choose total_candies terry_picks * Nat.choose (total_candies - terry_picks) mary_picks) :=
sorry

end candy_probability_l450_45021


namespace geometric_sequence_with_unit_modulus_ratio_l450_45002

theorem geometric_sequence_with_unit_modulus_ratio (α : ℝ) : 
  let a : ℕ → ℂ := λ n => Complex.cos (n * α) + Complex.I * Complex.sin (n * α)
  ∃ r : ℂ, (∀ n : ℕ, a (n + 1) = r * a n) ∧ Complex.abs r = 1 :=
by
  sorry

end geometric_sequence_with_unit_modulus_ratio_l450_45002


namespace banana_permutations_l450_45035

/-- The number of distinct permutations of letters in a word with repeated letters -/
def distinctPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The number of distinct permutations of the letters in "BANANA" -/
theorem banana_permutations :
  distinctPermutations 6 [3, 2] = 60 := by
  sorry

end banana_permutations_l450_45035


namespace temporary_worker_percentage_is_18_l450_45007

/-- Represents the composition of workers in a factory -/
structure WorkerComposition where
  total : ℝ
  technician_ratio : ℝ
  non_technician_ratio : ℝ
  permanent_technician_ratio : ℝ
  permanent_non_technician_ratio : ℝ
  (total_positive : total > 0)
  (technician_ratio_valid : technician_ratio ≥ 0 ∧ technician_ratio ≤ 1)
  (non_technician_ratio_valid : non_technician_ratio ≥ 0 ∧ non_technician_ratio ≤ 1)
  (ratios_sum_to_one : technician_ratio + non_technician_ratio = 1)
  (permanent_technician_ratio_valid : permanent_technician_ratio ≥ 0 ∧ permanent_technician_ratio ≤ 1)
  (permanent_non_technician_ratio_valid : permanent_non_technician_ratio ≥ 0 ∧ permanent_non_technician_ratio ≤ 1)

/-- Calculates the percentage of temporary workers in the factory -/
def temporaryWorkerPercentage (w : WorkerComposition) : ℝ :=
  (1 - (w.technician_ratio * w.permanent_technician_ratio + w.non_technician_ratio * w.permanent_non_technician_ratio)) * 100

/-- Theorem stating that given the specific worker composition, the percentage of temporary workers is 18% -/
theorem temporary_worker_percentage_is_18 (w : WorkerComposition)
  (h1 : w.technician_ratio = 0.9)
  (h2 : w.non_technician_ratio = 0.1)
  (h3 : w.permanent_technician_ratio = 0.9)
  (h4 : w.permanent_non_technician_ratio = 0.1) :
  temporaryWorkerPercentage w = 18 := by
  sorry


end temporary_worker_percentage_is_18_l450_45007


namespace max_profit_l450_45067

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total number of cars sold across both locations -/
def total_cars : ℝ := 15

/-- Total profit function -/
def L (x : ℝ) : ℝ := L₁ x + L₂ (total_cars - x)

theorem max_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ total_cars ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ total_cars → L y ≤ L x ∧ L x = 45.6 :=
sorry

end max_profit_l450_45067


namespace regular_polygon_exterior_angle_l450_45025

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 36 → n = 10 := by
  sorry

end regular_polygon_exterior_angle_l450_45025


namespace square_of_8y_minus_2_l450_45032

theorem square_of_8y_minus_2 (y : ℝ) (h : 4 * y^2 + 7 = 6 * y + 12) :
  (8 * y - 2)^2 = 248 := by
  sorry

end square_of_8y_minus_2_l450_45032


namespace regular_polygon_with_40_degree_exterior_angle_has_9_sides_l450_45026

/-- A regular polygon with an exterior angle of 40° has 9 sides. -/
theorem regular_polygon_with_40_degree_exterior_angle_has_9_sides :
  ∀ (n : ℕ), n > 0 →
  (360 : ℝ) / n = 40 →
  n = 9 := by
  sorry

end regular_polygon_with_40_degree_exterior_angle_has_9_sides_l450_45026


namespace simplify_negative_cube_squared_l450_45031

theorem simplify_negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end simplify_negative_cube_squared_l450_45031


namespace zoo_revenue_calculation_l450_45056

/-- Calculates the total revenue for a zoo over two days with given attendance and pricing information --/
def zoo_revenue (
  monday_children monday_adults monday_seniors : ℕ)
  (tuesday_children tuesday_adults tuesday_seniors : ℕ)
  (monday_child_price monday_adult_price monday_senior_price : ℚ)
  (tuesday_child_price tuesday_adult_price tuesday_senior_price : ℚ)
  (tuesday_discount : ℚ) : ℚ :=
  let monday_total := 
    monday_children * monday_child_price + 
    monday_adults * monday_adult_price + 
    monday_seniors * monday_senior_price
  let tuesday_total := 
    tuesday_children * tuesday_child_price + 
    tuesday_adults * tuesday_adult_price + 
    tuesday_seniors * tuesday_senior_price
  let tuesday_discounted := tuesday_total * (1 - tuesday_discount)
  monday_total + tuesday_discounted

theorem zoo_revenue_calculation : 
  zoo_revenue 7 5 3 9 6 2 3 4 3 4 5 3 (1/10) = 114.8 := by
  sorry

end zoo_revenue_calculation_l450_45056


namespace subset_implies_m_leq_two_l450_45011

def A : Set ℝ := {x | x < 2}
def B (m : ℝ) : Set ℝ := {x | x < m}

theorem subset_implies_m_leq_two (m : ℝ) : B m ⊆ A → m ≤ 2 := by
  sorry

end subset_implies_m_leq_two_l450_45011


namespace max_mn_and_min_4m2_plus_n2_l450_45042

theorem max_mn_and_min_4m2_plus_n2 (m n : ℝ) 
  (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → m * n ≥ x * y) ∧
  (m * n = 1/8) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 4 * m^2 + n^2 ≤ 4 * x^2 + y^2) ∧
  (4 * m^2 + n^2 = 1/2) :=
by sorry

end max_mn_and_min_4m2_plus_n2_l450_45042


namespace pigeonhole_sum_to_ten_l450_45053

theorem pigeonhole_sum_to_ten :
  ∀ (S : Finset ℕ), 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 10) → 
    S.card ≥ 7 → 
    ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 10 :=
by sorry

end pigeonhole_sum_to_ten_l450_45053


namespace adult_ticket_price_l450_45004

/-- Proves that the price of an adult ticket is $15 given the conditions of the problem -/
theorem adult_ticket_price (total_cost : ℕ) (child_ticket_price : ℕ) (num_children : ℕ) :
  total_cost = 720 →
  child_ticket_price = 8 →
  num_children = 15 →
  ∃ (adult_ticket_price : ℕ),
    adult_ticket_price * (num_children + 25) + child_ticket_price * num_children = total_cost ∧
    adult_ticket_price = 15 := by
  sorry

end adult_ticket_price_l450_45004


namespace hyperbola_equation_l450_45089

/-- A hyperbola with center at the origin, transverse axis on the y-axis, and one focus at (0, 6) -/
structure Hyperbola where
  center : ℝ × ℝ
  transverse_axis : ℝ → ℝ × ℝ
  focus : ℝ × ℝ
  h_center : center = (0, 0)
  h_transverse : ∀ x, transverse_axis x = (0, x)
  h_focus : focus = (0, 6)

/-- The equation of the hyperbola is y^2 - x^2 = 18 -/
theorem hyperbola_equation (h : Hyperbola) : 
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | y^2 - x^2 = 18} ↔ 
  ∃ t : ℝ, h.transverse_axis t = (x, y) :=
sorry

end hyperbola_equation_l450_45089


namespace tournament_probability_l450_45041

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : ℕ
  games_per_team : ℕ
  win_probability : ℝ

/-- Calculates the probability of team A finishing with more points than team B -/
def probability_A_beats_B (tournament : SoccerTournament) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem tournament_probability : 
  let tournament := SoccerTournament.mk 7 6 (1/2)
  probability_A_beats_B tournament = 319/512 := by sorry

end tournament_probability_l450_45041


namespace exists_valid_grid_l450_45019

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if all elements in a list are equal -/
def allEqual (l : List ℕ) : Prop :=
  l.all (· = l.head!)

/-- Checks if a grid satisfies the required properties -/
def validGrid (g : Grid) : Prop :=
  -- 0 is in the central position
  g 1 1 = 0 ∧
  -- Digits 1-8 are used exactly once each in the remaining positions
  (∀ i : Fin 9, i ≠ 0 → ∃! (r c : Fin 3), g r c = i.val) ∧
  -- The sum of digits in each row and each column is the same
  allEqual [
    g 0 0 + g 0 1 + g 0 2,
    g 1 0 + g 1 1 + g 1 2,
    g 2 0 + g 2 1 + g 2 2,
    g 0 0 + g 1 0 + g 2 0,
    g 0 1 + g 1 1 + g 2 1,
    g 0 2 + g 1 2 + g 2 2
  ]

/-- There exists a grid satisfying the required properties -/
theorem exists_valid_grid : ∃ g : Grid, validGrid g := by
  sorry

end exists_valid_grid_l450_45019


namespace nearest_integer_to_3_plus_sqrt5_4th_power_l450_45003

theorem nearest_integer_to_3_plus_sqrt5_4th_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5) ^ 4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5) ^ 4 - (m : ℝ)| :=
by sorry

end nearest_integer_to_3_plus_sqrt5_4th_power_l450_45003


namespace ninth_term_is_15_l450_45006

/-- An arithmetic sequence with properties S3 = 3 and S6 = 24 -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n / 2) * (a 1 + a n)
  S3_eq_3 : S 3 = 3
  S6_eq_24 : S 6 = 24

/-- The 9th term of the arithmetic sequence is 15 -/
theorem ninth_term_is_15 (seq : ArithmeticSequence) : seq.a 9 = 15 := by
  sorry

end ninth_term_is_15_l450_45006


namespace sphere_radius_l450_45091

theorem sphere_radius (A : ℝ) (h : A = 64 * Real.pi) :
  ∃ (r : ℝ), A = 4 * Real.pi * r^2 ∧ r = 4 := by
  sorry

end sphere_radius_l450_45091


namespace origin_outside_circle_iff_a_in_range_l450_45062

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*y + a - 2 = 0

/-- A point is outside a circle if the left side of the equation is positive when substituting the point's coordinates -/
def point_outside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*y + a - 2 > 0

theorem origin_outside_circle_iff_a_in_range (a : ℝ) :
  point_outside_circle 0 0 a ↔ 2 < a ∧ a < 3 :=
sorry

end origin_outside_circle_iff_a_in_range_l450_45062


namespace james_excess_calories_james_specific_excess_calories_l450_45001

/-- Calculates the excess calories eaten by James after eating Cheezits and going for a run. -/
theorem james_excess_calories (bags : Nat) (ounces_per_bag : Nat) (calories_per_ounce : Nat) 
  (run_duration : Nat) (calories_burned_per_minute : Nat) : Nat :=
  let total_calories_consumed := bags * ounces_per_bag * calories_per_ounce
  let total_calories_burned := run_duration * calories_burned_per_minute
  total_calories_consumed - total_calories_burned

/-- Proves that James ate 420 excess calories given the specific conditions. -/
theorem james_specific_excess_calories :
  james_excess_calories 3 2 150 40 12 = 420 := by
  sorry

end james_excess_calories_james_specific_excess_calories_l450_45001


namespace betty_age_l450_45087

/-- Represents the ages of Albert, Mary, and Betty --/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions given in the problem --/
def age_conditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 22

/-- The theorem stating Betty's age --/
theorem betty_age (ages : Ages) (h : age_conditions ages) : ages.betty = 11 := by
  sorry

end betty_age_l450_45087


namespace power_of_three_mod_eight_l450_45033

theorem power_of_three_mod_eight : 3^2023 % 8 = 3 := by
  sorry

end power_of_three_mod_eight_l450_45033


namespace followers_exceed_thousand_l450_45029

/-- 
Given that Daniel starts with 5 followers on Sunday and his followers triple each day,
this theorem proves that Saturday (6 days after Sunday) is the first day 
when Daniel has more than 1000 followers.
-/
theorem followers_exceed_thousand (k : ℕ) : 
  (∀ n < k, 5 * 3^n ≤ 1000) ∧ 5 * 3^k > 1000 → k = 6 := by
  sorry

end followers_exceed_thousand_l450_45029


namespace delores_initial_amount_l450_45095

/-- The amount of money Delores had initially -/
def initial_amount : ℕ := sorry

/-- The cost of the computer -/
def computer_cost : ℕ := 400

/-- The cost of the printer -/
def printer_cost : ℕ := 40

/-- The amount of money Delores had left after purchases -/
def remaining_amount : ℕ := 10

/-- Theorem stating that Delores' initial amount was $450 -/
theorem delores_initial_amount : 
  initial_amount = computer_cost + printer_cost + remaining_amount := by sorry

end delores_initial_amount_l450_45095


namespace diana_reading_time_l450_45094

/-- The number of hours Diana read this week -/
def hours_read : ℝ := 12

/-- The initial reward rate in minutes per hour -/
def initial_rate : ℝ := 30

/-- The percentage increase in the reward rate -/
def rate_increase : ℝ := 0.2

/-- The total increase in video game time due to the raise in minutes -/
def total_increase : ℝ := 72

theorem diana_reading_time :
  hours_read * initial_rate * rate_increase = total_increase := by
  sorry

end diana_reading_time_l450_45094


namespace quadratic_function_problem_l450_45074

/-- Given a quadratic function f(x) = x^2 + ax + b, if f(f(x) + x) / f(x) = x^2 + 2023x + 3000,
    then a = 2021 and b = 979. -/
theorem quadratic_function_problem (a b : ℝ) : 
  (let f := fun x => x^2 + a*x + b
   (∀ x, (f (f x + x)) / (f x) = x^2 + 2023*x + 3000)) → 
  (a = 2021 ∧ b = 979) := by
  sorry

end quadratic_function_problem_l450_45074


namespace max_abc_value_l450_45071

theorem max_abc_value (a b c : ℕ+) 
  (h1 : a * b + b * c = 518)
  (h2 : a * b - a * c = 360) :
  ∀ x y z : ℕ+, x * y * z ≤ a * b * c → x * y + y * z = 518 → x * y - x * z = 360 → 
  a * b * c = 1008 := by
sorry

end max_abc_value_l450_45071


namespace four_heads_before_three_tails_l450_45096

/-- The probability of encountering 4 consecutive heads before 3 consecutive tails in repeated fair coin flips -/
def q : ℚ := 16/23

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℚ → Prop) : Prop := p (1/2)

theorem four_heads_before_three_tails (fair_coin : (ℚ → Prop) → Prop) : q = 16/23 := by
  sorry

end four_heads_before_three_tails_l450_45096


namespace sum_of_ages_matt_age_relation_l450_45051

/-- Given Matt's age and John's age, prove the sum of their ages -/
theorem sum_of_ages (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  matt_age + john_age = 52 := by
  sorry

/-- Matt's age in relation to John's -/
theorem matt_age_relation (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  matt_age = 4 * john_age - 3 := by
  sorry

end sum_of_ages_matt_age_relation_l450_45051


namespace plane_equation_is_correct_l450_45063

/-- The line passing through (2,4,-3) with direction vector (4,-1,5) -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (2 + 4*t, 4 - t, -3 + 5*t)

/-- The point that the plane passes through -/
def point : ℝ × ℝ × ℝ := (1, 6, -8)

/-- The coefficients of the plane equation -/
def plane_coeff : ℤ × ℤ × ℤ × ℤ := (5, 15, -7, 151)

theorem plane_equation_is_correct :
  let (A, B, C, D) := plane_coeff
  (A > 0) ∧ 
  (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1) ∧
  (∀ (x y z : ℝ), A * x + B * y + C * z - D = 0 ↔ 
    (∃ (t : ℝ), (x, y, z) = line t) ∨ (x, y, z) = point) := by sorry

end plane_equation_is_correct_l450_45063


namespace apple_basket_problem_l450_45037

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem apple_basket_problem (total_apples : ℕ) (first_basket : ℕ) (increment : ℕ) :
  total_apples = 495 →
  first_basket = 25 →
  increment = 2 →
  ∃ x : ℕ, x = 13 ∧ arithmetic_sum first_basket increment x = total_apples :=
by sorry

end apple_basket_problem_l450_45037


namespace amusement_park_problem_l450_45050

/-- The number of children in the group satisfies the given conditions -/
theorem amusement_park_problem (C : ℕ) : C = 5 ↔ 
  15 + 3 * C + 16 * C = 110 := by
  sorry

end amusement_park_problem_l450_45050


namespace original_selling_price_l450_45028

theorem original_selling_price 
  (P : ℝ) -- Original purchase price
  (S : ℝ) -- Original selling price
  (S_new : ℝ) -- New selling price
  (h1 : S = 1.1 * P) -- Original selling price is 110% of purchase price
  (h2 : S_new = 1.17 * P) -- New selling price based on 10% lower purchase and 30% profit
  (h3 : S_new - S = 63) -- Difference between new and original selling price is $63
  : S = 990 := by
sorry

end original_selling_price_l450_45028


namespace large_triangle_toothpicks_l450_45038

/-- The number of small triangles in the base row of the large equilateral triangle -/
def base_triangles : ℕ := 100

/-- The total number of small triangles in the large equilateral triangle -/
def total_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks required to assemble the large equilateral triangle -/
def toothpicks_required : ℕ := ((3 * total_triangles) / 2) + (3 * base_triangles)

theorem large_triangle_toothpicks :
  toothpicks_required = 7875 := by sorry

end large_triangle_toothpicks_l450_45038


namespace f_geq_a_iff_a_in_range_l450_45034

/-- The function f(x) = x^2 - 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

/-- The domain of x -/
def domain : Set ℝ := {x | x ≥ -1}

/-- The theorem stating the condition for a -/
theorem f_geq_a_iff_a_in_range (a : ℝ) : 
  (∀ x ∈ domain, f a x ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

end f_geq_a_iff_a_in_range_l450_45034


namespace arccos_sin_one_l450_45086

theorem arccos_sin_one : Real.arccos (Real.sin 1) = π / 2 - 1 := by
  sorry

end arccos_sin_one_l450_45086


namespace max_value_squared_l450_45017

theorem max_value_squared (a b x y : ℝ) : 
  a > 0 → b > 0 → a ≥ b → 
  0 ≤ x → x < a → 
  0 ≤ y → y < b → 
  a^2 + y^2 = b^2 + x^2 → 
  (a - x)^2 + (b + y)^2 = b^2 + x^2 →
  x = a - 2*b →
  y = b/2 →
  (∀ ρ : ℝ, (a/b)^2 ≤ ρ^2 → ρ^2 ≤ 4/9) :=
by sorry

end max_value_squared_l450_45017


namespace volunteer_assignment_count_l450_45079

theorem volunteer_assignment_count : 
  (volunteers : ℕ) → 
  (pavilions : ℕ) → 
  volunteers = 5 → 
  pavilions = 4 → 
  (arrangements : ℕ) → 
  arrangements = 240 := by sorry

end volunteer_assignment_count_l450_45079


namespace rhombus_diagonals_bisect_l450_45000

-- Define the property of diagonals bisecting each other
def diagonals_bisect (shape : Type) : Prop := sorry

-- Define the relationship between rhombus and parallelogram
def rhombus_is_parallelogram : Prop := sorry

-- Theorem statement
theorem rhombus_diagonals_bisect :
  diagonals_bisect Parallelogram →
  rhombus_is_parallelogram →
  diagonals_bisect Rhombus := by
  sorry

end rhombus_diagonals_bisect_l450_45000


namespace fourth_arrangement_follows_pattern_l450_45014

/-- Represents the four possible positions in a 2x2 grid --/
inductive Position
| topLeft
| topRight
| bottomLeft
| bottomRight

/-- Represents the orientation of the line segment --/
inductive LineOrientation
| horizontal
| vertical

/-- Represents a geometric shape --/
inductive Shape
| circle
| triangle
| square
| line

/-- Represents the arrangement of shapes in a square --/
structure Arrangement where
  circlePos : Position
  trianglePos : Position
  squarePos : Position
  lineOrientation : LineOrientation

/-- The sequence of arrangements in the first three squares --/
def firstThreeArrangements : List Arrangement := [
  { circlePos := Position.topLeft, trianglePos := Position.bottomLeft, 
    squarePos := Position.topRight, lineOrientation := LineOrientation.horizontal },
  { circlePos := Position.bottomLeft, trianglePos := Position.bottomRight, 
    squarePos := Position.topRight, lineOrientation := LineOrientation.vertical },
  { circlePos := Position.bottomRight, trianglePos := Position.topRight, 
    squarePos := Position.bottomLeft, lineOrientation := LineOrientation.horizontal }
]

/-- The predicted arrangement for the fourth square --/
def predictedFourthArrangement : Arrangement :=
  { circlePos := Position.topRight, trianglePos := Position.topLeft, 
    squarePos := Position.bottomLeft, lineOrientation := LineOrientation.vertical }

/-- Theorem stating that the predicted fourth arrangement follows the pattern --/
theorem fourth_arrangement_follows_pattern :
  predictedFourthArrangement = 
    { circlePos := Position.topRight, trianglePos := Position.topLeft, 
      squarePos := Position.bottomLeft, lineOrientation := LineOrientation.vertical } :=
by sorry

end fourth_arrangement_follows_pattern_l450_45014


namespace largest_multiple_of_seven_below_negative_fiftyfive_l450_45012

theorem largest_multiple_of_seven_below_negative_fiftyfive :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -55 → n ≤ -56 :=
by sorry

end largest_multiple_of_seven_below_negative_fiftyfive_l450_45012


namespace unique_m_opens_downwards_l450_45059

/-- A function f(x) = (m + 1)x^(|m|) that opens downwards -/
def opens_downwards (m : ℝ) : Prop :=
  (abs m = 2) ∧ (m + 1 < 0)

/-- The unique value of m for which the function opens downwards is -2 -/
theorem unique_m_opens_downwards :
  ∃! m : ℝ, opens_downwards m :=
sorry

end unique_m_opens_downwards_l450_45059


namespace trapezoid_pq_length_l450_45084

/-- Represents a trapezoid ABCD with a parallel line PQ intersecting diagonals -/
structure Trapezoid :=
  (a : ℝ) -- Length of base BC
  (b : ℝ) -- Length of base AD
  (pl : ℝ) -- Length of PL
  (lr : ℝ) -- Length of LR

/-- The main theorem about the length of PQ in a trapezoid -/
theorem trapezoid_pq_length (t : Trapezoid) (h : t.pl = t.lr) :
  ∃ (pq : ℝ), pq = (3 * t.a * t.b) / (2 * t.a + t.b) ∨ pq = (3 * t.a * t.b) / (t.a + 2 * t.b) :=
sorry

end trapezoid_pq_length_l450_45084


namespace puzzle_pieces_count_l450_45044

theorem puzzle_pieces_count :
  let border_pieces : ℕ := 75
  let trevor_pieces : ℕ := 105
  let joe_pieces : ℕ := 3 * trevor_pieces
  let missing_pieces : ℕ := 5
  let total_pieces : ℕ := border_pieces + trevor_pieces + joe_pieces + missing_pieces
  total_pieces = 500 := by
sorry

end puzzle_pieces_count_l450_45044


namespace system_solution_l450_45057

theorem system_solution : ∃ (a b c : ℝ), 
  (a^2 * b^2 - a^2 - a*b + 1 = 0) ∧ 
  (a^2 * c - a*b - a - c = 0) ∧ 
  (a*b*c = -1) ∧ 
  (a = -1) ∧ (b = -1) ∧ (c = -1) := by
  sorry

end system_solution_l450_45057


namespace expression_value_l450_45098

theorem expression_value : -25 + 5 * (4^2 / 2) = 15 := by
  sorry

end expression_value_l450_45098


namespace cyclic_quadrilateral_diagonal_intersection_property_l450_45099

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a 2D plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a quadrilateral is cyclic -/
def is_cyclic (q : Quadrilateral) (c : Circle) : Prop :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Definition omitted for brevity
  sorry

/-- The main theorem -/
theorem cyclic_quadrilateral_diagonal_intersection_property
  (ABCD : Quadrilateral) (c : Circle) (X : Point) :
  is_cyclic ABCD c →
  X = intersection ABCD.A ABCD.C ABCD.B ABCD.D →
  distance X ABCD.A * distance X ABCD.C = distance X ABCD.B * distance X ABCD.D :=
by
  sorry

end cyclic_quadrilateral_diagonal_intersection_property_l450_45099


namespace partition_uniqueness_l450_45005

/-- An arithmetic progression is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticProgression (a : ℤ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℤ, a (n + 1) = a n + d

/-- A set of integers X can be partitioned into N disjoint increasing
    arithmetic progressions. -/
def CanBePartitioned (X : Set ℤ) (N : ℕ) : Prop :=
  ∃ (partitions : Fin N → Set ℤ),
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ partitions i = Set.range a) ∧
    (∀ i j : Fin N, i ≠ j → partitions i ∩ partitions j = ∅) ∧
    (⋃ i : Fin N, partitions i) = X

/-- X cannot be partitioned into fewer than N arithmetic progressions. -/
def MinimalPartition (X : Set ℤ) (N : ℕ) : Prop :=
  CanBePartitioned X N ∧ ∀ k < N, ¬CanBePartitioned X k

/-- The partition of X into N arithmetic progressions is unique. -/
def UniquePartition (X : Set ℤ) (N : ℕ) : Prop :=
  ∀ p₁ p₂ : Fin N → Set ℤ,
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ p₁ i = Set.range a) →
    (∀ i : Fin N, ∃ a : ℤ → ℤ, ArithmeticProgression a ∧ p₂ i = Set.range a) →
    (∀ i j : Fin N, i ≠ j → p₁ i ∩ p₁ j = ∅) →
    (∀ i j : Fin N, i ≠ j → p₂ i ∩ p₂ j = ∅) →
    (⋃ i : Fin N, p₁ i) = X →
    (⋃ i : Fin N, p₂ i) = X →
    ∀ i : Fin N, ∃ j : Fin N, p₁ i = p₂ j

theorem partition_uniqueness (X : Set ℤ) :
  (∀ N : ℕ, MinimalPartition X N → (N = 2 → UniquePartition X N) ∧ (N = 3 → ¬UniquePartition X N)) := by
  sorry

end partition_uniqueness_l450_45005


namespace container_capacity_l450_45016

theorem container_capacity : ∀ (C : ℝ), 
  (0.30 * C + 27 = 0.75 * C) → C = 60 :=
by
  sorry

end container_capacity_l450_45016


namespace six_solutions_l450_45070

/-- The number of ordered pairs of positive integers (m,n) satisfying 6/m + 3/n = 1 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (m, n) := p
    m > 0 ∧ n > 0 ∧ 6 * n + 3 * m = m * n) (Finset.product (Finset.range 25) (Finset.range 22))).card

/-- The theorem stating that there are exactly 6 solutions -/
theorem six_solutions : solution_count = 6 := by
  sorry


end six_solutions_l450_45070


namespace min_value_theorem_l450_45052

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_theorem_l450_45052
