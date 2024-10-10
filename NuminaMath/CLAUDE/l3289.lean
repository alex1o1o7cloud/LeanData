import Mathlib

namespace abs_neg_2023_l3289_328961

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_neg_2023_l3289_328961


namespace polygon_properties_l3289_328913

theorem polygon_properties :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    -- Condition: Each interior angle is 30° more than four times its adjacent exterior angle
    (180 : ℝ) = exterior_angle + 4 * exterior_angle + 30 →
    -- Condition: Sum of exterior angles is always 360°
    (n : ℝ) * exterior_angle = 360 →
    -- Conclusions
    n = 12 ∧
    (n - 2 : ℝ) * 180 = 1800 ∧
    n * (n - 3) / 2 = 54 :=
by
  sorry


end polygon_properties_l3289_328913


namespace abc_product_value_l3289_328932

theorem abc_product_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5) (eq2 : b + 1/c = 2) (eq3 : c + 1/a = 3) :
  a * b * c = 10 + 3 * Real.sqrt 11 := by
  sorry

end abc_product_value_l3289_328932


namespace arithmetic_sequence_first_term_l3289_328907

/-- Sum of first n terms of an arithmetic sequence -/
def T (b : ℚ) (n : ℕ) : ℚ := n * (2 * b + (n - 1) * 5) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term :
  (∃ (k : ℚ), ∀ (n : ℕ), n > 0 → T b (4 * n) / T b n = k) →
  b = 5 / 2 :=
sorry

end arithmetic_sequence_first_term_l3289_328907


namespace factors_of_2310_l3289_328994

theorem factors_of_2310 : Finset.card (Nat.divisors 2310) = 32 := by sorry

end factors_of_2310_l3289_328994


namespace line_tangent_to_circle_l3289_328955

-- Define the line l
def line_l (c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x + 2*y + c = 0

-- Define the circle C
def circle_C : ℝ → ℝ → Prop :=
  fun x y => x^2 + y^2 + 2*x - 4*y = 0

-- Define the translated line l'
def line_l_prime (c : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x + 2*y + c + 5 = 0

-- Define the tangency condition
def is_tangent (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ C x y ∧ ∀ x' y', l x' y' ∧ C x' y' → (x = x' ∧ y = y')

theorem line_tangent_to_circle (c : ℝ) :
  is_tangent (line_l_prime c) circle_C → c = -3 ∨ c = -13 :=
by sorry

end line_tangent_to_circle_l3289_328955


namespace inverse_of_B_cubed_l3289_328930

open Matrix

/-- Given a 2x2 matrix B with its inverse, prove that the inverse of B^3 is equal to B^(-1) -/
theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = ![![3, 4], ![-2, -3]]) : 
  (B^3)⁻¹ = ![![3, 4], ![-2, -3]] := by
  sorry

end inverse_of_B_cubed_l3289_328930


namespace abs_neg_nine_equals_nine_l3289_328953

theorem abs_neg_nine_equals_nine : abs (-9 : ℤ) = 9 := by
  sorry

end abs_neg_nine_equals_nine_l3289_328953


namespace right_triangle_tan_l3289_328937

theorem right_triangle_tan (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) :
  b = 12 ∧ a / b = 5 / 12 := by
sorry

end right_triangle_tan_l3289_328937


namespace lcm_minus_gcd_equals_34_l3289_328977

theorem lcm_minus_gcd_equals_34 : Nat.lcm 40 8 - Nat.gcd 24 54 = 34 := by
  sorry

end lcm_minus_gcd_equals_34_l3289_328977


namespace complex_expressions_equality_l3289_328919

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equality to be proved
theorem complex_expressions_equality :
  ((-1/2 : ℂ) + (Real.sqrt 3/2)*i) * (2 - i) * (3 + i) = 
    (-3/2 : ℂ) + (5*Real.sqrt 3/2) + ((7*Real.sqrt 3 + 1)/2)*i ∧
  ((Real.sqrt 2 + Real.sqrt 2*i)^2 * (4 + 5*i)) / ((5 - 4*i) * (1 - i)) = 
    (62/41 : ℂ) + (80/41)*i :=
by sorry

-- Axiom for i^2 = -1
axiom i_squared : i^2 = -1

end complex_expressions_equality_l3289_328919


namespace parallelogram_external_bisectors_rectangle_diagonal_l3289_328923

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is a rectangle -/
def isRectangle (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the length of a line segment between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Represents the intersection points of external angle bisectors -/
structure ExternalBisectorPoints :=
  (P Q R S : Point)

/-- Checks if given points are formed by intersection of external angle bisectors -/
def areExternalBisectorPoints (q : Quadrilateral) (e : ExternalBisectorPoints) : Prop :=
  sorry

/-- Main theorem -/
theorem parallelogram_external_bisectors_rectangle_diagonal
  (ABCD : Quadrilateral)
  (PQRS : ExternalBisectorPoints) :
  isParallelogram ABCD →
  areExternalBisectorPoints ABCD PQRS →
  isRectangle ⟨PQRS.P, PQRS.Q, PQRS.R, PQRS.S⟩ →
  distance PQRS.P PQRS.R = distance ABCD.A ABCD.B + distance ABCD.B ABCD.C :=
by sorry

end parallelogram_external_bisectors_rectangle_diagonal_l3289_328923


namespace rectangular_to_polar_conversion_l3289_328970

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ),
    x = 3 ∧ y = -3 →
    ∃ (r θ : ℝ),
      r > 0 ∧
      0 ≤ θ ∧ θ < 2 * Real.pi ∧
      r = 3 * Real.sqrt 2 ∧
      θ = 7 * Real.pi / 4 ∧
      x = r * Real.cos θ ∧
      y = r * Real.sin θ :=
by sorry

end rectangular_to_polar_conversion_l3289_328970


namespace distance_traveled_l3289_328979

theorem distance_traveled (initial_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) :
  initial_speed = 10 →
  increased_speed = 15 →
  additional_distance = 15 →
  ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = increased_speed * time ∧
    actual_distance = 30 :=
by sorry

end distance_traveled_l3289_328979


namespace probability_purple_face_l3289_328958

/-- The probability of rolling a purple face on a 10-sided die with 3 purple faces is 3/10. -/
theorem probability_purple_face (total_faces : ℕ) (purple_faces : ℕ) 
  (h1 : total_faces = 10) (h2 : purple_faces = 3) : 
  (purple_faces : ℚ) / total_faces = 3 / 10 := by
  sorry

end probability_purple_face_l3289_328958


namespace unique_solution_geometric_series_l3289_328901

theorem unique_solution_geometric_series :
  ∃! x : ℝ, |x| < 1 ∧ x = (1 : ℝ) / (1 + x) ∧ x = (-1 + Real.sqrt 5) / 2 := by
  sorry

end unique_solution_geometric_series_l3289_328901


namespace sqrt_x_minus_4_real_range_l3289_328948

theorem sqrt_x_minus_4_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by
  sorry

end sqrt_x_minus_4_real_range_l3289_328948


namespace mode_most_relevant_for_restocking_l3289_328952

/-- Represents a shoe size -/
def ShoeSize := ℕ

/-- Represents the inventory of shoes -/
def Inventory := List ShoeSize

/-- A statistical measure for shoe sizes -/
class StatisticalMeasure where
  measure : Inventory → ℝ

/-- Variance of shoe sizes -/
def variance : StatisticalMeasure := sorry

/-- Mode of shoe sizes -/
def mode : StatisticalMeasure := sorry

/-- Median of shoe sizes -/
def median : StatisticalMeasure := sorry

/-- Mean of shoe sizes -/
def mean : StatisticalMeasure := sorry

/-- Relevance of a statistical measure for restocking -/
def relevance (m : StatisticalMeasure) : ℝ := sorry

/-- The shoe store -/
structure ShoeStore where
  inventory : Inventory

/-- Theorem: Mode is the most relevant statistical measure for restocking -/
theorem mode_most_relevant_for_restocking (store : ShoeStore) :
  ∀ m : StatisticalMeasure, m ≠ mode → relevance mode > relevance m :=
sorry

end mode_most_relevant_for_restocking_l3289_328952


namespace probability_of_vowel_in_four_consecutive_letters_l3289_328911

/-- Represents the English alphabet --/
def Alphabet : Finset Char := sorry

/-- Represents the vowels in the English alphabet --/
def Vowels : Finset Char := sorry

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of vowels --/
def vowel_count : ℕ := 5

/-- The number of possible sets of 4 consecutive letters --/
def consecutive_sets : ℕ := 23

/-- The number of sets of 4 consecutive letters without a vowel --/
def sets_without_vowel : ℕ := 5

/-- Theorem: The probability of selecting at least one vowel when choosing 4 consecutive letters at random from the English alphabet is 18/23 --/
theorem probability_of_vowel_in_four_consecutive_letters :
  (consecutive_sets - sets_without_vowel : ℚ) / consecutive_sets = 18 / 23 :=
sorry

end probability_of_vowel_in_four_consecutive_letters_l3289_328911


namespace least_c_for_triple_f_l3289_328950

def f (x : ℤ) : ℤ :=
  if x % 2 = 1 then x + 5 else x / 2

def is_odd (n : ℤ) : Prop := n % 2 = 1

theorem least_c_for_triple_f (b : ℤ) :
  ∃ c : ℤ, is_odd c ∧ f (f (f c)) = b ∧ ∀ d : ℤ, is_odd d ∧ f (f (f d)) = b → c ≤ d :=
sorry

end least_c_for_triple_f_l3289_328950


namespace withdrawal_amount_l3289_328935

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def final_balance : ℕ := 76

theorem withdrawal_amount : 
  initial_balance + deposit - final_balance = 4 := by
  sorry

end withdrawal_amount_l3289_328935


namespace determine_dracula_status_l3289_328903

/-- Represents the types of Transylvanians -/
inductive TransylvanianType
| Truthful
| Liar

/-- Represents the possible answers to a yes/no question -/
inductive Answer
| Yes
| No

/-- Represents Dracula's status -/
inductive DraculaStatus
| Alive
| NotAlive

/-- A Transylvanian's response to the question -/
def response (t : TransylvanianType) (d : DraculaStatus) : Answer :=
  match t, d with
  | TransylvanianType.Truthful, DraculaStatus.Alive => Answer.Yes
  | TransylvanianType.Truthful, DraculaStatus.NotAlive => Answer.No
  | TransylvanianType.Liar, DraculaStatus.Alive => Answer.Yes
  | TransylvanianType.Liar, DraculaStatus.NotAlive => Answer.No

/-- The main theorem: The question can determine Dracula's status -/
theorem determine_dracula_status :
  ∀ (t : TransylvanianType) (d : DraculaStatus),
    response t d = Answer.Yes ↔ d = DraculaStatus.Alive :=
by sorry

end determine_dracula_status_l3289_328903


namespace sqrt_diff_approx_three_l3289_328990

theorem sqrt_diff_approx_three (k : ℕ) (h : k ≥ 7) :
  |Real.sqrt (9 * (k + 1)^2 + (k + 1)) - Real.sqrt (9 * k^2 + k) - 3| < (1 : ℝ) / 1000 := by
  sorry

end sqrt_diff_approx_three_l3289_328990


namespace landscape_breadth_l3289_328926

theorem landscape_breadth (length width : ℝ) (playground_area : ℝ) : 
  width = 8 * length →
  playground_area = 3200 →
  playground_area = (1 / 9) * (length * width) →
  width = 480 := by
sorry

end landscape_breadth_l3289_328926


namespace divisibility_for_odd_n_l3289_328976

theorem divisibility_for_odd_n (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, (82 : ℤ)^n + 454 * (69 : ℤ)^n = 1963 * k := by
  sorry

end divisibility_for_odd_n_l3289_328976


namespace train_journey_distance_l3289_328978

/-- Calculates the total distance traveled by a train with increasing speed -/
def train_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initial_speed + (hours - 1) * speed_increase) / 2

/-- Theorem: A train traveling for 11 hours, with an initial speed of 10 miles/hr
    and increasing its speed by 10 miles/hr each hour, travels a total of 660 miles -/
theorem train_journey_distance :
  train_distance 10 10 11 = 660 := by
  sorry

end train_journey_distance_l3289_328978


namespace apartment_utilities_cost_l3289_328936

/-- Represents the monthly costs and driving distance for an apartment --/
structure Apartment where
  rent : ℝ
  utilities : ℝ
  driveMiles : ℝ

/-- Calculates the total monthly cost for an apartment --/
def totalMonthlyCost (apt : Apartment) (workdays : ℝ) (driveCostPerMile : ℝ) : ℝ :=
  apt.rent + apt.utilities + (apt.driveMiles * workdays * driveCostPerMile)

/-- The problem statement --/
theorem apartment_utilities_cost 
  (apt1 : Apartment)
  (apt2 : Apartment)
  (workdays : ℝ)
  (driveCostPerMile : ℝ)
  (totalCostDifference : ℝ)
  (h1 : apt1.rent = 800)
  (h2 : apt1.utilities = 260)
  (h3 : apt1.driveMiles = 31)
  (h4 : apt2.rent = 900)
  (h5 : apt2.driveMiles = 21)
  (h6 : workdays = 20)
  (h7 : driveCostPerMile = 0.58)
  (h8 : totalMonthlyCost apt1 workdays driveCostPerMile - 
        totalMonthlyCost apt2 workdays driveCostPerMile = totalCostDifference)
  (h9 : totalCostDifference = 76) :
  apt2.utilities = 200 := by
  sorry


end apartment_utilities_cost_l3289_328936


namespace add_fractions_simplest_form_l3289_328942

theorem add_fractions_simplest_form :
  (7 : ℚ) / 8 + (3 : ℚ) / 5 = (59 : ℚ) / 40 ∧ 
  ∀ n d : ℤ, (d ≠ 0 ∧ (59 : ℚ) / 40 = (n : ℚ) / d) → n.gcd d = 1 := by
  sorry

end add_fractions_simplest_form_l3289_328942


namespace increasing_on_zero_one_iff_decreasing_on_three_four_l3289_328986

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem increasing_on_zero_one_iff_decreasing_on_three_four
  (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : has_period f 2) :
  is_increasing_on f 0 1 ↔ is_decreasing_on f 3 4 :=
sorry

end increasing_on_zero_one_iff_decreasing_on_three_four_l3289_328986


namespace units_digit_sum_factorials_2010_l3289_328925

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2010 :
  unitsDigit (sumFactorials 2010) = 3 := by
  sorry

end units_digit_sum_factorials_2010_l3289_328925


namespace units_digit_of_expression_l3289_328974

theorem units_digit_of_expression : 
  (2 * 21 * 2019 + 2^5 - 4^3) % 10 = 6 := by
  sorry

end units_digit_of_expression_l3289_328974


namespace no_solution_iff_n_eq_neg_two_l3289_328940

theorem no_solution_iff_n_eq_neg_two (n : ℤ) :
  (∀ x y : ℚ, 2 * x = 1 + n * y ∧ n * x = 1 + 2 * y) ↔ n = -2 := by
  sorry

end no_solution_iff_n_eq_neg_two_l3289_328940


namespace vaishali_hats_l3289_328984

/-- The number of hats with 4 stripes each that Vaishali has -/
def hats_with_four_stripes : ℕ :=
  let three_stripe_hats := 4
  let three_stripe_count := 3
  let no_stripe_hats := 6
  let five_stripe_hats := 2
  let five_stripe_count := 5
  let total_stripes := 34
  let remaining_stripes := total_stripes - 
    (three_stripe_hats * three_stripe_count + 
     no_stripe_hats * 0 + 
     five_stripe_hats * five_stripe_count)
  remaining_stripes / 4

theorem vaishali_hats : hats_with_four_stripes = 3 := by
  sorry

end vaishali_hats_l3289_328984


namespace min_ratio_digit_difference_l3289_328960

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def ratio (n : ℕ) : ℚ := n / (digit_sum n)

def ten_thousands_digit (n : ℕ) : ℕ := n / 10000

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

theorem min_ratio_digit_difference :
  ∃ (n : ℕ), is_five_digit n ∧
  (∀ (m : ℕ), is_five_digit m → ratio n ≤ ratio m) ∧
  (thousands_digit n - ten_thousands_digit n = 8) :=
sorry

end min_ratio_digit_difference_l3289_328960


namespace less_likely_white_ball_l3289_328993

theorem less_likely_white_ball (red_balls white_balls : ℕ) 
  (h_red : red_balls = 8) (h_white : white_balls = 2) :
  (white_balls : ℚ) / (red_balls + white_balls) < (red_balls : ℚ) / (red_balls + white_balls) :=
by sorry

end less_likely_white_ball_l3289_328993


namespace correct_sum_exists_l3289_328957

def num1 : ℕ := 3742586
def num2 : ℕ := 4829430
def given_sum : ℕ := 72120116

def replace_digit (n : ℕ) (d e : ℕ) : ℕ :=
  -- Function to replace all occurrences of d with e in n
  sorry

theorem correct_sum_exists : ∃ (d e : ℕ), d ≠ e ∧ 
  d < 10 ∧ e < 10 ∧ 
  replace_digit num1 d e + replace_digit num2 d e = given_sum ∧
  d + e = 10 := by
  sorry

end correct_sum_exists_l3289_328957


namespace inequality_solution_l3289_328962

theorem inequality_solution (x : ℝ) :
  (4 ≤ x^2 - 3*x - 6 ∧ x^2 - 3*x - 6 ≤ 2*x + 8) ↔ (5 ≤ x ∧ x ≤ 7) ∨ x = -2 :=
by sorry

end inequality_solution_l3289_328962


namespace cat_litter_weight_l3289_328969

/-- Calculates the weight of cat litter in each container given the problem conditions. -/
theorem cat_litter_weight 
  (container_price : ℝ) 
  (litter_box_capacity : ℝ) 
  (total_cost : ℝ) 
  (total_days : ℝ) 
  (h1 : container_price = 21)
  (h2 : litter_box_capacity = 15)
  (h3 : total_cost = 210)
  (h4 : total_days = 210) :
  (total_cost * litter_box_capacity) / (container_price * total_days / 7) = 3 := by
  sorry

end cat_litter_weight_l3289_328969


namespace transformed_line_y_intercept_l3289_328991

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Reflects a point in the line y = x -/
def reflectInDiagonal (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Applies a series of transformations to a line -/
def transformLine (l : Line) : Line :=
  sorry  -- The actual transformation is implemented here

/-- The main theorem stating that the transformed line has a y-intercept of -7 -/
theorem transformed_line_y_intercept :
  let originalLine : Line := { slope := 3, intercept := 6 }
  let transformedLine := transformLine originalLine
  transformedLine.intercept = -7 := by
  sorry


end transformed_line_y_intercept_l3289_328991


namespace equation_solution_l3289_328945

theorem equation_solution (x : ℝ) : 
  3 / (x + 2) = 2 / (x - 1) → x = 7 := by
  sorry

end equation_solution_l3289_328945


namespace unique_solution_condition_l3289_328906

/-- The equation (x + 3) / (kx - 2) = x + 1 has exactly one solution if and only if k = -7 + 2√10 or k = -7 - 2√10 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x + 1 ∧ k * x - 2 ≠ 0) ↔ 
  (k = -7 + 2 * Real.sqrt 10 ∨ k = -7 - 2 * Real.sqrt 10) :=
sorry

end unique_solution_condition_l3289_328906


namespace solve_equation_l3289_328981

theorem solve_equation (x : ℚ) : 
  (x - 30) / 2 = (5 - 3*x) / 6 + 2 → x = 167/6 := by
  sorry

end solve_equation_l3289_328981


namespace quadratic_transformation_l3289_328900

theorem quadratic_transformation (a b c : ℝ) :
  (∃ m q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = 5 * (x - 3)^2 + 7) →
  (∃ m p q : ℝ, ∀ x : ℝ, 4 * a * x^2 + 4 * b * x + 4 * c = m * (x - p)^2 + q) →
  (∃ m p q : ℝ, ∀ x : ℝ, 4 * a * x^2 + 4 * b * x + 4 * c = m * (x - p)^2 + q ∧ p = 3) := by
sorry

end quadratic_transformation_l3289_328900


namespace roots_quadratic_equation_l3289_328910

theorem roots_quadratic_equation (x₁ x₂ : ℝ) : 
  (x₁^2 + 3*x₁ - 2 = 0) → 
  (x₂^2 + 3*x₂ - 2 = 0) → 
  (x₁^2 + 2*x₁ - x₂ = 5) := by
  sorry

end roots_quadratic_equation_l3289_328910


namespace tomato_difference_l3289_328939

theorem tomato_difference (initial_tomatoes picked_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 17)
  (h2 : picked_tomatoes = 9) :
  initial_tomatoes - picked_tomatoes = 8 := by
  sorry

end tomato_difference_l3289_328939


namespace max_product_sum_300_l3289_328909

theorem max_product_sum_300 : 
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500 :=
by sorry

end max_product_sum_300_l3289_328909


namespace abc_product_equals_k_l3289_328912

theorem abc_product_equals_k (a b c k : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → k ≠ 0 →
  a ≠ b → b ≠ c → a ≠ c →
  (a + k / b = b + k / c) → (b + k / c = c + k / a) →
  |a * b * c| = |k| := by
sorry

end abc_product_equals_k_l3289_328912


namespace line_equation_proof_l3289_328983

-- Define the point A
def A : ℝ × ℝ := (-1, 4)

-- Define the x-intercept
def x_intercept : ℝ := 3

-- Theorem statement
theorem line_equation_proof :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x + y - 3 = 0)) ∧ 
    (A.2 = m * A.1 + b) ∧
    (0 = m * x_intercept + b) := by
  sorry

end line_equation_proof_l3289_328983


namespace f_properties_l3289_328915

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/4)^x - 8 * (1/2)^x - 1
  else if x = 0 then 0
  else -4^x + 8 * 2^x + 1

theorem f_properties :
  (∀ x, f x + f (-x) = 0) →
  (∀ x < 0, f x = (1/4)^x - 8 * (1/2)^x - 1) →
  (∀ x > 0, f x = -4^x + 8 * 2^x + 1) ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f y ≤ f x) ∧
  f 2 = 17 ∧
  (∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f x ≤ f y) ∧
  f 3 = 1 :=
by sorry

end f_properties_l3289_328915


namespace sequence_problem_l3289_328928

theorem sequence_problem (n : ℕ+) (b : ℕ → ℝ)
  (h0 : b 0 = 25)
  (h1 : b 1 = 56)
  (hn : b n = 0)
  (hk : ∀ k : ℕ, 1 ≤ k → k < n → b (k + 1) = b (k - 1) - 7 / b k) :
  n = 201 := by
  sorry

end sequence_problem_l3289_328928


namespace trefoil_cases_l3289_328972

theorem trefoil_cases (total_boxes : ℕ) (boxes_per_case : ℕ) (h1 : total_boxes = 24) (h2 : boxes_per_case = 8) :
  total_boxes / boxes_per_case = 3 := by
  sorry

end trefoil_cases_l3289_328972


namespace base_4_divisible_by_19_l3289_328999

def base_4_to_decimal (a b c d : ℕ) : ℕ := a * 4^3 + b * 4^2 + c * 4 + d

theorem base_4_divisible_by_19 :
  ∃! x : ℕ, x < 4 ∧ 19 ∣ base_4_to_decimal 2 1 x 2 :=
by
  sorry

end base_4_divisible_by_19_l3289_328999


namespace consecutive_product_divisible_by_six_l3289_328938

theorem consecutive_product_divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end consecutive_product_divisible_by_six_l3289_328938


namespace parabola_passes_origin_l3289_328996

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

/-- The origin point (0, 0) -/
def origin : Point :=
  { x := 0, y := 0 }

/-- Check if a point lies on a parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = para.f p.x

/-- The given parabola y = (x+2)^2 -/
def given_parabola : Parabola :=
  { f := λ x => (x + 2)^2 }

/-- Theorem: Rightward translation by 2 units makes the parabola pass through the origin -/
theorem parabola_passes_origin :
  ∃ (p : Point), lies_on (translate p 2 0) given_parabola ∧ p = origin := by
  sorry

end parabola_passes_origin_l3289_328996


namespace rectangular_plot_shorter_side_l3289_328975

theorem rectangular_plot_shorter_side
  (width : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (h1 : width = 50)
  (h2 : num_poles = 32)
  (h3 : pole_distance = 5)
  : ∃ (length : ℝ), length = 27.5 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance :=
by sorry

end rectangular_plot_shorter_side_l3289_328975


namespace range_of_m_l3289_328941

-- Define p as a proposition depending on m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

-- Define q as a proposition depending on m
def q (m : ℝ) : Prop := m > 2

-- Define the set of m satisfying the conditions
def S : Set ℝ := {m : ℝ | p m ∧ ¬(q m) ∧ ¬(¬(p m)) ∧ ¬(p m ∧ q m)}

-- Theorem statement
theorem range_of_m : S = {m : ℝ | 1 < m ∧ m ≤ 2} := by sorry

end range_of_m_l3289_328941


namespace largest_four_digit_sum_23_l3289_328918

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_23 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 23 → n ≤ 9950 :=
by sorry

end largest_four_digit_sum_23_l3289_328918


namespace hexagon_exterior_angles_sum_l3289_328954

/-- A hexagon is a polygon with 6 sides -/
def Hexagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a hexagon is 360 degrees -/
theorem hexagon_exterior_angles_sum :
  sum_exterior_angles Hexagon = 360 :=
sorry

end hexagon_exterior_angles_sum_l3289_328954


namespace square_areas_and_perimeters_l3289_328966

theorem square_areas_and_perimeters (x : ℝ) : 
  (x^2 + 4*x + 4) > 0 ∧ 
  (4*x^2 - 12*x + 9) > 0 ∧ 
  4 * (x + 2) + 4 * (2*x - 3) = 32 → 
  x = 3 := by sorry

end square_areas_and_perimeters_l3289_328966


namespace tim_total_sleep_l3289_328947

/-- Tim's weekly sleep schedule -/
structure SleepSchedule where
  weekdays : Nat -- Number of weekdays
  weekdaySleep : Nat -- Hours of sleep on weekdays
  weekends : Nat -- Number of weekend days
  weekendSleep : Nat -- Hours of sleep on weekends

/-- Calculate total sleep based on a sleep schedule -/
def totalSleep (schedule : SleepSchedule) : Nat :=
  schedule.weekdays * schedule.weekdaySleep + schedule.weekends * schedule.weekendSleep

/-- Tim's actual sleep schedule -/
def timSchedule : SleepSchedule :=
  { weekdays := 5
    weekdaySleep := 6
    weekends := 2
    weekendSleep := 10 }

/-- Theorem: Tim's total sleep per week is 50 hours -/
theorem tim_total_sleep : totalSleep timSchedule = 50 := by
  sorry

end tim_total_sleep_l3289_328947


namespace grid_shading_theorem_l3289_328920

/-- Represents a square on the grid -/
structure Square where
  row : Fin 6
  col : Fin 6

/-- Determines if a square is shaded based on its position -/
def is_shaded (s : Square) : Prop :=
  (s.row % 2 = 0 ∧ s.col % 2 = 1) ∨ (s.row % 2 = 1 ∧ s.col % 2 = 0)

/-- The total number of squares in the grid -/
def total_squares : Nat := 36

/-- The number of shaded squares in the grid -/
def shaded_squares : Nat := 21

/-- The fraction of shaded squares in the grid -/
def shaded_fraction : Rat := 7 / 12

theorem grid_shading_theorem :
  (shaded_squares : Rat) / total_squares = shaded_fraction := by
  sorry

end grid_shading_theorem_l3289_328920


namespace min_y_value_l3289_328971

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 18*x + 54*y) :
  ∃ (y_min : ℝ), y_min = 27 - Real.sqrt 810 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 18*x' + 54*y' → y' ≥ y_min :=
sorry

end min_y_value_l3289_328971


namespace pants_price_decrease_percentage_l3289_328982

/-- Proves that the percentage decrease in selling price is 20% given the conditions --/
theorem pants_price_decrease_percentage (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) :
  purchase_price = 210 →
  markup_percentage = 0.25 →
  gross_profit = 14 →
  let original_price := purchase_price / (1 - markup_percentage)
  let final_price := purchase_price + gross_profit
  let price_decrease := original_price - final_price
  let percentage_decrease := (price_decrease / original_price) * 100
  percentage_decrease = 20 := by
  sorry

end pants_price_decrease_percentage_l3289_328982


namespace volume_of_rhombus_revolution_l3289_328989

/-- A rhombus with side length 1 and shorter diagonal equal to its side -/
structure Rhombus where
  side_length : ℝ
  side_length_is_one : side_length = 1
  shorter_diagonal_eq_side : ℝ
  shorter_diagonal_eq_side_prop : shorter_diagonal_eq_side = side_length

/-- The volume of the solid of revolution formed by rotating the rhombus -/
noncomputable def volume_of_revolution (r : Rhombus) : ℝ := 
  3 * Real.pi / 2

/-- Theorem stating that the volume of the solid of revolution is 3π/2 -/
theorem volume_of_rhombus_revolution (r : Rhombus) : 
  volume_of_revolution r = 3 * Real.pi / 2 := by sorry

end volume_of_rhombus_revolution_l3289_328989


namespace cube_sum_equals_275_l3289_328904

theorem cube_sum_equals_275 (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 32) :
  a^3 + b^3 = 275 := by
  sorry

end cube_sum_equals_275_l3289_328904


namespace intersection_M_N_l3289_328905

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 - 3*x = 0}

theorem intersection_M_N : M ∩ N = {0} := by sorry

end intersection_M_N_l3289_328905


namespace three_digit_sum_9_l3289_328902

/-- A function that generates all three-digit numbers using digits 1 to 5 -/
def generateNumbers : List (Fin 5 × Fin 5 × Fin 5) := sorry

/-- A function that checks if the sum of digits in a three-digit number is 9 -/
def sumIs9 (n : Fin 5 × Fin 5 × Fin 5) : Bool := sorry

/-- The theorem to be proved -/
theorem three_digit_sum_9 : 
  (generateNumbers.filter sumIs9).length = 19 := by sorry

end three_digit_sum_9_l3289_328902


namespace complex_modulus_problem_l3289_328924

theorem complex_modulus_problem (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1/5 := by
  sorry

end complex_modulus_problem_l3289_328924


namespace average_molar_mass_of_compound_l3289_328997

/-- Given a compound where 4 moles weigh 672 grams, prove that its average molar mass is 168 grams/mole -/
theorem average_molar_mass_of_compound (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 672)
  (h2 : num_moles = 4) :
  total_weight / num_moles = 168 := by
  sorry

end average_molar_mass_of_compound_l3289_328997


namespace no_integer_solution_l3289_328973

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
  sorry

end no_integer_solution_l3289_328973


namespace marble_prism_weight_l3289_328951

/-- Represents the properties of a rectangular prism -/
structure RectangularPrism where
  height : ℝ
  baseLength : ℝ
  density : ℝ

/-- Calculates the weight of a rectangular prism -/
def weight (prism : RectangularPrism) : ℝ :=
  prism.height * prism.baseLength * prism.baseLength * prism.density

/-- Theorem: The weight of the specified marble rectangular prism is 86400 kg -/
theorem marble_prism_weight :
  let prism : RectangularPrism := {
    height := 8,
    baseLength := 2,
    density := 2700
  }
  weight prism = 86400 := by
  sorry

end marble_prism_weight_l3289_328951


namespace sum_of_products_l3289_328998

theorem sum_of_products : 5 * 12 + 7 * 15 + 13 * 4 + 6 * 9 = 271 := by
  sorry

end sum_of_products_l3289_328998


namespace quadratic_root_reciprocal_sum_l3289_328964

theorem quadratic_root_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 2 = 0 → 
  x₂^2 - 4*x₂ - 2 = 0 → 
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = -2 := by
  sorry

end quadratic_root_reciprocal_sum_l3289_328964


namespace prism_with_21_edges_has_9_faces_l3289_328963

/-- A prism is a polyhedron with two congruent parallel faces (bases) and whose other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

theorem prism_with_21_edges_has_9_faces (p : Prism) (h : p.edges = 21) : num_faces p = 9 := by
  sorry

end prism_with_21_edges_has_9_faces_l3289_328963


namespace monthly_revenue_is_4000_l3289_328931

/-- A store's financial data -/
structure StoreFinancials where
  initial_investment : ℕ
  monthly_expenses : ℕ
  payback_period : ℕ

/-- Calculate the monthly revenue required to break even -/
def calculate_monthly_revenue (store : StoreFinancials) : ℕ :=
  (store.initial_investment + store.monthly_expenses * store.payback_period) / store.payback_period

/-- Theorem: Given the store's financial data, the monthly revenue is $4000 -/
theorem monthly_revenue_is_4000 (store : StoreFinancials) 
    (h1 : store.initial_investment = 25000)
    (h2 : store.monthly_expenses = 1500)
    (h3 : store.payback_period = 10) :
  calculate_monthly_revenue store = 4000 := by
  sorry

end monthly_revenue_is_4000_l3289_328931


namespace gcf_of_24_and_16_l3289_328965

theorem gcf_of_24_and_16 :
  let n : ℕ := 24
  let m : ℕ := 16
  let lcm_nm : ℕ := 48
  lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end gcf_of_24_and_16_l3289_328965


namespace equation_has_two_solutions_l3289_328943

-- Define the equation
def f (x : ℝ) : Prop := Real.sqrt (5 - x) = x * Real.sqrt (5 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ f a ∧ f b ∧ ∀ (x : ℝ), f x → (x = a ∨ x = b) :=
sorry

end equation_has_two_solutions_l3289_328943


namespace all_dice_even_probability_l3289_328927

/-- The probability of a single standard six-sided die showing an even number -/
def prob_single_even : ℚ := 1 / 2

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 5

/-- The probability of all dice showing an even number -/
def prob_all_even : ℚ := (prob_single_even) ^ num_dice

theorem all_dice_even_probability :
  prob_all_even = 1 / 32 := by
  sorry

end all_dice_even_probability_l3289_328927


namespace arithmetic_geometric_progression_sine_l3289_328922

theorem arithmetic_geometric_progression_sine (x y z : ℝ) :
  let α := Real.arccos (-1/5)
  (∃ d, x = y - d ∧ z = y + d ∧ d = α) →
  (∃ r ≠ 1, (2 + Real.sin x) * (2 + Real.sin z) = (2 + Real.sin y)^2 ∧ 
             (2 + Real.sin y) = r * (2 + Real.sin x) ∧
             (2 + Real.sin z) = r * (2 + Real.sin y)) →
  Real.sin y = -1 := by
sorry

end arithmetic_geometric_progression_sine_l3289_328922


namespace focus_of_given_parabola_l3289_328949

/-- The parabola equation is y = (1/8)x^2 -/
def parabola_equation (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola with equation x^2 = 4py is at (0, p) -/
def focus_of_standard_parabola (p : ℝ) : ℝ × ℝ := (0, p)

/-- The theorem stating that the focus of the parabola y = (1/8)x^2 is at (0, 2) -/
theorem focus_of_given_parabola :
  ∃ (p : ℝ), (∀ x y : ℝ, parabola_equation x y ↔ x^2 = 4*p*y) ∧
             focus_of_standard_parabola p = (0, 2) := by sorry

end focus_of_given_parabola_l3289_328949


namespace negation_of_odd_function_implication_l3289_328967

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem negation_of_odd_function_implication :
  (¬ (is_odd f → is_odd (λ x => f (-x)))) ↔ (is_odd f → ¬ is_odd (λ x => f (-x))) :=
by sorry

end negation_of_odd_function_implication_l3289_328967


namespace juku_exit_position_l3289_328921

/-- Represents the state of Juku on the escalator -/
structure EscalatorState where
  time : ℕ
  position : ℕ

/-- The escalator system with Juku's movement -/
def escalator_system (total_steps : ℕ) (start_position : ℕ) : ℕ → EscalatorState
| 0 => ⟨0, start_position⟩
| t + 1 => 
  let prev := escalator_system total_steps start_position t
  let new_pos := 
    if t % 3 == 0 then prev.position - 1
    else if t % 3 == 1 then prev.position + 1
    else prev.position - 2
  ⟨t + 1, new_pos⟩

/-- Theorem: Juku exits at the 23rd step relative to the ground -/
theorem juku_exit_position : 
  ∃ (t : ℕ), (escalator_system 75 38 t).position + (t / 2) = 23 := by
  sorry

#eval (escalator_system 75 38 45).position + 45 / 2

end juku_exit_position_l3289_328921


namespace second_meeting_time_correct_l3289_328908

/-- Represents a vehicle on a race track -/
structure Vehicle where
  name : String
  lap_time : ℕ  -- lap time in seconds

/-- Calculates the time until two vehicles meet at the starting point for the second time -/
def timeToSecondMeeting (v1 v2 : Vehicle) : ℚ :=
  (Nat.lcm v1.lap_time v2.lap_time : ℚ) / 60

/-- The main theorem to prove -/
theorem second_meeting_time_correct 
  (magic : Vehicle) 
  (bull : Vehicle) 
  (h1 : magic.lap_time = 150)
  (h2 : bull.lap_time = 3600 / 40) :
  timeToSecondMeeting magic bull = 7.5 := by
  sorry

#eval timeToSecondMeeting 
  { name := "The Racing Magic", lap_time := 150 } 
  { name := "The Charging Bull", lap_time := 3600 / 40 }

end second_meeting_time_correct_l3289_328908


namespace score_difference_is_seven_l3289_328959

-- Define the score distribution
def score_distribution : List (Float × Float) := [
  (0.20, 60),
  (0.30, 70),
  (0.25, 85),
  (0.25, 95)
]

-- Define the mean score
def mean_score : Float :=
  (score_distribution.map (λ p => p.1 * p.2)).sum

-- Define the median score
def median_score : Float := 85

-- Theorem statement
theorem score_difference_is_seven :
  median_score - mean_score = 7 := by
  sorry


end score_difference_is_seven_l3289_328959


namespace line_point_sum_l3289_328929

/-- The line equation y = -2/5 * x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -2/5 * x + 10

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (25, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 10)

/-- Point T is on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop := 
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 4 * abs ((P.1 * s - r * P.2) / 2)

/-- Main theorem -/
theorem line_point_sum (r s : ℝ) 
  (h1 : line_equation r s) 
  (h2 : T_on_PQ r s) 
  (h3 : area_condition r s) : 
  r + s = 21.25 := by sorry

end line_point_sum_l3289_328929


namespace smallest_sum_B_plus_c_l3289_328917

theorem smallest_sum_B_plus_c : ∃ (B c : ℕ),
  (0 ≤ B ∧ B ≤ 4) ∧
  (c > 6) ∧
  (31 * B = 4 * (c + 1)) ∧
  (∀ (B' c' : ℕ), (0 ≤ B' ∧ B' ≤ 4) ∧ (c' > 6) ∧ (31 * B' = 4 * (c' + 1)) → B + c ≤ B' + c') ∧
  B + c = 34 :=
by sorry

end smallest_sum_B_plus_c_l3289_328917


namespace inverse_composition_theorem_l3289_328992

-- Define the functions f and g
variables (f g : ℝ → ℝ)

-- Define the condition f⁻¹ ∘ g = 3x - 2
def condition (f g : ℝ → ℝ) : Prop :=
  ∀ x, (f⁻¹ ∘ g) x = 3 * x - 2

-- Theorem statement
theorem inverse_composition_theorem (hfg : condition f g) :
  g⁻¹ (f (-10)) = -8/3 := by
  sorry

end inverse_composition_theorem_l3289_328992


namespace clara_weight_l3289_328944

/-- Given two weights satisfying certain conditions, prove that one of them is 88 pounds. -/
theorem clara_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight + clara_weight = 220)
  (h2 : clara_weight - alice_weight = clara_weight / 3) : 
  clara_weight = 88 := by
  sorry

end clara_weight_l3289_328944


namespace arithmetic_sequence_sum_l3289_328987

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₂ + a₁₂ = 32, prove that a₃ + a₁₁ = 32 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 2 + a 12 = 32) :
  a 3 + a 11 = 32 := by
  sorry

end arithmetic_sequence_sum_l3289_328987


namespace fruit_arrangements_l3289_328956

/-- The number of distinct arrangements of 9 items, where there are 4 indistinguishable items of type A, 3 indistinguishable items of type B, and 2 indistinguishable items of type C. -/
def distinct_arrangements (total : Nat) (a : Nat) (b : Nat) (c : Nat) : Nat :=
  Nat.factorial total / (Nat.factorial a * Nat.factorial b * Nat.factorial c)

/-- Theorem stating that the number of distinct arrangements of 9 items, 
    where there are 4 indistinguishable items of type A, 
    3 indistinguishable items of type B, and 2 indistinguishable items of type C, 
    is equal to 1260. -/
theorem fruit_arrangements : distinct_arrangements 9 4 3 2 = 1260 := by
  sorry

end fruit_arrangements_l3289_328956


namespace gaeun_taller_than_nana_l3289_328995

/-- Nana's height in meters -/
def nana_height_m : ℝ := 1.618

/-- Gaeun's height in centimeters -/
def gaeun_height_cm : ℝ := 162.3

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem gaeun_taller_than_nana : 
  gaeun_height_cm > nana_height_m * m_to_cm := by sorry

end gaeun_taller_than_nana_l3289_328995


namespace tetrahedron_in_spheres_l3289_328980

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron defined by four points -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere defined by its center and radius -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Checks if a point is inside a sphere -/
def isInSphere (p : Point3D) (s : Sphere) : Prop := sorry

/-- Checks if a point is inside a tetrahedron -/
def isInTetrahedron (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Creates a sphere with diameter AB -/
def sphereAB (t : Tetrahedron) : Sphere := sorry

/-- Creates a sphere with diameter AC -/
def sphereAC (t : Tetrahedron) : Sphere := sorry

/-- Creates a sphere with diameter AD -/
def sphereAD (t : Tetrahedron) : Sphere := sorry

/-- The main theorem: every point in the tetrahedron is in at least one of the three spheres -/
theorem tetrahedron_in_spheres (t : Tetrahedron) (p : Point3D) :
  isInTetrahedron p t →
  (isInSphere p (sphereAB t) ∨ isInSphere p (sphereAC t) ∨ isInSphere p (sphereAD t)) :=
by sorry

end tetrahedron_in_spheres_l3289_328980


namespace sum_max_at_5_l3289_328914

/-- An arithmetic sequence with its first term and sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum : ℕ → ℝ
  sum_def : ∀ n : ℕ, sum n = (n : ℝ) / 2 * (a 1 + a n)
  sum_9_positive : sum 9 > 0
  sum_10_negative : sum 10 < 0

/-- The sum of the arithmetic sequence is maximized at n = 5 -/
theorem sum_max_at_5 (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), seq.sum m ≤ seq.sum n ∧ n = 5 :=
sorry

end sum_max_at_5_l3289_328914


namespace reach_11_from_1_l3289_328988

/-- Represents the set of operations allowed by the calculator -/
inductive CalcOp
  | mul3 : CalcOp  -- Multiply by 3
  | add3 : CalcOp  -- Add 3
  | div3 : CalcOp  -- Divide by 3 (when divisible)

/-- Applies a single calculator operation to a number -/
def applyOp (n : ℕ) (op : CalcOp) : ℕ :=
  match op with
  | CalcOp.mul3 => n * 3
  | CalcOp.add3 => n + 3
  | CalcOp.div3 => if n % 3 = 0 then n / 3 else n

/-- Checks if it's possible to reach the target number from the start number using the given operations -/
def canReach (start target : ℕ) : Prop :=
  ∃ (steps : List CalcOp), (steps.foldl applyOp start) = target

/-- Theorem stating that it's possible to reach 11 from 1 using the calculator operations -/
theorem reach_11_from_1 : canReach 1 11 := by
  sorry

end reach_11_from_1_l3289_328988


namespace running_speed_calculation_l3289_328916

def walking_speed : ℝ := 4
def total_distance : ℝ := 20
def total_time : ℝ := 3.75

theorem running_speed_calculation (R : ℝ) :
  (total_distance / 2 / walking_speed) + (total_distance / 2 / R) = total_time →
  R = 8 := by
  sorry

end running_speed_calculation_l3289_328916


namespace angle_sum_at_point_l3289_328946

theorem angle_sum_at_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by
  sorry

end angle_sum_at_point_l3289_328946


namespace chairs_difference_l3289_328934

theorem chairs_difference (initial : ℕ) (remaining : ℕ) : 
  initial = 15 → remaining = 3 → initial - remaining = 12 := by
  sorry

end chairs_difference_l3289_328934


namespace officer_selection_theorem_l3289_328985

def total_members : ℕ := 18
def officer_positions : ℕ := 6
def past_officers : ℕ := 8

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem officer_selection_theorem : 
  choose total_members officer_positions - 
  (choose (total_members - past_officers) officer_positions + 
   past_officers * choose (total_members - past_officers) (officer_positions - 1)) = 16338 :=
sorry

end officer_selection_theorem_l3289_328985


namespace expression_evaluation_l3289_328933

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -8
  (Real.sqrt (9 * x * y) - 2 * Real.sqrt (x^3 * y) + Real.sqrt (x * y^3)) = 20 := by
sorry

end expression_evaluation_l3289_328933


namespace inequality_solution_set_l3289_328968

/-- The function representing the given inequality -/
def f (k x : ℝ) : ℝ := ((k^2 + 6*k + 14)*x - 9) * ((k^2 + 28)*x - 2*k^2 - 12*k)

/-- The solution set M for the inequality f(k, x) < 0 -/
def M (k : ℝ) : Set ℝ := {x | f k x < 0}

/-- The statement of the problem -/
theorem inequality_solution_set (k : ℝ) : 
  (M k ∩ Set.range (Int.cast : ℤ → ℝ) = {1}) ↔ (k < -14 ∨ (2 < k ∧ k ≤ 14/3)) :=
sorry

end inequality_solution_set_l3289_328968
