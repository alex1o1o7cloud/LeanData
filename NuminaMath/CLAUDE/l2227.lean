import Mathlib

namespace equation_solutions_l2227_222762

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) ∧
  (∀ x : ℝ, x^2 + 5*x + 6 = 0 ↔ x = -2 ∨ x = -3) := by
  sorry

end equation_solutions_l2227_222762


namespace hotel_bubble_bath_l2227_222735

/-- The amount of bubble bath needed for a hotel with couples and single rooms -/
def bubble_bath_needed (couple_rooms single_rooms : ℕ) (bath_per_person : ℕ) : ℕ :=
  (2 * couple_rooms + single_rooms) * bath_per_person

/-- Theorem: The amount of bubble bath needed for 13 couple rooms and 14 single rooms is 400ml -/
theorem hotel_bubble_bath :
  bubble_bath_needed 13 14 10 = 400 := by
  sorry

end hotel_bubble_bath_l2227_222735


namespace sweets_expenditure_correct_l2227_222791

/-- Calculates the amount spent on sweets given the initial amount and the amount given to each friend -/
def amount_spent_on_sweets (initial_amount : ℚ) (amount_per_friend : ℚ) : ℚ :=
  initial_amount - 2 * amount_per_friend

/-- Proves that the amount spent on sweets is correct for the given problem -/
theorem sweets_expenditure_correct (initial_amount : ℚ) (amount_per_friend : ℚ) 
  (h1 : initial_amount = 10.50)
  (h2 : amount_per_friend = 3.40) : 
  amount_spent_on_sweets initial_amount amount_per_friend = 3.70 := by
  sorry

#eval amount_spent_on_sweets 10.50 3.40

end sweets_expenditure_correct_l2227_222791


namespace shaded_area_rectangle_with_quarter_circles_l2227_222781

/-- The area of the shaded region in a rectangle with quarter circles at corners -/
theorem shaded_area_rectangle_with_quarter_circles 
  (length : ℝ) (width : ℝ) (radius : ℝ) 
  (h_length : length = 12) 
  (h_width : width = 8) 
  (h_radius : radius = 4) : 
  length * width - π * radius^2 = 96 - 16 * π := by
sorry

end shaded_area_rectangle_with_quarter_circles_l2227_222781


namespace min_distance_squared_l2227_222752

/-- Given real numbers a, b, c, d satisfying |b+a^2-4ln a|+|2c-d+2|=0,
    the minimum value of (a-c)^2+(b-d)^2 is 5. -/
theorem min_distance_squared (a b c d : ℝ) 
    (h : |b + a^2 - 4*Real.log a| + |2*c - d + 2| = 0) : 
  (∀ x y z w : ℝ, |w + x^2 - 4*Real.log x| + |2*y - z + 2| = 0 →
    (a - c)^2 + (b - d)^2 ≤ (x - y)^2 + (w - z)^2) ∧
  (∃ x y z w : ℝ, |w + x^2 - 4*Real.log x| + |2*y - z + 2| = 0 ∧
    (a - c)^2 + (b - d)^2 = (x - y)^2 + (w - z)^2) ∧
  (a - c)^2 + (b - d)^2 = 5 := by
  sorry

end min_distance_squared_l2227_222752


namespace handshake_count_l2227_222722

theorem handshake_count (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

#check handshake_count

end handshake_count_l2227_222722


namespace tearing_process_l2227_222721

/-- Represents the number of parts after a series of tearing operations -/
def NumParts : ℕ → ℕ
  | 0 => 1  -- Start with one piece
  | n + 1 => NumParts n + 2  -- Each tear adds 2 parts

theorem tearing_process (n : ℕ) :
  ∀ k, Odd (NumParts k) ∧ 
    (¬∃ m, NumParts m = 100) ∧
    (∃ m, NumParts m = 2017) := by
  sorry

#eval NumParts 1008  -- Should evaluate to 2017

end tearing_process_l2227_222721


namespace first_half_speed_l2227_222734

/-- Given a journey with the following properties:
  * The total distance is 224 km
  * The total time is 10 hours
  * The second half of the journey is traveled at 24 km/hr
  Prove that the speed during the first half of the journey is 21 km/hr -/
theorem first_half_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (second_half_speed : ℝ)
  (h1 : total_distance = 224)
  (h2 : total_time = 10)
  (h3 : second_half_speed = 24)
  : (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 21 := by
  sorry

end first_half_speed_l2227_222734


namespace percentage_married_employees_l2227_222757

/-- The percentage of married employees in a company given specific conditions -/
theorem percentage_married_employees :
  let percent_women : ℝ := 0.61
  let percent_married_women : ℝ := 0.7704918032786885
  let percent_single_men : ℝ := 2/3
  
  let percent_men : ℝ := 1 - percent_women
  let percent_married_men : ℝ := 1 - percent_single_men
  
  let married_women : ℝ := percent_women * percent_married_women
  let married_men : ℝ := percent_men * percent_married_men
  
  let total_married : ℝ := married_women + married_men
  
  total_married = 0.60020016000000005
  := by sorry

end percentage_married_employees_l2227_222757


namespace northton_capsule_depth_l2227_222785

/-- The depth of Southton's time capsule in feet -/
def southton_depth : ℝ := 15

/-- The depth of Northton's time capsule in feet -/
def northton_depth : ℝ := 4 * southton_depth - 12

/-- Theorem stating the depth of Northton's time capsule -/
theorem northton_capsule_depth : northton_depth = 48 := by
  sorry

end northton_capsule_depth_l2227_222785


namespace stating_cube_coloring_theorem_l2227_222778

/-- Represents the number of available colors -/
def num_colors : ℕ := 5

/-- Represents the number of faces in a cube -/
def num_faces : ℕ := 6

/-- Represents the number of faces already painted -/
def painted_faces : ℕ := 3

/-- Represents the number of remaining faces to be painted -/
def remaining_faces : ℕ := num_faces - painted_faces

/-- 
  Represents the number of valid coloring schemes for the remaining faces of a cube,
  given that three adjacent faces are already painted with different colors,
  and no two adjacent faces can have the same color.
-/
def valid_coloring_schemes : ℕ := 13

/-- 
  Theorem stating that the number of valid coloring schemes for the remaining faces
  of a cube is equal to 13, given the specified conditions.
-/
theorem cube_coloring_theorem :
  valid_coloring_schemes = 13 :=
sorry

end stating_cube_coloring_theorem_l2227_222778


namespace rainfall_problem_l2227_222727

/-- Rainfall problem -/
theorem rainfall_problem (sunday monday tuesday : ℝ) 
  (h1 : tuesday = 2 * monday)
  (h2 : monday = sunday + 3)
  (h3 : sunday + monday + tuesday = 25) :
  sunday = 4 := by
sorry

end rainfall_problem_l2227_222727


namespace exponential_solution_l2227_222798

/-- A function satisfying f(x+1) = 2f(x) for all real x -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = 2 * f x

/-- The theorem stating that if f satisfies the functional equation,
    then f(x) = C * 2^x for some constant C -/
theorem exponential_solution (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ C, ∀ x, f x = C * 2^x := by
  sorry

end exponential_solution_l2227_222798


namespace product_divisible_by_twelve_l2227_222740

theorem product_divisible_by_twelve (a b c d : ℤ) :
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b) = 12 * k := by
  sorry

end product_divisible_by_twelve_l2227_222740


namespace lamp_post_height_l2227_222744

/-- The height of a lamp post given specific conditions --/
theorem lamp_post_height (cable_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ)
  (h1 : cable_ground_distance = 4)
  (h2 : person_distance = 3)
  (h3 : person_height = 1.6)
  (h4 : person_distance < cable_ground_distance) :
  ∃ (post_height : ℝ),
    post_height = (cable_ground_distance * person_height) / (cable_ground_distance - person_distance) ∧
    post_height = 6.4 := by
  sorry

end lamp_post_height_l2227_222744


namespace necklace_diamond_count_l2227_222787

theorem necklace_diamond_count (total_necklaces : ℕ) (total_diamonds : ℕ) : 
  total_necklaces = 20 →
  total_diamonds = 79 →
  ∃ (two_diamond_necklaces five_diamond_necklaces : ℕ),
    two_diamond_necklaces + five_diamond_necklaces = total_necklaces ∧
    2 * two_diamond_necklaces + 5 * five_diamond_necklaces = total_diamonds ∧
    five_diamond_necklaces = 13 := by
  sorry

end necklace_diamond_count_l2227_222787


namespace factors_of_prime_factorization_l2227_222749

def prime_factorization := 2^3 * 3^5 * 5^4 * 7^2 * 11^6

def number_of_factors (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (3 + 1) * (5 + 1) * (4 + 1) * (2 + 1) * (6 + 1)

theorem factors_of_prime_factorization :
  number_of_factors prime_factorization = 2520 := by
  sorry

end factors_of_prime_factorization_l2227_222749


namespace quadratic_roots_midpoint_l2227_222733

theorem quadratic_roots_midpoint (a b : ℝ) (x₁ x₂ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f 2014 = f 2016) →
  (x₁^2 + a*x₁ + b = 0) →
  (x₂^2 + a*x₂ + b = 0) →
  (x₁ + x₂) / 2 = 2015 := by
sorry

end quadratic_roots_midpoint_l2227_222733


namespace logarithm_expression_equals_two_plus_pi_l2227_222716

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two_plus_pi :
  lg 5 * (Real.log 20 / Real.log (Real.sqrt 10)) + (lg (2 ^ Real.sqrt 2))^2 + Real.exp (Real.log π) = 2 + π := by
  sorry

end logarithm_expression_equals_two_plus_pi_l2227_222716


namespace cody_money_theorem_l2227_222745

def cody_money_problem (initial_money birthday_gift game_price discount friend_debt : ℝ) : Prop :=
  let total_before_purchase := initial_money + birthday_gift
  let discount_amount := game_price * discount
  let actual_game_cost := game_price - discount_amount
  let money_after_purchase := total_before_purchase - actual_game_cost
  let final_amount := money_after_purchase + friend_debt
  final_amount = 48.90

theorem cody_money_theorem :
  cody_money_problem 45 9 19 0.1 12 := by
  sorry

end cody_money_theorem_l2227_222745


namespace at_least_one_greater_than_one_l2227_222713

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) :
  x > 1 ∨ y > 1 := by
  sorry

end at_least_one_greater_than_one_l2227_222713


namespace students_not_enrolled_l2227_222700

theorem students_not_enrolled (total : ℕ) (english : ℕ) (history : ℕ) (both : ℕ) : 
  total = 60 → english = 42 → history = 30 → both = 18 →
  total - (english + history - both) = 6 := by
sorry

end students_not_enrolled_l2227_222700


namespace inequality_proof_l2227_222730

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/2) :
  (1 - a) * (1 - b) ≤ 9/16 := by
  sorry

end inequality_proof_l2227_222730


namespace total_toys_proof_l2227_222799

/-- The number of toys Kamari has -/
def kamari_toys : ℝ := 65

/-- The number of toys Anais has -/
def anais_toys : ℝ := kamari_toys + 30.5

/-- The number of toys Lucien has -/
def lucien_toys : ℝ := 2 * kamari_toys

/-- The total number of toys Anais and Kamari have together -/
def anais_kamari_total : ℝ := 160.5

theorem total_toys_proof :
  kamari_toys + anais_toys + lucien_toys = 290.5 ∧
  anais_toys = kamari_toys + 30.5 ∧
  lucien_toys = 2 * kamari_toys ∧
  anais_toys + kamari_toys = anais_kamari_total :=
by sorry

end total_toys_proof_l2227_222799


namespace inequality_system_integer_solutions_l2227_222770

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (x - 1 : ℚ) / 2 < x / 3 ∧ (2 * x - 5 : ℤ) ≤ 3 * (x - 2)}
  S = {1, 2} := by
  sorry

end inequality_system_integer_solutions_l2227_222770


namespace acid_solution_replacement_l2227_222789

theorem acid_solution_replacement (x : ℝ) :
  x ≥ 0 ∧ x ≤ 1 →
  0.5 * (1 - x) + 0.3 * x = 0.4 →
  x = 1/2 := by sorry

end acid_solution_replacement_l2227_222789


namespace magic_square_sum_div_by_3_l2227_222777

/-- Definition of a 3x3 magic square -/
def is_magic_square (a : Fin 9 → ℕ) (S : ℕ) : Prop :=
  -- Row sums
  (a 0 + a 1 + a 2 = S) ∧
  (a 3 + a 4 + a 5 = S) ∧
  (a 6 + a 7 + a 8 = S) ∧
  -- Column sums
  (a 0 + a 3 + a 6 = S) ∧
  (a 1 + a 4 + a 7 = S) ∧
  (a 2 + a 5 + a 8 = S) ∧
  -- Diagonal sums
  (a 0 + a 4 + a 8 = S) ∧
  (a 2 + a 4 + a 6 = S)

/-- Theorem: The sum of a third-order magic square is divisible by 3 -/
theorem magic_square_sum_div_by_3 (a : Fin 9 → ℕ) (S : ℕ) 
  (h : is_magic_square a S) : 
  3 ∣ S :=
by sorry

end magic_square_sum_div_by_3_l2227_222777


namespace food_duration_l2227_222766

theorem food_duration (initial_cows : ℕ) (days_passed : ℕ) (cows_left : ℕ) : 
  initial_cows = 1000 →
  days_passed = 10 →
  cows_left = 800 →
  (initial_cows * x - initial_cows * days_passed = cows_left * x) →
  x = 50 :=
by
  sorry

end food_duration_l2227_222766


namespace simplify_expression_l2227_222739

theorem simplify_expression (y : ℝ) : 2*y + 8*y^2 + 6 - (3 - 2*y - 8*y^2) = 16*y^2 + 4*y + 3 := by
  sorry

end simplify_expression_l2227_222739


namespace reduce_to_single_digit_l2227_222741

/-- Represents the operation of splitting digits and summing -/
def digit_split_sum (n : ℕ) : ℕ → ℕ → ℕ := sorry

/-- Predicate for a number being single-digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- Theorem stating that any natural number can be reduced to a single digit in at most 15 steps -/
theorem reduce_to_single_digit (N : ℕ) :
  ∃ (sequence : Fin 16 → ℕ),
    sequence 0 = N ∧
    (∀ i : Fin 15, ∃ a b : ℕ, sequence (i.succ) = digit_split_sum (sequence i) a b) ∧
    is_single_digit (sequence 15) :=
  sorry

end reduce_to_single_digit_l2227_222741


namespace blue_pens_count_l2227_222711

/-- Given a total number of pens and a number of black pens, 
    calculate the number of blue pens. -/
def blue_pens (total : ℕ) (black : ℕ) : ℕ :=
  total - black

/-- Theorem: When the total number of pens is 8 and the number of black pens is 4,
    the number of blue pens is 4. -/
theorem blue_pens_count : blue_pens 8 4 = 4 := by
  sorry

end blue_pens_count_l2227_222711


namespace hyperbola_ellipse_equations_l2227_222714

-- Define the foci
def F₁ : ℝ × ℝ := (0, -5)
def F₂ : ℝ × ℝ := (0, 5)

-- Define the intersection point
def P : ℝ × ℝ := (3, 4)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  y^2 / 40 + x^2 / 15 = 1

-- Define the hyperbola equation
def is_on_hyperbola (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

-- Define the asymptote equation
def is_on_asymptote (x y : ℝ) : Prop :=
  y = (4/3) * x

-- Theorem statement
theorem hyperbola_ellipse_equations :
  (is_on_ellipse P.1 P.2) ∧
  (is_on_hyperbola P.1 P.2) ∧
  (is_on_asymptote P.1 P.2) ∧
  (F₁.2 = -F₂.2) ∧
  (F₁.1 = F₂.1) :=
sorry

end hyperbola_ellipse_equations_l2227_222714


namespace star_sqrt3_minus_one_minus_sqrt7_l2227_222796

/-- Custom operation ※ -/
def star (a b : ℝ) : ℝ := (a + 1)^2 - b^2

/-- Theorem stating that (√3-1)※(-√7) = -4 -/
theorem star_sqrt3_minus_one_minus_sqrt7 :
  star (Real.sqrt 3 - 1) (-Real.sqrt 7) = -4 := by
  sorry

end star_sqrt3_minus_one_minus_sqrt7_l2227_222796


namespace yoongi_total_carrots_l2227_222786

/-- The number of carrots Yoongi has -/
def yoongi_carrots (initial : ℕ) (from_sister : ℕ) : ℕ :=
  initial + from_sister

theorem yoongi_total_carrots :
  yoongi_carrots 3 2 = 5 := by
  sorry

end yoongi_total_carrots_l2227_222786


namespace calculation_proof_equation_solution_l2227_222772

-- Part 1
theorem calculation_proof :
  (Real.sqrt (25 / 9) + (Real.log 5 / Real.log 10) ^ 0 + (27 / 64) ^ (-(1/3 : ℝ))) = 4 := by
  sorry

-- Part 2
theorem equation_solution :
  ∀ x : ℝ, (Real.log (6^x - 9) / Real.log 3) = 3 → x = 2 := by
  sorry

end calculation_proof_equation_solution_l2227_222772


namespace dvd_sales_proof_l2227_222747

theorem dvd_sales_proof (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end dvd_sales_proof_l2227_222747


namespace quadratic_root_product_l2227_222712

theorem quadratic_root_product (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I) ^ 2 + p * (1 - Complex.I) + q = 0 →
  p * q = -4 :=
by sorry

end quadratic_root_product_l2227_222712


namespace reflection_line_sum_l2227_222709

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, 7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x - 2)^2 + (y - 3)^2 = (10 - x)^2 + (7 - y)^2 ∧ 
    y = m * x + b ∧
    (y - 3) = -1 / m * (x - 2)) → 
  m + b = 15 := by sorry

end reflection_line_sum_l2227_222709


namespace notebook_purchase_cost_l2227_222759

def pen_cost : ℝ := 1.50
def notebook_cost : ℝ := 3 * pen_cost
def number_of_notebooks : ℕ := 4

theorem notebook_purchase_cost : 
  number_of_notebooks * notebook_cost = 18 := by
  sorry

end notebook_purchase_cost_l2227_222759


namespace sixth_term_value_l2227_222720

/-- Represents a geometric sequence --/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- Properties of the geometric sequence --/
def GeometricSequence.properties (seq : GeometricSequence) : Prop :=
  -- Sum of first four terms is 40
  seq.a * (1 + seq.r + seq.r^2 + seq.r^3) = 40 ∧
  -- Fifth term is 32
  seq.a * seq.r^4 = 32

/-- Sixth term of the geometric sequence --/
def GeometricSequence.sixthTerm (seq : GeometricSequence) : ℝ :=
  seq.a * seq.r^5

/-- Theorem stating that the sixth term is 1280/15 --/
theorem sixth_term_value (seq : GeometricSequence) 
  (h : seq.properties) : seq.sixthTerm = 1280/15 := by
  sorry

end sixth_term_value_l2227_222720


namespace intersection_of_M_and_N_l2227_222706

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -4 ≤ x ∧ x < -2} ∪ {x | 3 < x ∧ x ≤ 7} := by sorry

end intersection_of_M_and_N_l2227_222706


namespace days_to_read_book_290_l2227_222736

/-- Calculates the number of days required to read a book with the given reading pattern -/
def daysToReadBook (totalPages : ℕ) (sundayPages : ℕ) (otherDayPages : ℕ) : ℕ :=
  let pagesPerWeek := sundayPages + 6 * otherDayPages
  let completeWeeks := totalPages / pagesPerWeek
  let remainingPages := totalPages % pagesPerWeek
  let additionalDays := 
    if remainingPages ≤ sundayPages 
    then 1
    else 1 + ((remainingPages - sundayPages) + (otherDayPages - 1)) / otherDayPages
  7 * completeWeeks + additionalDays

/-- Theorem stating that it takes 41 days to read a 290-page book with the given reading pattern -/
theorem days_to_read_book_290 : 
  daysToReadBook 290 25 4 = 41 := by
  sorry

end days_to_read_book_290_l2227_222736


namespace complex_expression_equality_l2227_222760

theorem complex_expression_equality : -(-1 - (-2*(-3-4) - 5 - 6*(-7-80))) - 9 = 523 := by
  sorry

end complex_expression_equality_l2227_222760


namespace initial_daily_steps_is_1000_l2227_222756

/-- Calculates the total steps logged over 4 weeks given the initial daily step count -/
def totalSteps (initialDailySteps : ℕ) : ℕ :=
  7 * initialDailySteps +
  7 * (initialDailySteps + 1000) +
  7 * (initialDailySteps + 2000) +
  7 * (initialDailySteps + 3000)

/-- Proves that the initial daily step count is 1000 given the problem conditions -/
theorem initial_daily_steps_is_1000 :
  ∃ (initialDailySteps : ℕ),
    totalSteps initialDailySteps = 100000 - 30000 ∧
    initialDailySteps = 1000 :=
by
  sorry

end initial_daily_steps_is_1000_l2227_222756


namespace shortest_side_length_l2227_222715

theorem shortest_side_length (A B C : Real) (a b c : Real) : 
  B = π/4 → C = π/3 → c = 1 → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  b / (Real.sin B) = c / (Real.sin C) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  b ≤ a ∧ b ≤ c → 
  b = Real.sqrt 6 / 3 := by sorry

end shortest_side_length_l2227_222715


namespace second_subdivision_house_count_l2227_222793

/-- The number of houses in the second subdivision where Billy goes trick-or-treating -/
def second_subdivision_houses : ℕ := 75

/-- Anna's candy per house -/
def anna_candy_per_house : ℕ := 14

/-- Number of houses Anna visits -/
def anna_houses : ℕ := 60

/-- Billy's candy per house -/
def billy_candy_per_house : ℕ := 11

/-- Difference in total candy between Anna and Billy -/
def candy_difference : ℕ := 15

theorem second_subdivision_house_count :
  anna_candy_per_house * anna_houses = 
  billy_candy_per_house * second_subdivision_houses + candy_difference := by
  sorry

end second_subdivision_house_count_l2227_222793


namespace total_sum_calculation_l2227_222737

/-- 
Given that Maggie's share is 75% of the total sum and equals $4,500, 
prove that the total sum is $6,000.
-/
theorem total_sum_calculation (maggies_share : ℝ) (total_sum : ℝ) : 
  maggies_share = 4500 ∧ 
  maggies_share = 0.75 * total_sum →
  total_sum = 6000 := by
  sorry

end total_sum_calculation_l2227_222737


namespace prob_eight_rolls_divisible_by_four_l2227_222797

/-- The probability that a single die roll is even -/
def p_even : ℚ := 1/2

/-- The number of dice rolls -/
def n : ℕ := 8

/-- The probability mass function of the binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability that the product of n dice rolls is divisible by 4 -/
def prob_divisible_by_four (n : ℕ) (p : ℚ) : ℚ :=
  1 - (binomial_pmf n p 0 + binomial_pmf n p 1)

theorem prob_eight_rolls_divisible_by_four :
  prob_divisible_by_four n p_even = 247/256 := by
  sorry

#eval prob_divisible_by_four n p_even

end prob_eight_rolls_divisible_by_four_l2227_222797


namespace four_digit_divisor_characterization_l2227_222782

/-- Represents a four-digit number in decimal notation -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Converts a FourDigitNumber to its decimal value -/
def to_decimal (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Checks if one FourDigitNumber divides another -/
def divides (m n : FourDigitNumber) : Prop :=
  ∃ k : Nat, k * (to_decimal m) = to_decimal n

/-- Main theorem: Characterization of four-digit numbers that divide their rotations -/
theorem four_digit_divisor_characterization (n : FourDigitNumber) :
  (divides n {a := n.b, b := n.c, c := n.d, d := n.a, 
              a_nonzero := sorry, b_digit := n.c_digit, c_digit := n.d_digit, d_digit := sorry}) ∨
  (divides n {a := n.c, b := n.d, c := n.a, d := n.b, 
              a_nonzero := sorry, b_digit := n.d_digit, c_digit := sorry, d_digit := n.b_digit}) ∨
  (divides n {a := n.d, b := n.a, c := n.b, d := n.c, 
              a_nonzero := sorry, b_digit := sorry, c_digit := n.b_digit, d_digit := n.c_digit})
  ↔
  n.a = n.c ∧ n.b = n.d ∧ n.b ≠ 0 :=
sorry

end four_digit_divisor_characterization_l2227_222782


namespace square_position_after_2023_transformations_l2227_222725

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | CDAB
  | DCBA
  | BADC

-- Define the transformations
def rotate180 (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.ABCD
  | SquarePosition.DCBA => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DCBA

def reflectHorizontal (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.DCBA => SquarePosition.ABCD
  | SquarePosition.BADC => SquarePosition.CDAB

-- Define the alternating transformations
def alternateTransform (n : Nat) (pos : SquarePosition) : SquarePosition :=
  match n with
  | 0 => pos
  | n + 1 => 
    if n % 2 == 0
    then reflectHorizontal (alternateTransform n pos)
    else rotate180 (alternateTransform n pos)

-- The theorem to prove
theorem square_position_after_2023_transformations :
  alternateTransform 2023 SquarePosition.ABCD = SquarePosition.DCBA := by
  sorry


end square_position_after_2023_transformations_l2227_222725


namespace intersection_of_A_and_B_l2227_222710

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l2227_222710


namespace river_width_river_width_example_l2227_222784

/-- Calculates the width of a river given its depth, flow rate, and discharge volume. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (discharge_volume : ℝ) : ℝ :=
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  let width := discharge_volume / (flow_rate_mpm * depth)
  width

/-- The width of a river with given parameters is 45 meters. -/
theorem river_width_example : river_width 2 6 9000 = 45 := by
  sorry

end river_width_river_width_example_l2227_222784


namespace second_person_speed_l2227_222780

/-- Given two people walking in the same direction, this theorem proves
    the speed of the second person given the conditions of the problem. -/
theorem second_person_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 9.5)
  (h2 : distance = 9.5)
  (h3 : speed1 = 4.5)
  : ∃ (speed2 : ℝ), speed2 = 5.5 ∧ distance = (speed2 - speed1) * time :=
by
  sorry

#check second_person_speed

end second_person_speed_l2227_222780


namespace det_is_zero_l2227_222705

variables {α : Type*} [Field α]
variables (s p q : α)

-- Define the polynomial
def f (x : α) := x^3 - s*x^2 + p*x + q

-- Define the roots
structure Roots (s p q : α) where
  a : α
  b : α
  c : α
  root_a : f s p q a = 0
  root_b : f s p q b = 0
  root_c : f s p q c = 0

-- Define the matrix
def matrix (r : Roots s p q) : Matrix (Fin 3) (Fin 3) α :=
  ![![r.a, r.b, r.c],
    ![r.c, r.a, r.b],
    ![r.b, r.c, r.a]]

-- Theorem statement
theorem det_is_zero (r : Roots s p q) : 
  Matrix.det (matrix s p q r) = 0 := by
  sorry

end det_is_zero_l2227_222705


namespace least_five_digit_congruent_to_7_mod_17_is_correct_l2227_222775

/-- The least five-digit positive integer congruent to 7 (mod 17) -/
def least_five_digit_congruent_to_7_mod_17 : ℕ := 10003

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem least_five_digit_congruent_to_7_mod_17_is_correct :
  is_five_digit least_five_digit_congruent_to_7_mod_17 ∧
  least_five_digit_congruent_to_7_mod_17 % 17 = 7 ∧
  ∀ n : ℕ, is_five_digit n ∧ n % 17 = 7 → n ≥ least_five_digit_congruent_to_7_mod_17 :=
by sorry

end least_five_digit_congruent_to_7_mod_17_is_correct_l2227_222775


namespace square_greater_than_l2227_222776

theorem square_greater_than (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end square_greater_than_l2227_222776


namespace difference_x_y_l2227_222728

theorem difference_x_y (x y : ℤ) (h1 : x + y = 20) (h2 : x = 28) : x - y = 36 := by
  sorry

end difference_x_y_l2227_222728


namespace complete_square_quadratic_l2227_222769

theorem complete_square_quadratic (x : ℝ) :
  ∃ (a b : ℝ), (x^2 + 10*x - 3 = 0) ↔ ((x + a)^2 = b) ∧ b = 28 := by
  sorry

end complete_square_quadratic_l2227_222769


namespace price_difference_l2227_222718

/-- Given the total cost of a shirt and sweater, and the price of the shirt,
    calculate the difference in price between the sweater and the shirt. -/
theorem price_difference (total_cost shirt_price : ℚ) 
  (h1 : total_cost = 80.34)
  (h2 : shirt_price = 36.46)
  (h3 : shirt_price < total_cost - shirt_price) :
  total_cost - shirt_price - shirt_price = 7.42 := by
  sorry

end price_difference_l2227_222718


namespace tan_fraction_equality_l2227_222707

theorem tan_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end tan_fraction_equality_l2227_222707


namespace second_to_last_digit_of_power_of_three_is_even_l2227_222742

theorem second_to_last_digit_of_power_of_three_is_even (n : ℕ) :
  ∃ (k : ℕ), 3^n ≡ 20 * k + 2 * (3^n / 10 % 10) [ZMOD 100] :=
sorry

end second_to_last_digit_of_power_of_three_is_even_l2227_222742


namespace toy_store_solution_l2227_222767

/-- Represents the selling and cost prices of toys, and the optimal purchase strategy -/
structure ToyStore where
  sell_price_A : ℝ
  sell_price_B : ℝ
  cost_price_A : ℝ
  cost_price_B : ℝ
  optimal_purchase_A : ℕ
  optimal_purchase_B : ℕ

/-- The toy store problem with given conditions -/
def toy_store_problem : Prop :=
  ∃ (store : ToyStore),
    -- Selling price condition
    store.sell_price_B - store.sell_price_A = 30 ∧
    -- Total sales condition
    2 * store.sell_price_A + 3 * store.sell_price_B = 740 ∧
    -- Cost prices
    store.cost_price_A = 90 ∧
    store.cost_price_B = 110 ∧
    -- Total purchase constraint
    store.optimal_purchase_A + store.optimal_purchase_B = 80 ∧
    -- Total cost constraint
    store.cost_price_A * store.optimal_purchase_A + store.cost_price_B * store.optimal_purchase_B ≤ 8400 ∧
    -- Correct selling prices
    store.sell_price_A = 130 ∧
    store.sell_price_B = 160 ∧
    -- Optimal purchase strategy
    store.optimal_purchase_A = 20 ∧
    store.optimal_purchase_B = 60 ∧
    -- Profit maximization (implied by the optimal strategy)
    ∀ (m : ℕ), m + (80 - m) = 80 →
      (store.sell_price_A - store.cost_price_A) * store.optimal_purchase_A +
      (store.sell_price_B - store.cost_price_B) * store.optimal_purchase_B ≥
      (store.sell_price_A - store.cost_price_A) * m +
      (store.sell_price_B - store.cost_price_B) * (80 - m)

theorem toy_store_solution : toy_store_problem := by
  sorry


end toy_store_solution_l2227_222767


namespace product_of_solutions_l2227_222726

theorem product_of_solutions (x : ℝ) : 
  (∃ α β : ℝ, x^2 + 4*x + 49 = 0 ∧ x = α ∨ x = β) → 
  (∃ α β : ℝ, x^2 + 4*x + 49 = 0 ∧ x = α ∨ x = β ∧ α * β = -49) :=
by sorry

end product_of_solutions_l2227_222726


namespace constant_term_value_l2227_222748

theorem constant_term_value (x y : ℝ) (C : ℝ) : 
  5 * x + y = C →
  x + 3 * y = 1 →
  3 * x + 2 * y = 10 →
  C = 19 :=
by
  sorry

end constant_term_value_l2227_222748


namespace decision_symbol_is_diamond_l2227_222763

-- Define the type for flowchart symbols
inductive FlowchartSymbol
  | Diamond
  | Rectangle
  | Oval
  | Parallelogram

-- Define the function that determines if a symbol represents a decision
def representsDecision (symbol : FlowchartSymbol) : Prop :=
  symbol = FlowchartSymbol.Diamond

-- Theorem: The symbol that represents a decision in a flowchart is a diamond-shaped box
theorem decision_symbol_is_diamond :
  ∃ (symbol : FlowchartSymbol), representsDecision symbol :=
sorry

end decision_symbol_is_diamond_l2227_222763


namespace platter_total_is_26_l2227_222792

/-- Represents the number of fruits of each type in the initial set --/
structure InitialFruits :=
  (green_apples : ℕ)
  (red_apples : ℕ)
  (yellow_apples : ℕ)
  (red_oranges : ℕ)
  (yellow_oranges : ℕ)
  (green_kiwis : ℕ)
  (purple_grapes : ℕ)
  (green_grapes : ℕ)

/-- Represents the ratio of apples in the platter --/
structure AppleRatio :=
  (green : ℕ)
  (red : ℕ)
  (yellow : ℕ)

/-- Calculates the total number of fruits in the platter --/
def calculate_platter_total (initial : InitialFruits) (ratio : AppleRatio) : ℕ :=
  let green_apples := ratio.green
  let red_apples := ratio.red
  let yellow_apples := ratio.yellow
  let red_oranges := 1
  let yellow_oranges := 2
  let kiwis_and_grapes := min initial.green_kiwis initial.purple_grapes
  green_apples + red_apples + yellow_apples + red_oranges + yellow_oranges + 2 * kiwis_and_grapes

/-- Theorem stating that the total number of fruits in the platter is 26 --/
theorem platter_total_is_26 (initial : InitialFruits) (ratio : AppleRatio) : 
  initial.green_apples = 2 →
  initial.red_apples = 3 →
  initial.yellow_apples = 14 →
  initial.red_oranges = 4 →
  initial.yellow_oranges = 8 →
  initial.green_kiwis = 10 →
  initial.purple_grapes = 7 →
  initial.green_grapes = 5 →
  ratio.green = 2 →
  ratio.red = 4 →
  ratio.yellow = 3 →
  calculate_platter_total initial ratio = 26 := by
  sorry


end platter_total_is_26_l2227_222792


namespace symmetric_expressions_l2227_222788

-- Define what it means for an expression to be completely symmetric
def is_completely_symmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), f a b c = f b a c ∧ f a b c = f a c b ∧ f a b c = f c b a

-- Define the three expressions
def expr1 (a b c : ℝ) : ℝ := (a - b)^2
def expr2 (a b c : ℝ) : ℝ := a * b + b * c + c * a
def expr3 (a b c : ℝ) : ℝ := a^2 * b + b^2 * c + c^2 * a

-- State the theorem
theorem symmetric_expressions :
  is_completely_symmetric expr1 ∧
  is_completely_symmetric expr2 ∧
  ¬ is_completely_symmetric expr3 := by sorry

end symmetric_expressions_l2227_222788


namespace solution_set_abs_inequality_l2227_222764

theorem solution_set_abs_inequality :
  Set.Icc (1 : ℝ) 2 = {x : ℝ | |2*x - 3| ≤ 1} := by sorry

end solution_set_abs_inequality_l2227_222764


namespace circumscribed_sphere_radius_eq_side_length_l2227_222761

/-- A regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  /-- The side length of the base -/
  baseSideLength : ℝ
  /-- The height of the pyramid -/
  height : ℝ

/-- The radius of a sphere circumscribed around a regular hexagonal pyramid -/
def circumscribedSphereRadius (p : RegularHexagonalPyramid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed sphere of a regular hexagonal pyramid
    with base side length a and height a is equal to a -/
theorem circumscribed_sphere_radius_eq_side_length
    (p : RegularHexagonalPyramid)
    (h1 : p.baseSideLength = p.height)
    (h2 : p.baseSideLength > 0) :
    circumscribedSphereRadius p = p.baseSideLength :=
  sorry

end circumscribed_sphere_radius_eq_side_length_l2227_222761


namespace inverse_function_inequality_solution_set_l2227_222750

/-- Given two functions f and g that intersect at two points, 
    prove the solution set of the inequality between their inverse functions. -/
theorem inverse_function_inequality_solution_set 
  (f g : ℝ → ℝ)
  (h_f : ∃ k b : ℝ, ∀ x, f x = k * x + b)
  (h_g : ∀ x, g x = 2^x + 1)
  (h_intersect : ∃ x₁ x₂ : ℝ, 
    f x₁ = g x₁ ∧ f x₁ = 2 ∧ 
    f x₂ = g x₂ ∧ f x₂ = 4 ∧
    x₁ < x₂)
  (f_inv g_inv : ℝ → ℝ)
  (h_f_inv : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)
  (h_g_inv : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x)
  : {x : ℝ | f_inv x ≥ g_inv x} = Set.Ici 4 ∪ Set.Ioc 1 2 :=
sorry

end inverse_function_inequality_solution_set_l2227_222750


namespace n2o3_molecular_weight_is_76_02_l2227_222795

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in g/mol -/
def n2o3_molecular_weight : ℝ := nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

/-- Theorem stating that the molecular weight of N2O3 is 76.02 g/mol -/
theorem n2o3_molecular_weight_is_76_02 : 
  n2o3_molecular_weight = 76.02 := by sorry

end n2o3_molecular_weight_is_76_02_l2227_222795


namespace coconut_grove_average_yield_l2227_222731

/-- The yield of coconuts per year for a group of trees -/
structure CoconutYield where
  trees : ℕ
  nuts_per_year : ℕ

/-- The total yield of coconuts from multiple groups of trees -/
def total_yield (yields : List CoconutYield) : ℕ :=
  yields.map (λ y => y.trees * y.nuts_per_year) |>.sum

/-- The total number of trees from multiple groups -/
def total_trees (yields : List CoconutYield) : ℕ :=
  yields.map (λ y => y.trees) |>.sum

/-- The average yield per tree per year -/
def average_yield (yields : List CoconutYield) : ℚ :=
  (total_yield yields : ℚ) / (total_trees yields : ℚ)

theorem coconut_grove_average_yield : 
  let yields : List CoconutYield := [
    { trees := 3, nuts_per_year := 60 },
    { trees := 2, nuts_per_year := 120 },
    { trees := 1, nuts_per_year := 180 }
  ]
  average_yield yields = 100 := by
  sorry

end coconut_grove_average_yield_l2227_222731


namespace largest_constant_inequality_l2227_222743

theorem largest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (K : ℝ), K = Real.sqrt 3 ∧ 
  (∀ (K' : ℝ), (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0), 
    Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ K' * Real.sqrt (x + y + z)) → 
  K' ≤ K) ∧
  Real.sqrt (a * b / c) + Real.sqrt (b * c / a) + Real.sqrt (a * c / b) ≥ K * Real.sqrt (a + b + c) :=
sorry

end largest_constant_inequality_l2227_222743


namespace prob_ratio_l2227_222732

/- Define the total number of cards -/
def total_cards : ℕ := 50

/- Define the number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/- Define the number of cards for each number -/
def cards_per_number : ℕ := 5

/- Define the number of cards drawn -/
def cards_drawn : ℕ := 5

/- Function to calculate the probability of drawing 5 cards of the same number -/
def prob_same_number : ℚ :=
  (distinct_numbers : ℚ) / Nat.choose total_cards cards_drawn

/- Function to calculate the probability of drawing 4 cards of one number and 1 of another -/
def prob_four_and_one : ℚ :=
  (distinct_numbers * (distinct_numbers - 1) * cards_per_number * cards_per_number : ℚ) / 
  Nat.choose total_cards cards_drawn

/- Theorem stating the ratio of probabilities -/
theorem prob_ratio : 
  prob_four_and_one / prob_same_number = 225 := by sorry

end prob_ratio_l2227_222732


namespace fifth_term_of_arithmetic_sequence_l2227_222790

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem fifth_term_of_arithmetic_sequence 
  (a d : ℤ) 
  (h1 : arithmetic_sequence a d 10 = 15) 
  (h2 : arithmetic_sequence a d 11 = 18) : 
  arithmetic_sequence a d 5 = 0 := by
sorry

end fifth_term_of_arithmetic_sequence_l2227_222790


namespace line_slope_and_intercept_l2227_222703

/-- For a line with equation 2x + y + 1 = 0, its slope is -2 and y-intercept is -1 -/
theorem line_slope_and_intercept :
  ∀ (x y : ℝ), 2*x + y + 1 = 0 → 
  ∃ (k b : ℝ), k = -2 ∧ b = -1 ∧ y = k*x + b := by
sorry

end line_slope_and_intercept_l2227_222703


namespace remainder_thirteen_six_twelve_seven_eleven_eight_mod_five_l2227_222773

theorem remainder_thirteen_six_twelve_seven_eleven_eight_mod_five :
  (13^6 + 12^7 + 11^8) % 5 = 3 := by
  sorry

end remainder_thirteen_six_twelve_seven_eleven_eight_mod_five_l2227_222773


namespace square_arrangement_exists_l2227_222758

-- Define the structure of the square
structure Square where
  bottomLeft : ℕ
  topRight : ℕ
  bottomRight : ℕ
  topLeft : ℕ
  center : ℕ

-- Define the property of having a common divisor greater than 1
def hasCommonDivisorGreaterThanOne (m n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k ∣ m ∧ k ∣ n

-- Define the property of being relatively prime
def isRelativelyPrime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

-- Main theorem
theorem square_arrangement_exists : ∃ (a b c d : ℕ), ∃ (s : Square),
  s.bottomLeft = a * b ∧
  s.topRight = c * d ∧
  s.bottomRight = a * d ∧
  s.topLeft = b * c ∧
  s.center = a * b * c * d ∧
  (hasCommonDivisorGreaterThanOne s.bottomLeft s.center) ∧
  (hasCommonDivisorGreaterThanOne s.topRight s.center) ∧
  (hasCommonDivisorGreaterThanOne s.bottomRight s.center) ∧
  (hasCommonDivisorGreaterThanOne s.topLeft s.center) ∧
  (isRelativelyPrime s.bottomLeft s.topRight) ∧
  (isRelativelyPrime s.bottomRight s.topLeft) :=
sorry

end square_arrangement_exists_l2227_222758


namespace smallest_number_with_prime_property_l2227_222794

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def remove_first_digit (n : ℕ) : ℕ := n % 1000

theorem smallest_number_with_prime_property : 
  ∃! n : ℕ, 
    (∀ m : ℕ, m < n → 
      ¬(∃ p q : ℕ, 
        is_prime p ∧ 
        is_prime q ∧ 
        remove_first_digit m = 4 * p ∧ 
        remove_first_digit m + 1 = 5 * q)) ∧
    (∃ p q : ℕ, 
      is_prime p ∧ 
      is_prime q ∧ 
      remove_first_digit n = 4 * p ∧ 
      remove_first_digit n + 1 = 5 * q) ∧
    n = 1964 :=
by sorry

end smallest_number_with_prime_property_l2227_222794


namespace birds_count_l2227_222779

/-- The number of birds on the fence -/
def birds : ℕ := sorry

/-- Ten more than twice the number of birds on the fence is 50 -/
axiom birds_condition : 10 + 2 * birds = 50

/-- Prove that the number of birds on the fence is 20 -/
theorem birds_count : birds = 20 := by sorry

end birds_count_l2227_222779


namespace lcm_factor_problem_l2227_222724

theorem lcm_factor_problem (A B : ℕ) (H : ℕ) (X Y : ℕ) :
  H = 23 →
  Y = 14 →
  max A B = 322 →
  H = Nat.gcd A B →
  Nat.lcm A B = H * X * Y →
  X = 23 := by
sorry

end lcm_factor_problem_l2227_222724


namespace supplementary_angle_difference_l2227_222729

theorem supplementary_angle_difference : 
  let angle1 : ℝ := 99
  let angle2 : ℝ := 81
  -- Supplementary angles sum to 180°
  angle1 + angle2 = 180 →
  -- The difference between the larger and smaller angle is 18°
  max angle1 angle2 - min angle1 angle2 = 18 :=
by
  sorry

end supplementary_angle_difference_l2227_222729


namespace line_circle_intersection_slope_range_l2227_222717

/-- Given a line passing through (4,0) and intersecting the circle (x-2)^2 + y^2 = 1,
    prove that its slope k is between -√3/3 and √3/3 inclusive. -/
theorem line_circle_intersection_slope_range :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), y = k * (x - 4) ∧ (x - 2)^2 + y^2 = 1) →
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 := by
sorry

end line_circle_intersection_slope_range_l2227_222717


namespace smallest_b_value_l2227_222755

theorem smallest_b_value (a b : ℕ) : 
  (a ≥ 1000) → (a ≤ 9999) → (b ≥ 100000) → (b ≤ 999999) → 
  (1 : ℚ) / 2006 = 1 / a + 1 / b → 
  ∀ b' ≥ 100000, b' ≤ 999999 → 
    ∃ a' ≥ 1000, a' ≤ 9999 → (1 : ℚ) / 2006 = 1 / a' + 1 / b' → 
      b ≤ b' → b = 120360 := by
sorry

end smallest_b_value_l2227_222755


namespace power_product_equals_75600_l2227_222701

theorem power_product_equals_75600 : 2^4 * 5^2 * 3^3 * 7 = 75600 := by
  sorry

end power_product_equals_75600_l2227_222701


namespace systematic_sampling_interval_example_l2227_222774

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (population : ℕ) (sampleSize : ℕ) : ℕ :=
  population / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 50 is 20 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 1000 50 = 20 := by
  sorry

end systematic_sampling_interval_example_l2227_222774


namespace rectangle_opposite_sides_equal_square_all_sides_equal_l2227_222719

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a square
structure Square where
  side : ℝ

-- Theorem for rectangle
theorem rectangle_opposite_sides_equal (r : Rectangle) : 
  r.width = r.width ∧ r.height = r.height := by
  sorry

-- Theorem for square
theorem square_all_sides_equal (s : Square) : 
  s.side = s.side ∧ s.side = s.side ∧ s.side = s.side ∧ s.side = s.side := by
  sorry

end rectangle_opposite_sides_equal_square_all_sides_equal_l2227_222719


namespace highest_power_of_seven_in_100_factorial_l2227_222783

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem highest_power_of_seven_in_100_factorial :
  ∃ (k : ℕ), factorial 100 % (7^16) = 0 ∧ factorial 100 % (7^17) ≠ 0 :=
by sorry

end highest_power_of_seven_in_100_factorial_l2227_222783


namespace shark_teeth_multiple_l2227_222765

theorem shark_teeth_multiple : 
  let tiger_teeth : ℕ := 180
  let hammerhead_teeth : ℕ := tiger_teeth / 6
  let sum_teeth : ℕ := tiger_teeth + hammerhead_teeth
  let great_white_teeth : ℕ := 420
  great_white_teeth / sum_teeth = 2 := by
  sorry

end shark_teeth_multiple_l2227_222765


namespace phone_package_comparison_l2227_222704

/-- Represents the monthly bill for a phone package as a function of call duration. -/
structure PhonePackage where
  monthly_fee : ℝ
  call_fee : ℝ
  bill : ℝ → ℝ

/-- Package A with a monthly fee of 15 yuan and a call fee of 0.1 yuan per minute. -/
def package_a : PhonePackage :=
  { monthly_fee := 15
    call_fee := 0.1
    bill := λ x => 0.1 * x + 15 }

/-- Package B with no monthly fee and a call fee of 0.15 yuan per minute. -/
def package_b : PhonePackage :=
  { monthly_fee := 0
    call_fee := 0.15
    bill := λ x => 0.15 * x }

theorem phone_package_comparison :
  ∃ (x : ℝ),
    (x > 0) ∧
    (package_a.bill x = package_b.bill x) ∧
    (x = 300) ∧
    (∀ y : ℝ, y > x → package_a.bill y < package_b.bill y) :=
by sorry

end phone_package_comparison_l2227_222704


namespace motorboat_travel_time_l2227_222754

/-- The time taken for a motorboat to travel from pier X to pier Y downstream,
    given the conditions of the river journey problem. -/
theorem motorboat_travel_time (s r : ℝ) (h₁ : s > 0) (h₂ : r > 0) (h₃ : s > r) : 
  ∃ t : ℝ, t = (12 * (s - r)) / (s + r) ∧ 
    (s + r) * t + (s - r) * (12 - t) = 12 * r := by
  sorry

end motorboat_travel_time_l2227_222754


namespace quadratic_minimum_l2227_222768

theorem quadratic_minimum (x : ℝ) : x^2 + 8*x + 3 ≥ -13 ∧ 
  (x^2 + 8*x + 3 = -13 ↔ x = -4) := by
  sorry

end quadratic_minimum_l2227_222768


namespace imaginary_part_of_z_l2227_222771

theorem imaginary_part_of_z (z : ℂ) (h : z + z * Complex.I = 2) : 
  z.im = -1 := by sorry

end imaginary_part_of_z_l2227_222771


namespace congruent_count_l2227_222738

theorem congruent_count (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 3) (Finset.range 300)).card = 43 :=
by sorry

end congruent_count_l2227_222738


namespace root_sum_reciprocal_l2227_222723

theorem root_sum_reciprocal (p q r : ℂ) : 
  p^3 - p + 1 = 0 → q^3 - q + 1 = 0 → r^3 - r + 1 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) : ℂ) = -10 / 13 := by
  sorry

end root_sum_reciprocal_l2227_222723


namespace n_equals_six_l2227_222753

/-- The number of coins flipped simultaneously -/
def n : ℕ := sorry

/-- The probability of exactly two tails when flipping n coins -/
def prob_two_tails (n : ℕ) : ℚ := n * (n - 1) / (2^(n + 1))

/-- Theorem stating that n equals 6 when the probability of two tails is 5/32 -/
theorem n_equals_six : 
  (prob_two_tails n = 5/32) → n = 6 := by
  sorry

end n_equals_six_l2227_222753


namespace gray_trees_count_l2227_222708

/-- Represents a drone photograph of an area --/
structure Photograph where
  visible_trees : ℕ
  total_trees : ℕ

/-- Represents a set of three drone photographs of the same area --/
structure PhotoSet where
  photo1 : Photograph
  photo2 : Photograph
  photo3 : Photograph
  equal_total : photo1.total_trees = photo2.total_trees ∧ photo2.total_trees = photo3.total_trees

/-- Calculates the number of trees in gray areas given a set of three photographs --/
def gray_trees (photos : PhotoSet) : ℕ :=
  (photos.photo1.total_trees - photos.photo1.visible_trees) +
  (photos.photo2.total_trees - photos.photo2.visible_trees)

/-- Theorem stating that for the given set of photographs, the number of trees in gray areas is 26 --/
theorem gray_trees_count (photos : PhotoSet)
  (h1 : photos.photo1.visible_trees = 100)
  (h2 : photos.photo2.visible_trees = 90)
  (h3 : photos.photo3.visible_trees = 82) :
  gray_trees photos = 26 := by
  sorry


end gray_trees_count_l2227_222708


namespace guest_payment_divisibility_l2227_222751

theorem guest_payment_divisibility (A : Nat) (h1 : A < 10) : 
  (100 + 10 * A + 2) % 11 = 0 ↔ A = 3 := by
  sorry

end guest_payment_divisibility_l2227_222751


namespace f_increasing_after_3_l2227_222702

def f (x : ℝ) := 2 * (x - 3)^2 - 1

theorem f_increasing_after_3 :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 3 → x₂ ≥ 3 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end f_increasing_after_3_l2227_222702


namespace prob_less_than_five_and_even_is_one_third_l2227_222746

/-- The probability of rolling a number less than 5 on a six-sided die -/
def prob_less_than_five : ℚ := 4 / 6

/-- The probability of rolling an even number on a six-sided die -/
def prob_even : ℚ := 3 / 6

/-- The probability of rolling a number less than 5 on the first die
    and an even number on the second die -/
def prob_less_than_five_and_even : ℚ := prob_less_than_five * prob_even

theorem prob_less_than_five_and_even_is_one_third :
  prob_less_than_five_and_even = 1 / 3 := by
  sorry

end prob_less_than_five_and_even_is_one_third_l2227_222746
