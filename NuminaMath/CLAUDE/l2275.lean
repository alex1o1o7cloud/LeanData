import Mathlib

namespace car_speed_comparison_l2275_227596

theorem car_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v + w) / 3 := by
  sorry

end car_speed_comparison_l2275_227596


namespace triangle_k_values_l2275_227545

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the vectors
def vectorAB (t : Triangle) : ℝ × ℝ := (t.B.1 - t.A.1, t.B.2 - t.A.2)
def vectorAC (t : Triangle) (k : ℝ) : ℝ × ℝ := (t.C.1 - t.A.1, t.C.2 - t.A.2)

-- Define the dot product
def dotProduct (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for a right angle
def hasRightAngle (t : Triangle) (k : ℝ) : Prop :=
  dotProduct (vectorAB t) (vectorAC t k) = 0 ∨
  dotProduct (vectorAB t) (1, k - 3) = 0 ∨
  dotProduct (vectorAC t k) ((-1 : ℝ), k - 3) = 0

-- The main theorem
theorem triangle_k_values (t : Triangle) (k : ℝ) 
  (h1 : vectorAB t = (2, 3))
  (h2 : vectorAC t k = (1, k))
  (h3 : hasRightAngle t k) :
  k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13)/2 ∨ k = (3 - Real.sqrt 13)/2 :=
sorry

end triangle_k_values_l2275_227545


namespace binomial_square_constant_l2275_227580

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + c = (x + a)^2) → c = 5625 := by
  sorry

end binomial_square_constant_l2275_227580


namespace circle_radius_problem_l2275_227565

theorem circle_radius_problem (r : ℝ) (h : r > 0) :
  3 * (2 * 2 * Real.pi * r) = 3 * (Real.pi * r ^ 2) → r = 4 := by
  sorry

end circle_radius_problem_l2275_227565


namespace airport_exchange_rate_fraction_l2275_227576

def official_rate : ℚ := 5 / 1
def willie_euros : ℚ := 70
def airport_dollars : ℚ := 10

theorem airport_exchange_rate_fraction : 
  (airport_dollars / (willie_euros / official_rate)) = 5 / 7 := by
  sorry

end airport_exchange_rate_fraction_l2275_227576


namespace x_minus_y_equals_106_over_21_l2275_227547

theorem x_minus_y_equals_106_over_21 (x y : ℚ) : 
  x + 2*y = 16/3 → 5*x + 3*y = 26 → x - y = 106/21 := by
  sorry

end x_minus_y_equals_106_over_21_l2275_227547


namespace two_mice_boring_l2275_227593

/-- The sum of distances bored by two mice in n days -/
def S (n : ℕ) : ℚ :=
  let big_mouse := 2^n - 1  -- Sum of geometric sequence with a₁ = 1, r = 2
  let small_mouse := 2 - 1 / 2^(n-1)  -- Sum of geometric sequence with a₁ = 1, r = 1/2
  big_mouse + small_mouse

theorem two_mice_boring (n : ℕ) : S n = 2^n - 1/2^(n-1) + 1 := by
  sorry

end two_mice_boring_l2275_227593


namespace largest_angle_in_triangle_l2275_227513

theorem largest_angle_in_triangle (a b c : ℝ) (h1 : a + 2*b + 2*c = a^2) (h2 : a + 2*b - 2*c = -3) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A ≤ 120 ∧ B ≤ 120 ∧ C = 120 := by
  sorry

end largest_angle_in_triangle_l2275_227513


namespace max_power_under_500_l2275_227508

theorem max_power_under_500 :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 1 ∧ x^y < 500 ∧
    (∀ (a b : ℕ), a > 0 → b > 1 → a^b < 500 → a^b ≤ x^y) ∧
    x = 22 ∧ y = 2 ∧ x + y = 24 :=
by sorry

end max_power_under_500_l2275_227508


namespace cookies_sum_l2275_227570

/-- The number of cookies Mona brought -/
def mona_cookies : ℕ := 20

/-- The number of cookies Jasmine brought -/
def jasmine_cookies : ℕ := mona_cookies - 5

/-- The number of cookies Rachel brought -/
def rachel_cookies : ℕ := jasmine_cookies + 10

/-- The total number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies : ℕ := mona_cookies + jasmine_cookies + rachel_cookies

theorem cookies_sum : total_cookies = 60 := by
  sorry

end cookies_sum_l2275_227570


namespace probability_five_blue_marbles_in_eight_draws_l2275_227518

/-- The probability of drawing exactly k blue marbles in n draws with replacement -/
def probability_k_blue_marbles (total_marbles blue_marbles k n : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (blue_marbles / total_marbles : ℚ) ^ k * 
  ((total_marbles - blue_marbles) / total_marbles : ℚ) ^ (n - k)

/-- The probability of drawing exactly 5 blue marbles in 8 draws with replacement
    from a bag containing 9 blue marbles and 6 red marbles -/
theorem probability_five_blue_marbles_in_eight_draws : 
  probability_k_blue_marbles 15 9 5 8 = 108864 / 390625 := by
  sorry

end probability_five_blue_marbles_in_eight_draws_l2275_227518


namespace length_of_PQ_l2275_227546

-- Define the points and lines
def R : ℝ × ℝ := (10, 8)
def line1 (x y : ℝ) : Prop := 7 * y = 9 * x
def line2 (x y : ℝ) : Prop := 12 * y = 5 * x

-- Define the theorem
theorem length_of_PQ : 
  ∀ (P Q : ℝ × ℝ),
  -- R is the midpoint of PQ
  (P.1 + Q.1) / 2 = R.1 ∧ (P.2 + Q.2) / 2 = R.2 ∧
  -- P is on line1
  line1 P.1 P.2 ∧
  -- Q is on line2
  line2 Q.1 Q.2 →
  -- The length of PQ is 4√134481/73
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 * Real.sqrt 134481 / 73 := by
  sorry

end length_of_PQ_l2275_227546


namespace number_difference_problem_l2275_227538

theorem number_difference_problem : ∃ x : ℚ, x - (3/5) * x = 60 ∧ x = 150 := by
  sorry

end number_difference_problem_l2275_227538


namespace domain_and_even_function_implies_a_eq_neg_one_l2275_227575

/-- A function is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem domain_and_even_function_implies_a_eq_neg_one
  (a : ℝ)
  (f : ℝ → ℝ)
  (h_domain : Set.Ioo (4*a - 3) (3 - 2*a^2) = Set.range f)
  (h_even : IsEven (fun x ↦ f (2*x - 3))) :
  a = -1 := by
sorry

end domain_and_even_function_implies_a_eq_neg_one_l2275_227575


namespace m_range_characterization_l2275_227503

/-- 
Given a real number m, this theorem states that m is in the open interval (2, 3)
if and only if both of the following conditions are satisfied:
1. The equation x^2 + mx + 1 = 0 has two distinct negative roots.
2. The equation 4x^2 + 4(m - 2)x + 1 = 0 has no real roots.
-/
theorem m_range_characterization (m : ℝ) : 
  (2 < m ∧ m < 3) ↔ 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ∧
  (∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0) :=
by sorry

end m_range_characterization_l2275_227503


namespace sara_jim_savings_equality_l2275_227514

def sara_weekly_savings (S : ℚ) : ℚ := S

theorem sara_jim_savings_equality (S : ℚ) : 
  (4100 : ℚ) + 820 * sara_weekly_savings S = 15 * 820 → S = 10 := by
  sorry

end sara_jim_savings_equality_l2275_227514


namespace clown_balloons_l2275_227531

/-- The number of balloons a clown has after blowing up two sets of balloons -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the clown has 60 balloons after blowing up 47 and then 13 more -/
theorem clown_balloons :
  total_balloons 47 13 = 60 := by
  sorry

end clown_balloons_l2275_227531


namespace factor_theorem_p_factorization_p_factorization_q_l2275_227599

-- Define the polynomials
def p (x : ℝ) := 6 * x^2 - x - 5
def q (x : ℝ) := x^3 - 7 * x + 6

-- State the theorems
theorem factor_theorem_p : ∃ (r : ℝ → ℝ), ∀ x, p x = (x - 1) * r x := by sorry

theorem factorization_p : ∀ x, p x = (x - 1) * (6 * x + 5) := by sorry

theorem factorization_q : ∀ x, q x = (x - 1) * (x + 3) * (x - 2) := by sorry

-- Given condition
axiom p_root : p 1 = 0

end factor_theorem_p_factorization_p_factorization_q_l2275_227599


namespace matrix_power_2023_l2275_227509

theorem matrix_power_2023 :
  let A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 0, 1]
  A ^ 2023 = !![1, 2023; 0, 1] := by sorry

end matrix_power_2023_l2275_227509


namespace smallest_multiple_thirty_two_satisfies_smallest_multiple_is_32_l2275_227584

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 900 * x % 1152 = 0 → x ≥ 32 := by
  sorry

theorem thirty_two_satisfies : 900 * 32 % 1152 = 0 := by
  sorry

theorem smallest_multiple_is_32 : 
  ∃ (x : ℕ), x > 0 ∧ 900 * x % 1152 = 0 ∧ ∀ (y : ℕ), y > 0 ∧ 900 * y % 1152 = 0 → x ≤ y := by
  sorry

end smallest_multiple_thirty_two_satisfies_smallest_multiple_is_32_l2275_227584


namespace complement_of_A_l2275_227523

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem complement_of_A (A B : Set ℕ) 
  (h1 : A ∪ B = {1, 2, 3, 4, 5})
  (h2 : A ∩ B = {3, 4, 5}) :
  (U \ A) = {6} := by
  sorry

end complement_of_A_l2275_227523


namespace rachel_albums_count_l2275_227533

/-- The number of songs per album -/
def songs_per_album : ℕ := 2

/-- The total number of songs Rachel bought -/
def total_songs : ℕ := 16

/-- The number of albums Rachel bought -/
def albums_bought : ℕ := total_songs / songs_per_album

theorem rachel_albums_count : albums_bought = 8 := by
  sorry

end rachel_albums_count_l2275_227533


namespace inverse_contrapositive_negation_l2275_227500

/-- Given a proposition p, its inverse q, and its contrapositive r,
    prove that q and r are negations of each other. -/
theorem inverse_contrapositive_negation (p q r : Prop) 
  (h_inverse : q ↔ (p → q))
  (h_contrapositive : r ↔ (¬q → ¬p)) :
  q ↔ ¬r := by
  sorry

end inverse_contrapositive_negation_l2275_227500


namespace exists_x_squared_plus_two_x_plus_one_nonpositive_l2275_227516

theorem exists_x_squared_plus_two_x_plus_one_nonpositive :
  ∃ x : ℝ, x^2 + 2*x + 1 ≤ 0 := by sorry

end exists_x_squared_plus_two_x_plus_one_nonpositive_l2275_227516


namespace regular_polygon_sides_l2275_227589

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (∀ angle : ℝ, angle = 150 ∧ (n * angle : ℝ) = 180 * (n - 2 : ℝ)) → n = 12 := by
  sorry

end regular_polygon_sides_l2275_227589


namespace right_triangle_squares_problem_l2275_227520

theorem right_triangle_squares_problem (x : ℝ) : 
  (3 * x)^2 + (6 * x)^2 + (1/2 * 3 * x * 6 * x) = 1200 → x = (10 * Real.sqrt 2) / 3 := by
  sorry

end right_triangle_squares_problem_l2275_227520


namespace liters_to_gallons_conversion_l2275_227564

/-- Conversion factor from liters to gallons -/
def liters_to_gallons : ℝ := 0.26

/-- The volume in liters -/
def volume_in_liters : ℝ := 2.5

/-- Theorem stating that 2.5 liters is equal to 0.65 gallons -/
theorem liters_to_gallons_conversion :
  volume_in_liters * liters_to_gallons = 0.65 := by
  sorry

end liters_to_gallons_conversion_l2275_227564


namespace abc_inequality_l2275_227595

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a + b + c + a * b + b * c + c * a ≥ 6 := by
  sorry

end abc_inequality_l2275_227595


namespace complex_number_location_l2275_227560

theorem complex_number_location :
  let z : ℂ := 1 / (2 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_location_l2275_227560


namespace arithmetic_mean_of_special_set_l2275_227507

def S (n : ℕ) : ℕ := 11 * (10^n - 1) / 9

def arithmetic_mean (nums : List ℕ) : ℚ :=
  (nums.sum : ℚ) / nums.length

theorem arithmetic_mean_of_special_set :
  let nums := List.range 9
  let special_set := nums.map (fun i => S (i + 1))
  let mean := arithmetic_mean special_set
  ∃ (n : ℕ),
    n = ⌊mean⌋ ∧
    n ≥ 100000000 ∧ n < 1000000000 ∧
    (List.range 10).all (fun d => d ≠ 0 → (n / 10^d % 10 ≠ n / 10^(d+1) % 10)) ∧
    n % 10 ≠ 0 ∧
    (n / 10 % 10) ≠ 0 ∧
    (n / 100 % 10) ≠ 0 ∧
    (n / 1000 % 10) ≠ 0 ∧
    (n / 10000 % 10) ≠ 0 ∧
    (n / 100000 % 10) ≠ 0 ∧
    (n / 1000000 % 10) ≠ 0 ∧
    (n / 10000000 % 10) ≠ 0 ∧
    (n / 100000000 % 10) ≠ 0 := by
  sorry

end arithmetic_mean_of_special_set_l2275_227507


namespace last_two_digits_sum_l2275_227578

theorem last_two_digits_sum (n : ℕ) : n = 30 → (7^n + 13^n) % 100 = 18 := by
  sorry

end last_two_digits_sum_l2275_227578


namespace cos_36_minus_cos_72_eq_half_l2275_227592

theorem cos_36_minus_cos_72_eq_half :
  Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end cos_36_minus_cos_72_eq_half_l2275_227592


namespace product_of_numbers_with_given_sum_and_difference_l2275_227556

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 6 → x * y = 216 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l2275_227556


namespace arcsin_one_half_equals_pi_sixth_l2275_227561

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_equals_pi_sixth_l2275_227561


namespace worker_a_payment_share_l2275_227548

/-- Calculates the share of payment for worker A given the total days needed for each worker and the total payment -/
def worker_a_share (days_a days_b : ℕ) (total_payment : ℚ) : ℚ :=
  let work_rate_a := 1 / days_a
  let work_rate_b := 1 / days_b
  let combined_rate := work_rate_a + work_rate_b
  let a_share_ratio := work_rate_a / combined_rate
  a_share_ratio * total_payment

/-- Theorem stating that worker A's share is 89.55 given the problem conditions -/
theorem worker_a_payment_share :
  worker_a_share 12 18 (149.25 : ℚ) = (8955 : ℚ) / 100 := by
  sorry

#eval worker_a_share 12 18 (149.25 : ℚ)

end worker_a_payment_share_l2275_227548


namespace unripe_apples_correct_l2275_227542

/-- Calculates the number of unripe apples given the total number of apples picked,
    the number of pies that can be made, and the number of apples needed per pie. -/
def unripe_apples (total_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total_apples - (num_pies * apples_per_pie)

/-- Proves that the number of unripe apples is correct for the given scenario. -/
theorem unripe_apples_correct : unripe_apples 34 7 4 = 6 := by
  sorry

end unripe_apples_correct_l2275_227542


namespace gym_equipment_cost_l2275_227529

/-- The cost of replacing all cardio machines in a global gym chain --/
theorem gym_equipment_cost (num_gyms : ℕ) (num_bikes : ℕ) (num_treadmills : ℕ) (num_ellipticals : ℕ)
  (treadmill_cost_factor : ℚ) (elliptical_cost_factor : ℚ) (total_cost : ℚ) :
  num_gyms = 20 →
  num_bikes = 10 →
  num_treadmills = 5 →
  num_ellipticals = 5 →
  treadmill_cost_factor = 3/2 →
  elliptical_cost_factor = 2 →
  total_cost = 455000 →
  ∃ (bike_cost : ℚ),
    bike_cost = 700 ∧
    total_cost = num_gyms * (num_bikes * bike_cost +
                             num_treadmills * treadmill_cost_factor * bike_cost +
                             num_ellipticals * elliptical_cost_factor * treadmill_cost_factor * bike_cost) :=
by sorry

end gym_equipment_cost_l2275_227529


namespace nested_radical_solution_l2275_227521

theorem nested_radical_solution :
  ∃ x : ℝ, x = Real.sqrt (3 - x) → x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_radical_solution_l2275_227521


namespace smallest_reducible_even_l2275_227562

def is_reducible (n : ℕ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (15 * n - 7) % k = 0 ∧ (22 * n - 5) % k = 0

theorem smallest_reducible_even : 
  (∀ n : ℕ, n > 2013 → n % 2 = 0 → is_reducible n → n ≥ 2144) ∧ 
  (2144 > 2013 ∧ 2144 % 2 = 0 ∧ is_reducible 2144) :=
sorry

end smallest_reducible_even_l2275_227562


namespace break_even_point_l2275_227525

/-- The break-even point for a plastic handle molding company -/
theorem break_even_point
  (cost_per_handle : ℝ)
  (fixed_cost : ℝ)
  (selling_price : ℝ)
  (h1 : cost_per_handle = 0.60)
  (h2 : fixed_cost = 7640)
  (h3 : selling_price = 4.60) :
  ∃ x : ℕ, x = 1910 ∧ selling_price * x = fixed_cost + cost_per_handle * x :=
by sorry

end break_even_point_l2275_227525


namespace collinearity_iff_harmonic_l2275_227553

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the incidence relation
variable (incident : Point → Line → Prop)

-- Define the collinearity relation
variable (collinear : Point → Point → Point → Prop)

-- Define the harmonic relation
variable (harmonic : Point → Point → Point → Point → Prop)

-- Define the points and lines
variable (A B C D E F H P X Y : Point)
variable (hA hB gA gB : Line)

-- Define the geometric conditions
variable (h1 : incident A hA)
variable (h2 : incident A gA)
variable (h3 : incident B hB)
variable (h4 : incident B gB)
variable (h5 : incident C hA ∧ incident C gB)
variable (h6 : incident D hB ∧ incident D gA)
variable (h7 : incident E gA ∧ incident E gB)
variable (h8 : incident F hA ∧ incident F hB)
variable (h9 : incident P hB)
variable (h10 : incident H gA)
variable (h11 : ∃ CP EF, incident X CP ∧ incident X EF ∧ incident C CP ∧ incident P CP ∧ incident E EF ∧ incident F EF)
variable (h12 : ∃ EP HF, incident Y EP ∧ incident Y HF ∧ incident E EP ∧ incident P EP ∧ incident H HF ∧ incident F HF)

-- State the theorem
theorem collinearity_iff_harmonic :
  collinear X Y B ↔ harmonic A H E D :=
sorry

end collinearity_iff_harmonic_l2275_227553


namespace min_fence_length_is_650_l2275_227574

/-- Represents a triangular grid with side length 50 meters -/
structure TriangularGrid where
  side_length : ℝ
  side_length_eq : side_length = 50

/-- Represents the number of paths between cabbage and goat areas -/
def num_paths : ℕ := 13

/-- The minimum total length of fences required to separate cabbage from goats -/
def min_fence_length (grid : TriangularGrid) : ℝ :=
  (num_paths : ℝ) * grid.side_length

/-- Theorem stating the minimum fence length required -/
theorem min_fence_length_is_650 (grid : TriangularGrid) :
  min_fence_length grid = 650 := by
  sorry

#check min_fence_length_is_650

end min_fence_length_is_650_l2275_227574


namespace square_of_1027_l2275_227515

theorem square_of_1027 : (1027 : ℕ)^2 = 1054729 := by
  sorry

end square_of_1027_l2275_227515


namespace tangent_sum_identity_l2275_227557

theorem tangent_sum_identity (α β γ : Real) (h : α + β + γ = Real.pi / 2) :
  Real.tan α * Real.tan β + Real.tan β * Real.tan γ + Real.tan γ * Real.tan α = 1 := by
  sorry

end tangent_sum_identity_l2275_227557


namespace female_officers_count_l2275_227590

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 500 →
  female_on_duty_ratio = 1/4 →
  female_ratio = 1/2 →
  (female_on_duty_ratio * (total_on_duty * female_ratio)) / female_on_duty_ratio = 1000 := by
  sorry

end female_officers_count_l2275_227590


namespace probability_consecutive_cards_l2275_227539

/-- A type representing the cards labeled A, B, C, D, E -/
inductive Card : Type
  | A | B | C | D | E

/-- A function to check if two cards are consecutive -/
def consecutive (c1 c2 : Card) : Bool :=
  match c1, c2 with
  | Card.A, Card.B | Card.B, Card.A => true
  | Card.B, Card.C | Card.C, Card.B => true
  | Card.C, Card.D | Card.D, Card.C => true
  | Card.D, Card.E | Card.E, Card.D => true
  | _, _ => false

/-- The total number of ways to choose 2 cards from 5 -/
def totalChoices : Nat := 10

/-- The number of ways to choose 2 consecutive cards -/
def consecutiveChoices : Nat := 4

/-- Theorem stating the probability of drawing two consecutive cards -/
theorem probability_consecutive_cards :
  (consecutiveChoices : ℚ) / totalChoices = 2 / 5 := by
  sorry


end probability_consecutive_cards_l2275_227539


namespace max_supervisors_is_three_l2275_227512

/-- Represents the number of years in a supervisor's term -/
def termLength : ℕ := 4

/-- Represents the gap year between supervisors -/
def gapYear : ℕ := 1

/-- Represents the total period in years -/
def totalPeriod : ℕ := 15

/-- Calculates the maximum number of supervisors that can be hired -/
def maxSupervisors : ℕ := (totalPeriod + gapYear) / (termLength + gapYear)

theorem max_supervisors_is_three :
  maxSupervisors = 3 :=
sorry

end max_supervisors_is_three_l2275_227512


namespace range_of_a_for_increasing_f_l2275_227528

/-- Given that f(x) = e^(2x) - ae^x + 2x is an increasing function on ℝ, 
    prove that the range of a is (-∞, 4]. -/
theorem range_of_a_for_increasing_f (a : ℝ) : 
  (∀ x : ℝ, Monotone (fun x => Real.exp (2 * x) - a * Real.exp x + 2 * x)) →
  a ≤ 4 :=
by sorry

end range_of_a_for_increasing_f_l2275_227528


namespace circle_line_intersection_l2275_227587

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 15 = 0

-- Define the line
def line (x y k : ℝ) : Prop := y = k*x - 2

-- Define the condition for common points
def has_common_points (k : ℝ) : Prop :=
  ∃ x y : ℝ, line x y k ∧ 
    ∃ x' y' : ℝ, circle_C x' y' ∧ 
      (x - x')^2 + (y - y')^2 ≤ 4

-- The main theorem
theorem circle_line_intersection (k : ℝ) :
  has_common_points k → -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end circle_line_intersection_l2275_227587


namespace sum_of_local_values_equals_number_l2275_227511

/-- The local value of a digit in a number -/
def local_value (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

/-- The number we're considering -/
def number : ℕ := 2345

/-- Theorem: The sum of local values of digits in 2345 equals 2345 -/
theorem sum_of_local_values_equals_number :
  local_value 2 3 + local_value 3 2 + local_value 4 1 + local_value 5 0 = number := by
  sorry

end sum_of_local_values_equals_number_l2275_227511


namespace simplify_expression_solve_inequality_system_l2275_227543

-- Part 1: Simplification
theorem simplify_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (2 - (x - 1) / (x + 2)) / ((x^2 + 10*x + 25) / (x^2 - 4)) = (x - 2) / (x + 5) := by
  sorry

-- Part 2: Inequality System
theorem solve_inequality_system (x : ℝ) :
  (2*x + 7 > 3 ∧ (x + 1) / 3 > (x - 1) / 2) ↔ -2 < x ∧ x < 5 := by
  sorry

end simplify_expression_solve_inequality_system_l2275_227543


namespace saltwater_solution_volume_l2275_227526

theorem saltwater_solution_volume :
  -- Initial conditions
  ∀ x : ℝ,
  let initial_salt_volume := 0.20 * x
  let evaporated_volume := 0.25 * x
  let remaining_volume := x - evaporated_volume
  let added_water := 6
  let added_salt := 12
  let final_volume := remaining_volume + added_water + added_salt
  let final_salt_volume := initial_salt_volume + added_salt
  -- Final salt concentration condition
  final_salt_volume / final_volume = 1/3 →
  -- Conclusion
  x = 120 := by
  sorry

end saltwater_solution_volume_l2275_227526


namespace currency_comparisons_l2275_227550

-- Define the conversion rate from jiao to yuan
def jiao_to_yuan (jiao : ℚ) : ℚ := jiao / 10

-- Define the theorem
theorem currency_comparisons :
  (2.3 < 3.2) ∧
  (10 > 9.9) ∧
  (1 + jiao_to_yuan 6 = 1.6) ∧
  (15 * 4 < 14 * 5) :=
by sorry

end currency_comparisons_l2275_227550


namespace difference_of_squares_l2275_227572

theorem difference_of_squares (t : ℝ) : t^2 - 121 = (t - 11) * (t + 11) := by
  sorry

end difference_of_squares_l2275_227572


namespace circle_center_l2275_227582

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form (x - h)² + (y - k)² = r² -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given circle equation -/
def given_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem circle_center :
  ∃ (c : Circle), (∀ x y : ℝ, given_equation x y ↔ c.equation x y) ∧ c.center = (1, 1) := by
  sorry

end circle_center_l2275_227582


namespace negation_of_existence_negation_of_proposition_l2275_227534

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x < 2) ↔ (∀ x : ℝ, x ≥ 2) :=
by sorry

end negation_of_existence_negation_of_proposition_l2275_227534


namespace sufficient_not_necessary_l2275_227583

/-- The set of real numbers x such that x^2 - 9 > 0 -/
def A : Set ℝ := {x | x^2 - 9 > 0}

/-- The set of real numbers x such that x^2 - 5/6*x + 1/6 > 0 -/
def B : Set ℝ := {x | x^2 - 5/6*x + 1/6 > 0}

/-- Theorem stating that A is a subset of B and there exists an element in B that is not in A -/
theorem sufficient_not_necessary : A ⊆ B ∧ ∃ x, x ∈ B ∧ x ∉ A :=
sorry

end sufficient_not_necessary_l2275_227583


namespace multiple_subtracted_l2275_227566

theorem multiple_subtracted (a b : ℝ) (h1 : a / b = 4 / 1) 
  (h2 : ∃ x : ℝ, (a - x * b) / (2 * a - b) = 0.14285714285714285) : 
  ∃ x : ℝ, (a - x * b) / (2 * a - b) = 0.14285714285714285 ∧ x = 3 := by
  sorry

end multiple_subtracted_l2275_227566


namespace jellybean_problem_l2275_227536

theorem jellybean_problem : ∃ (n : ℕ), n ≥ 150 ∧ n % 19 = 17 ∧ ∀ (m : ℕ), m ≥ 150 ∧ m % 19 = 17 → m ≥ n :=
by sorry

end jellybean_problem_l2275_227536


namespace probability_of_red_is_half_l2275_227577

/-- A cube with a specific color distribution -/
structure ColoredCube where
  total_faces : ℕ
  red_faces : ℕ
  yellow_faces : ℕ
  green_faces : ℕ
  tricolor_faces : ℕ

/-- The probability of a specific color facing up when throwing the cube -/
def probability_of_color (cube : ColoredCube) (color_faces : ℕ) : ℚ :=
  color_faces / cube.total_faces

/-- Our specific cube with the given color distribution -/
def our_cube : ColoredCube :=
  { total_faces := 6
  , red_faces := 2
  , yellow_faces := 2
  , green_faces := 1
  , tricolor_faces := 1 }

/-- Theorem stating that the probability of red facing up is 1/2 -/
theorem probability_of_red_is_half :
  probability_of_color our_cube (our_cube.red_faces + our_cube.tricolor_faces) = 1/2 := by
  sorry

end probability_of_red_is_half_l2275_227577


namespace inequality_proof_l2275_227594

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end inequality_proof_l2275_227594


namespace adults_trekking_l2275_227541

/-- The number of adults who went for trekking -/
def numAdults : ℕ := 56

/-- The number of children who went for trekking -/
def numChildren : ℕ := 70

/-- The number of adults the meal can feed -/
def mealAdults : ℕ := 70

/-- The number of children the meal can feed -/
def mealChildren : ℕ := 90

/-- The number of adults who have already eaten -/
def adultsEaten : ℕ := 14

/-- The number of children that can be fed with remaining food after some adults eat -/
def remainingChildren : ℕ := 72

theorem adults_trekking :
  numAdults = mealAdults - adultsEaten ∧
  numChildren = 70 ∧
  mealAdults = 70 ∧
  mealChildren = 90 ∧
  adultsEaten = 14 ∧
  remainingChildren = 72 ∧
  mealChildren = remainingChildren + adultsEaten * mealChildren / mealAdults :=
by sorry

end adults_trekking_l2275_227541


namespace fraction_of_ripe_oranges_eaten_l2275_227591

def total_oranges : ℕ := 96
def ripe_oranges : ℕ := total_oranges / 2
def unripe_oranges : ℕ := total_oranges - ripe_oranges
def eaten_unripe : ℕ := unripe_oranges / 8
def uneaten_oranges : ℕ := 78

theorem fraction_of_ripe_oranges_eaten :
  (total_oranges - uneaten_oranges - eaten_unripe) / ripe_oranges = 1 / 4 :=
by sorry

end fraction_of_ripe_oranges_eaten_l2275_227591


namespace tank_width_proof_l2275_227563

/-- Proves that a tank with given dimensions and plastering cost has a width of 12 meters -/
theorem tank_width_proof (length depth : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 25 →
  depth = 6 →
  cost_per_sqm = 0.30 →
  total_cost = 223.2 →
  ∃ width : ℝ,
    width = 12 ∧
    total_cost = cost_per_sqm * (length * width + 2 * (length * depth + width * depth)) :=
by sorry

end tank_width_proof_l2275_227563


namespace root_sum_reciprocal_l2275_227544

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^3 - 15*x^2 + 50*x - 60

-- Define the theorem
theorem root_sum_reciprocal (p q r A B C : ℝ) :
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →  -- p, q, r are distinct
  (poly p = 0 ∧ poly q = 0 ∧ poly r = 0) →  -- p, q, r are roots of poly
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 15*s^2 + 50*s - 60) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 135 :=
by sorry

end root_sum_reciprocal_l2275_227544


namespace inequality_solution_set_l2275_227522

theorem inequality_solution_set (x : ℝ) : 2 ≤ x / (2 * x - 1) ∧ x / (2 * x - 1) < 5 ↔ x ∈ Set.Ioo (5/9 : ℝ) (2/3 : ℝ) ∪ {2/3} := by
  sorry

end inequality_solution_set_l2275_227522


namespace min_box_value_l2275_227598

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + Box * x + 15) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∃ min_Box : ℤ, 
    (∀ a' b' Box' : ℤ, 
      (∀ x, (a' * x + b') * (b' * x + a') = 15 * x^2 + Box' * x + 15) →
      a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' →
      Box' ≥ min_Box) ∧
    min_Box = 34 ∧
    ((a = 3 ∧ b = 5) ∨ (a = -3 ∧ b = -5) ∨ (a = 5 ∧ b = 3) ∨ (a = -5 ∧ b = -3))) :=
by sorry

end min_box_value_l2275_227598


namespace james_bed_purchase_l2275_227505

/-- The price James pays for a bed and bed frame with a discount -/
theorem james_bed_purchase (bed_frame_price : ℝ) (bed_price_multiplier : ℕ) (discount_percentage : ℝ) : 
  bed_frame_price = 75 →
  bed_price_multiplier = 10 →
  discount_percentage = 0.2 →
  (bed_frame_price + bed_frame_price * bed_price_multiplier) * (1 - discount_percentage) = 660 :=
by sorry

end james_bed_purchase_l2275_227505


namespace domain_of_sqrt_sin_minus_cos_l2275_227585

open Real

theorem domain_of_sqrt_sin_minus_cos (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (sin x - cos x)) ↔
  (∃ k : ℤ, 2 * k * π + π / 4 ≤ x ∧ x ≤ 2 * k * π + 5 * π / 4) :=
by sorry

end domain_of_sqrt_sin_minus_cos_l2275_227585


namespace ellipse_eccentricity_l2275_227530

/-- Represents an ellipse with semi-major axis 2 and semi-minor axis b -/
structure Ellipse (b : ℝ) :=
  (equation : ℝ → ℝ → Prop)
  (b_pos : b > 0)

/-- Represents a point on the ellipse -/
structure EllipsePoint (E : Ellipse b) :=
  (x y : ℝ)
  (on_ellipse : E.equation x y)

/-- The left focus of the ellipse -/
def left_focus (E : Ellipse b) : ℝ × ℝ := sorry

/-- The right focus of the ellipse -/
def right_focus (E : Ellipse b) : ℝ × ℝ := sorry

/-- A line passing through the left focus -/
structure FocalLine (E : Ellipse b) :=
  (passes_through_left_focus : Prop)

/-- Intersection points of a focal line with the ellipse -/
def intersection_points (E : Ellipse b) (l : FocalLine E) : EllipsePoint E × EllipsePoint E := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Maximum sum of distances from intersection points to the right focus -/
def max_sum_distances (E : Ellipse b) : ℝ := sorry

/-- Eccentricity of the ellipse -/
def eccentricity (E : Ellipse b) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_eccentricity (b : ℝ) (E : Ellipse b) :
  max_sum_distances E = 5 → eccentricity E = 1/2 := by sorry

end ellipse_eccentricity_l2275_227530


namespace union_of_M_and_N_l2275_227506

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {y | y^2 + y = 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end union_of_M_and_N_l2275_227506


namespace book_arrangement_count_l2275_227517

def num_math_books : ℕ := 4
def num_history_books : ℕ := 6
def total_books : ℕ := num_math_books + num_history_books

def arrange_books : ℕ := num_math_books * (num_math_books - 1) * Nat.factorial (total_books - 2)

theorem book_arrangement_count :
  arrange_books = 145152 := by
  sorry

end book_arrangement_count_l2275_227517


namespace modulus_z₂_l2275_227571

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- State the conditions
axiom z₁_condition : (z₁ - 2) * Complex.I = 1 + Complex.I
axiom z₂_imag_part : z₂.im = 2
axiom product_real : (z₁ * z₂).im = 0

-- State the theorem
theorem modulus_z₂ : Complex.abs z₂ = 2 * Real.sqrt 10 := by
  sorry

end modulus_z₂_l2275_227571


namespace smallest_sum_ABAb_l2275_227527

/-- Represents a digit in base 4 -/
def Base4Digit := Fin 4

theorem smallest_sum_ABAb (A B : Base4Digit) (b : ℕ) : 
  A ≠ B →
  b > 5 →
  16 * A.val + 4 * B.val + A.val = 3 * b + 3 →
  ∀ (A' B' : Base4Digit) (b' : ℕ),
    A' ≠ B' →
    b' > 5 →
    16 * A'.val + 4 * B'.val + A'.val = 3 * b' + 3 →
    A.val + B.val + b ≤ A'.val + B'.val + b' →
  A.val + B.val + b = 8 :=
sorry

end smallest_sum_ABAb_l2275_227527


namespace parabola_r_value_l2275_227559

/-- A parabola in the xy-plane defined by x = py^2 + qy + r -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (para : Parabola) (y : ℝ) : ℝ :=
  para.p * y^2 + para.q * y + para.r

theorem parabola_r_value (para : Parabola) :
  para.x_coord 4 = 5 →
  para.x_coord 6 = 3 →
  para.x_coord 0 = 3 →
  para.r = 3 := by
  sorry

end parabola_r_value_l2275_227559


namespace gunther_tractor_payment_l2275_227579

/-- Calculates the monthly payment for a loan given the total amount and loan term in years -/
def monthly_payment (total_amount : ℕ) (years : ℕ) : ℚ :=
  (total_amount : ℚ) / (years * 12 : ℚ)

/-- Proves that for a $9000 loan over 5 years, the monthly payment is $150 -/
theorem gunther_tractor_payment :
  monthly_payment 9000 5 = 150 := by
  sorry

end gunther_tractor_payment_l2275_227579


namespace scientific_notation_505000_l2275_227588

theorem scientific_notation_505000 :
  505000 = 5.05 * (10 ^ 5) := by
  sorry

end scientific_notation_505000_l2275_227588


namespace fifth_term_value_l2275_227581

/-- Given a sequence {aₙ} with sum of first n terms Sₙ = 2n(n+1), prove a₅ = 20 -/
theorem fifth_term_value (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h : ∀ n : ℕ, S n = 2 * n * (n + 1)) : 
  a 5 = 20 := by
  sorry

end fifth_term_value_l2275_227581


namespace remainder_of_difference_l2275_227524

theorem remainder_of_difference (s t : ℕ) (hs : s > 0) (ht : t > 0) 
  (h_s_mod : s % 6 = 2) (h_t_mod : t % 6 = 3) (h_s_gt_t : s > t) : 
  (s - t) % 6 = 5 := by
  sorry

end remainder_of_difference_l2275_227524


namespace unique_solution_for_circ_equation_l2275_227540

-- Define the operation ∘
def circ (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

-- Theorem statement
theorem unique_solution_for_circ_equation :
  ∃! y : ℝ, circ 2 y = 10 := by sorry

end unique_solution_for_circ_equation_l2275_227540


namespace circle_area_in_square_l2275_227551

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + y^2 - 16*y + 65 = 0

-- Define the square
def square : Set (ℝ × ℝ) :=
  {p | 3 ≤ p.1 ∧ p.1 ≤ 8 ∧ 8 ≤ p.2 ∧ p.2 ≤ 13}

-- Theorem statement
theorem circle_area_in_square :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (∀ x y, circle_equation x y → (x, y) ∈ square) ∧
    (π * radius^2 = 24 * π) :=
sorry

end circle_area_in_square_l2275_227551


namespace part1_part2_l2275_227567

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part1 (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.C = 5 * Real.pi / 6) 
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2) : 
  t.c = Real.sqrt 13 := by
sorry

-- Part 2
theorem part2 (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.B = Real.pi / 3) : 
  ∃ (x y : ℝ), x = -Real.sqrt 3 ∧ y = 2 * Real.sqrt 3 ∧ 
  ∀ z, (2 * t.c - t.a = z) → (x < z ∧ z < y) := by
sorry

end part1_part2_l2275_227567


namespace ratio_sum_difference_l2275_227552

theorem ratio_sum_difference (a b c : ℝ) : 
  (a : ℝ) / 1 = (b : ℝ) / 3 ∧ (b : ℝ) / 3 = (c : ℝ) / 6 →
  a + b + c = 30 →
  c - b - a = 6 := by
sorry

end ratio_sum_difference_l2275_227552


namespace power_of_power_l2275_227504

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l2275_227504


namespace abs_sum_eq_sum_abs_iff_product_nonneg_l2275_227532

theorem abs_sum_eq_sum_abs_iff_product_nonneg (x y : ℝ) :
  abs (x + y) = abs x + abs y ↔ x * y ≥ 0 := by sorry

end abs_sum_eq_sum_abs_iff_product_nonneg_l2275_227532


namespace two_possible_products_l2275_227569

theorem two_possible_products (a b : ℝ) (ha : |a| = 5) (hb : |b| = 3) :
  ∃ (x y : ℝ), (∀ z, a * b = z → z = x ∨ z = y) ∧ x ≠ y :=
sorry

end two_possible_products_l2275_227569


namespace common_difference_is_negative_three_l2275_227501

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℕ → ℤ
  first_seventh_sum : a 1 + a 7 = -8
  second_term : a 2 = 2
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The common difference of an arithmetic progression -/
def common_difference (ap : ArithmeticProgression) : ℤ :=
  ap.a 2 - ap.a 1

theorem common_difference_is_negative_three (ap : ArithmeticProgression) :
  common_difference ap = -3 := by
  sorry

end common_difference_is_negative_three_l2275_227501


namespace boat_speed_in_still_water_l2275_227558

theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 5) 
  (h2 : downstream_distance = 6.25) 
  (h3 : downstream_time = 0.25) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 20 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end boat_speed_in_still_water_l2275_227558


namespace three_propositions_l2275_227549

theorem three_propositions :
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ a b : ℝ, |a + b| - 2*|a| ≤ |a - b|) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2/3) := by
  sorry

end three_propositions_l2275_227549


namespace sequence_bounds_l2275_227597

variable (n : ℕ)

def a : ℕ → ℚ
  | 0 => 1/2
  | k + 1 => a k + (1/n : ℚ) * (a k)^2

theorem sequence_bounds (hn : n > 0) : 1 - 1/n < a n n ∧ a n n < 1 := by
  sorry

end sequence_bounds_l2275_227597


namespace f_sum_lower_bound_f_squared_sum_lower_bound_l2275_227586

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1)

-- Theorem 1
theorem f_sum_lower_bound : ∀ x : ℝ, f x + f (1 - x) ≥ 1 := by sorry

-- Theorem 2
theorem f_squared_sum_lower_bound (a b : ℝ) (h : a + 2 * b = 8) : f a ^ 2 + f b ^ 2 ≥ 5 := by sorry

end f_sum_lower_bound_f_squared_sum_lower_bound_l2275_227586


namespace translation_right_proof_l2275_227510

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate_right (p : Point) (units : ℝ) : Point :=
  (p.1 + units, p.2)

-- Theorem statement
theorem translation_right_proof :
  let A : Point := (-4, 3)
  let A' : Point := translate_right A 2
  A' = (-2, 3) := by sorry

end translation_right_proof_l2275_227510


namespace ratio_problem_l2275_227555

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 5) :
  d / a = 4 / 15 := by
  sorry

end ratio_problem_l2275_227555


namespace complement_of_intersection_l2275_227537

open Set

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_of_intersection (U A B : Finset ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {2, 3, 4}) :
  (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end complement_of_intersection_l2275_227537


namespace distance_from_origin_l2275_227502

theorem distance_from_origin (x : ℝ) : |x| = 2 ↔ x = 2 ∨ x = -2 := by
  sorry

end distance_from_origin_l2275_227502


namespace exists_coverable_prism_l2275_227568

/-- A regular triangular prism with side edge length √3 times the base edge length -/
structure RegularTriangularPrism where
  base_edge : ℝ
  side_edge : ℝ
  side_edge_eq : side_edge = base_edge * Real.sqrt 3

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- A covering of a prism by equilateral triangles -/
structure PrismCovering where
  prism : RegularTriangularPrism
  triangles : Set EquilateralTriangle
  covers_prism : Bool
  no_overlaps : Bool

/-- Theorem stating the existence of a regular triangular prism that can be covered by equilateral triangles -/
theorem exists_coverable_prism : ∃ (p : RegularTriangularPrism) (c : PrismCovering), 
  c.prism = p ∧ c.covers_prism ∧ c.no_overlaps := by
  sorry

end exists_coverable_prism_l2275_227568


namespace chess_tournament_games_l2275_227519

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournamentGames (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 8 players, where each player plays twice with every other player, the total number of games played is 112 -/
theorem chess_tournament_games :
  tournamentGames 8 * 2 = 112 := by
  sorry

end chess_tournament_games_l2275_227519


namespace odd_products_fraction_l2275_227554

/-- The number of integers from 0 to 15 inclusive -/
def table_size : ℕ := 16

/-- The count of odd numbers from 0 to 15 inclusive -/
def odd_count : ℕ := 8

/-- The total number of entries in the multiplication table -/
def total_entries : ℕ := table_size * table_size

/-- The number of odd products in the multiplication table -/
def odd_products : ℕ := odd_count * odd_count

/-- The fraction of odd products in the multiplication table -/
def odd_fraction : ℚ := odd_products / total_entries

theorem odd_products_fraction :
  odd_fraction = 1 / 4 := by sorry

end odd_products_fraction_l2275_227554


namespace equation_solution_l2275_227535

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -4/3 ∧ x₂ = 3 ∧
  ∀ (x : ℝ), x ≠ 2/3 → x ≠ -4/3 →
  ((6*x + 4) / (3*x^2 + 6*x - 8) = (3*x) / (3*x - 2) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end equation_solution_l2275_227535


namespace volume_union_tetrahedrons_is_half_l2275_227573

/-- A regular tetrahedron formed from vertices of a unit cube -/
structure CubeTetrahedron where
  vertices : Finset (Fin 8)
  is_regular : Bool
  from_cube : Bool

/-- The volume of the union of two regular tetrahedrons formed from the vertices of a unit cube -/
def volume_union_tetrahedrons (t1 t2 : CubeTetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the union of two regular tetrahedrons
    formed from the vertices of a unit cube is 1/2 -/
theorem volume_union_tetrahedrons_is_half
  (t1 t2 : CubeTetrahedron)
  (h1 : t1.is_regular)
  (h2 : t2.is_regular)
  (h3 : t1.from_cube)
  (h4 : t2.from_cube)
  (h5 : t1.vertices ≠ t2.vertices)
  : volume_union_tetrahedrons t1 t2 = 1/2 :=
sorry

end volume_union_tetrahedrons_is_half_l2275_227573
