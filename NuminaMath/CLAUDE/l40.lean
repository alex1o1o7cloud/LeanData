import Mathlib

namespace quadratic_negative_on_unit_interval_l40_4028

/-- Given a quadratic function f(x) = ax² + bx + c with a > b > c and a + b + c = 0,
    prove that f(x) < 0 for all x in the open interval (0, 1). -/
theorem quadratic_negative_on_unit_interval
  (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x, x ∈ Set.Ioo 0 1 → a * x^2 + b * x + c < 0 :=
sorry

end quadratic_negative_on_unit_interval_l40_4028


namespace three_common_points_l40_4025

def equation1 (x y : ℝ) : Prop := (x - 2*y + 3) * (4*x + y - 5) = 0

def equation2 (x y : ℝ) : Prop := (x + 2*y - 3) * (3*x - 4*y + 6) = 0

def is_common_point (x y : ℝ) : Prop := equation1 x y ∧ equation2 x y

def distinct_points (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

theorem three_common_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_common_point p1.1 p1.2 ∧
    is_common_point p2.1 p2.2 ∧
    is_common_point p3.1 p3.2 ∧
    distinct_points p1 p2 ∧
    distinct_points p1 p3 ∧
    distinct_points p2 p3 ∧
    (∀ (p : ℝ × ℝ), is_common_point p.1 p.2 → p = p1 ∨ p = p2 ∨ p = p3) :=
by sorry

end three_common_points_l40_4025


namespace sum_zero_implies_opposites_l40_4037

theorem sum_zero_implies_opposites (a b : ℝ) : a + b = 0 → a = -b := by sorry

end sum_zero_implies_opposites_l40_4037


namespace ant_distance_l40_4090

def ant_path (n : ℕ) : ℝ × ℝ := 
  let rec path_sum (k : ℕ) : ℝ × ℝ := 
    if k = 0 then (0, 0)
    else 
      let (x, y) := path_sum (k-1)
      match k % 4 with
      | 0 => (x - k, y)
      | 1 => (x, y + k)
      | 2 => (x + k, y)
      | _ => (x, y - k)
  path_sum n

theorem ant_distance : 
  let (x, y) := ant_path 41
  Real.sqrt (x^2 + y^2) = Real.sqrt 221 := by sorry

end ant_distance_l40_4090


namespace max_divisor_of_prime_sum_l40_4034

theorem max_divisor_of_prime_sum (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a > 3 → b > 3 → c > 3 →
  2 * a + 5 * b = c →
  (∃ (n : ℕ), n ∣ (a + b + c) ∧ ∀ (m : ℕ), m ∣ (a + b + c) → m ≤ n) →
  (∃ (n : ℕ), n ∣ (a + b + c) ∧ ∀ (m : ℕ), m ∣ (a + b + c) → m ≤ 9) :=
by sorry

end max_divisor_of_prime_sum_l40_4034


namespace one_percent_as_decimal_l40_4055

theorem one_percent_as_decimal : (1 : ℚ) / 100 = (1 : ℚ) / 100 := by sorry

end one_percent_as_decimal_l40_4055


namespace excluded_students_average_mark_l40_4072

theorem excluded_students_average_mark
  (N : ℕ)  -- Total number of students
  (A : ℝ)  -- Average mark of all students
  (X : ℕ)  -- Number of excluded students
  (R : ℝ)  -- Average mark of remaining students
  (h1 : N = 10)
  (h2 : A = 80)
  (h3 : X = 5)
  (h4 : R = 90)
  : ∃ E : ℝ,  -- Average mark of excluded students
    N * A = X * E + (N - X) * R ∧ E = 70 :=
by sorry

end excluded_students_average_mark_l40_4072


namespace bills_rats_l40_4004

theorem bills_rats (total : ℕ) (ratio : ℕ) (h1 : total = 70) (h2 : ratio = 6) : 
  (ratio * total) / (ratio + 1) = 60 := by
  sorry

end bills_rats_l40_4004


namespace problem_solution_l40_4022

theorem problem_solution :
  ∀ (x a b c : ℤ),
    x ≠ 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
    ((a * x^4) / b * c)^3 = x^3 →
    a + b + c = 9 →
    ((x = 1 ∨ x = -1) ∧ a = 1 ∧ b = 4 ∧ c = 4) :=
by sorry

end problem_solution_l40_4022


namespace one_and_one_third_of_number_is_45_l40_4096

theorem one_and_one_third_of_number_is_45 :
  ∃ x : ℚ, (4 : ℚ) / 3 * x = 45 ∧ x = 33.75 := by
  sorry

end one_and_one_third_of_number_is_45_l40_4096


namespace tax_rate_above_40k_l40_4065

/-- Proves that the tax rate on income above $40,000 is 20% given the conditions --/
theorem tax_rate_above_40k (total_income : ℝ) (total_tax : ℝ) :
  total_income = 58000 →
  total_tax = 8000 →
  (∃ (rate_above_40k : ℝ),
    total_tax = 0.11 * 40000 + rate_above_40k * (total_income - 40000) ∧
    rate_above_40k = 0.20) :=
by
  sorry

end tax_rate_above_40k_l40_4065


namespace f_min_value_l40_4030

/-- The quadratic function f(x) = 2x^2 - 8x + 9 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 9

/-- The minimum value of f(x) is 1 -/
theorem f_min_value : ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), f x ≥ m := by
  sorry

end f_min_value_l40_4030


namespace division_multiplication_problem_l40_4073

theorem division_multiplication_problem : 
  let x : ℝ := 7.5
  let y : ℝ := 6
  let z : ℝ := 12
  (x / y) * z = 15 := by sorry

end division_multiplication_problem_l40_4073


namespace max_volume_of_prism_l40_4009

/-- A right prism with a rectangular base -/
structure RectPrism where
  height : ℝ
  base_length : ℝ
  base_width : ℝ

/-- The surface area constraint for the prism -/
def surface_area_constraint (p : RectPrism) : Prop :=
  p.height * p.base_length + p.height * p.base_width + p.base_length * p.base_width = 36

/-- The constraint that base sides are twice the height -/
def base_height_constraint (p : RectPrism) : Prop :=
  p.base_length = 2 * p.height ∧ p.base_width = 2 * p.height

/-- The volume of the prism -/
def volume (p : RectPrism) : ℝ :=
  p.height * p.base_length * p.base_width

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_of_prism :
  ∃ (p : RectPrism), surface_area_constraint p ∧ base_height_constraint p ∧
    (∀ (q : RectPrism), surface_area_constraint q → base_height_constraint q →
      volume q ≤ volume p) ∧
    volume p = 27 * Real.sqrt 2 :=
  sorry

end max_volume_of_prism_l40_4009


namespace meeting_point_2015_l40_4041

/-- Represents a point on a line segment --/
structure Point where
  position : ℝ
  deriving Inhabited

/-- Represents an object moving on a line segment --/
structure MovingObject where
  startPoint : Point
  speed : ℝ
  startTime : ℝ
  deriving Inhabited

/-- Calculates the meeting point of two objects --/
def meetingPoint (obj1 obj2 : MovingObject) : Point :=
  sorry

/-- Theorem: The 2015th meeting point is the same as the 1st meeting point --/
theorem meeting_point_2015 (A B : Point) (obj1 obj2 : MovingObject) :
  obj1.startPoint = A ∧ obj2.startPoint = B →
  obj1.speed > 0 ∧ obj2.speed > 0 →
  meetingPoint obj1 obj2 = meetingPoint obj1 obj2 :=
by sorry

end meeting_point_2015_l40_4041


namespace mrs_jane_total_coins_l40_4029

def total_coins (jayden_coins jason_coins : ℕ) : ℕ :=
  jayden_coins + jason_coins

theorem mrs_jane_total_coins : 
  let jayden_coins : ℕ := 300
  let jason_coins : ℕ := jayden_coins + 60
  total_coins jayden_coins jason_coins = 660 := by
  sorry

end mrs_jane_total_coins_l40_4029


namespace union_of_A_and_complement_of_B_l40_4070

def U : Set Int := {x | -3 < x ∧ x < 3}
def A : Set Int := {1, 2}
def B : Set Int := {-2, -1, 2}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {0, 1, 2} := by sorry

end union_of_A_and_complement_of_B_l40_4070


namespace parabola_kite_theorem_l40_4056

/-- Represents a parabola of the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := sorry

/-- The sum of the coefficients of the x^2 terms in the two parabolas forming the kite -/
def coeff_sum (k : Kite) : ℝ := k.p1.a + k.p2.a

theorem parabola_kite_theorem (k : Kite) :
  k.p1 = Parabola.mk a (-4) ∧
  k.p2 = Parabola.mk (-b) 8 ∧
  kite_area k = 24 →
  coeff_sum k = 3 := by sorry

end parabola_kite_theorem_l40_4056


namespace eliza_cookies_l40_4017

theorem eliza_cookies (x : ℚ) 
  (h1 : x + 3*x + 4*(3*x) + 6*(4*(3*x)) = 234) : x = 117/44 := by
  sorry

end eliza_cookies_l40_4017


namespace power_function_passes_through_one_l40_4059

theorem power_function_passes_through_one (α : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x ^ α
  f 1 = 1 := by sorry

end power_function_passes_through_one_l40_4059


namespace average_age_first_fifth_dog_l40_4002

/-- The age of the nth fastest dog -/
def dog_age (n : ℕ) : ℕ :=
  match n with
  | 1 => 10
  | 2 => dog_age 1 - 2
  | 3 => dog_age 2 + 4
  | 4 => dog_age 3 / 2
  | 5 => dog_age 4 + 20
  | _ => 0

theorem average_age_first_fifth_dog :
  (dog_age 1 + dog_age 5) / 2 = 18 := by
  sorry

end average_age_first_fifth_dog_l40_4002


namespace least_positive_integer_divisible_by_four_primes_l40_4099

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ (∃ (p₁ p₂ p₃ p₄ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m > 0 → (∃ (q₁ q₂ q₃ q₄ : ℕ),
    Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
    m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0) → m ≥ n) ∧
  n = 210 :=
sorry

end least_positive_integer_divisible_by_four_primes_l40_4099


namespace problem_solution_l40_4019

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 30) :
  z + 1 / y = 38 / 179 := by
sorry

end problem_solution_l40_4019


namespace inequality_proof_l40_4069

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a + 1) * (b^2 + b + 1) * (c^2 + c + 1) / (a * b * c) ≥ 27 := by
  sorry

end inequality_proof_l40_4069


namespace quadratic_equation_roots_l40_4039

theorem quadratic_equation_roots : ∃ x y : ℝ, x ≠ y ∧ 
  x^2 - 6*x + 1 = 0 ∧ y^2 - 6*y + 1 = 0 := by
  sorry

end quadratic_equation_roots_l40_4039


namespace max_result_ahn_max_result_ahn_achievable_l40_4054

theorem max_result_ahn (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999) → 3 * (200 + n) ≤ 3597 := by
  sorry

theorem max_result_ahn_achievable : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 * (200 + n) = 3597 := by
  sorry

end max_result_ahn_max_result_ahn_achievable_l40_4054


namespace race_outcomes_count_l40_4038

/-- The number of participants in the race -/
def num_participants : Nat := 5

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_podium_positions : Nat := 3

/-- Calculate the number of permutations of k items chosen from n items -/
def permutations (n k : Nat) : Nat :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating that the number of different 1st-2nd-3rd place outcomes
    in a race with 5 participants and no ties is equal to 60 -/
theorem race_outcomes_count : 
  permutations num_participants num_podium_positions = 60 := by
  sorry


end race_outcomes_count_l40_4038


namespace units_digit_of_fraction_l40_4053

theorem units_digit_of_fraction : (30 * 31 * 32 * 33 * 34 * 35) / 7200 ≡ 6 [ZMOD 10] := by
  sorry

end units_digit_of_fraction_l40_4053


namespace rectangle_y_value_l40_4013

/-- Given a rectangle with vertices (-3, y), (1, y), (1, -2), and (-3, -2),
    if the area of the rectangle is 12, then y = 1. -/
theorem rectangle_y_value (y : ℝ) : 
  let vertices := [(-3, y), (1, y), (1, -2), (-3, -2)]
  let length := 1 - (-3)
  let height := y - (-2)
  let area := length * height
  area = 12 → y = 1 := by
sorry

end rectangle_y_value_l40_4013


namespace contrapositive_real_roots_negation_and_disjunction_correct_propositions_l40_4044

-- Proposition ②
theorem contrapositive_real_roots (q : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + q ≠ 0) → q > 1 :=
sorry

-- Proposition ③
theorem negation_and_disjunction (p q : Prop) :
  ¬p ∧ (p ∨ q) → q :=
sorry

-- Main theorem combining both propositions
theorem correct_propositions :
  (∃ q : ℝ, (∀ x : ℝ, x^2 + 2*x + q ≠ 0) → q > 1) ∧
  (∀ p q : Prop, ¬p ∧ (p ∨ q) → q) :=
sorry

end contrapositive_real_roots_negation_and_disjunction_correct_propositions_l40_4044


namespace polynomial_coefficient_sum_exists_l40_4000

theorem polynomial_coefficient_sum_exists : ∃ (a b c d : ℤ),
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 3*x^2 + 12*x - 8) ∧
  (∃ (s : ℤ), s = a + b + c + d) :=
by sorry

end polynomial_coefficient_sum_exists_l40_4000


namespace star_calculation_l40_4082

-- Define the star operation
def star (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem star_calculation :
  star (star (star 3 5) 2) 7 = -11/10 :=
by sorry

end star_calculation_l40_4082


namespace election_majority_l40_4071

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 5200 → 
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1040 :=
by sorry

end election_majority_l40_4071


namespace cat_distribution_theorem_l40_4097

/-- Represents the number of segments white cats are divided into by black cats -/
inductive X
| one
| two
| three
| four

/-- The probability distribution of X -/
def P (x : X) : ℚ :=
  match x with
  | X.one => 1 / 30
  | X.two => 9 / 30
  | X.three => 15 / 30
  | X.four => 5 / 30

theorem cat_distribution_theorem :
  (∀ x : X, 0 ≤ P x ∧ P x ≤ 1) ∧
  (P X.one + P X.two + P X.three + P X.four = 1) :=
sorry

end cat_distribution_theorem_l40_4097


namespace trailer_homes_calculation_l40_4021

/-- The number of new trailer homes added to Maple Drive -/
def new_homes : ℕ := 17

/-- The initial number of trailer homes on Maple Drive -/
def initial_homes : ℕ := 25

/-- The number of years that have passed -/
def years_passed : ℕ := 3

/-- The initial average age of the trailer homes -/
def initial_avg_age : ℚ := 15

/-- The current average age of all trailer homes -/
def current_avg_age : ℚ := 12

theorem trailer_homes_calculation :
  (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
  (initial_homes + new_homes) = current_avg_age :=
sorry

end trailer_homes_calculation_l40_4021


namespace y_axis_intersection_uniqueness_l40_4015

theorem y_axis_intersection_uniqueness (f : ℝ → ℝ) : 
  ∃! y, f 0 = y :=
sorry

end y_axis_intersection_uniqueness_l40_4015


namespace birdhouse_volume_difference_l40_4049

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Volume of a rectangular prism -/
def volume (width height depth : ℚ) : ℚ := width * height * depth

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℚ := 1
def sara_height : ℚ := 2
def sara_depth : ℚ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_width : ℚ := 16
def jake_height : ℚ := 20
def jake_depth : ℚ := 18

/-- Theorem: The difference in volume between Sara's and Jake's birdhouses is 1152 cubic inches -/
theorem birdhouse_volume_difference :
  volume (sara_width * feet_to_inches) (sara_height * feet_to_inches) (sara_depth * feet_to_inches) -
  volume jake_width jake_height jake_depth = 1152 := by
  sorry

end birdhouse_volume_difference_l40_4049


namespace system_solution_proof_l40_4020

theorem system_solution_proof :
  ∃ (x y : ℝ), 
    (4 * x + y = 12) ∧ 
    (3 * x - 2 * y = -2) ∧ 
    (x = 2) ∧ 
    (y = 4) := by
  sorry

end system_solution_proof_l40_4020


namespace fraction_simplification_l40_4078

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4 / 6) 
  (hy : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 := by
  sorry

end fraction_simplification_l40_4078


namespace max_consecutive_matching_terms_l40_4060

/-- Given two sequences with periods 7 and 13, prove that the maximum number of
consecutive matching terms is the LCM of their periods. -/
theorem max_consecutive_matching_terms
  (a b : ℕ → ℕ)  -- Two sequences of natural numbers
  (ha : ∀ n, a (n + 7) = a n)  -- a has period 7
  (hb : ∀ n, b (n + 13) = b n)  -- b has period 13
  : (∃ k, ∀ i ≤ k, a i = b i) ↔ (∃ k, k = Nat.lcm 7 13 ∧ ∀ i ≤ k, a i = b i) :=
by sorry

end max_consecutive_matching_terms_l40_4060


namespace walkway_diameter_l40_4079

theorem walkway_diameter (water_diameter : Real) (tile_width : Real) (walkway_width : Real) :
  water_diameter = 16 →
  tile_width = 12 →
  walkway_width = 10 →
  2 * (water_diameter / 2 + tile_width + walkway_width) = 60 := by
  sorry

end walkway_diameter_l40_4079


namespace area_difference_l40_4024

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem area_difference (sheet1_length sheet1_width sheet2_length sheet2_width : ℝ) 
  (h1 : sheet1_length = 11) 
  (h2 : sheet1_width = 13) 
  (h3 : sheet2_length = 6.5) 
  (h4 : sheet2_width = 11) : 
  2 * (sheet1_length * sheet1_width) - 2 * (sheet2_length * sheet2_width) = 143 := by
  sorry

end area_difference_l40_4024


namespace limit_p_n_sqrt_n_l40_4058

/-- The probability that the sum of two randomly selected integers from {1,2,...,n} is a perfect square -/
def p (n : ℕ) : ℝ := sorry

/-- The main theorem stating that the limit of p_n√n as n approaches infinity is 2/3 -/
theorem limit_p_n_sqrt_n :
  ∃ (L : ℝ), L = 2/3 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |p n * Real.sqrt n - L| < ε :=
sorry

end limit_p_n_sqrt_n_l40_4058


namespace square_sum_calculation_l40_4084

theorem square_sum_calculation (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 10)
  (h2 : a + b + c = 31) : 
  a^2 + b^2 + c^2 = 941 := by
sorry

end square_sum_calculation_l40_4084


namespace seed_without_water_impossible_l40_4062

/-- An event is a phenomenon that may or may not occur under certain conditions. -/
structure Event where
  description : String

/-- An impossible event is one that cannot occur under certain conditions. -/
def Event.impossible (e : Event) : Prop := sorry

/-- A certain event is one that will definitely occur under certain conditions. -/
def Event.certain (e : Event) : Prop := sorry

/-- A random event is one that may or may not occur under certain conditions. -/
def Event.random (e : Event) : Prop := sorry

def conductor_heating : Event :=
  { description := "A conductor heats up when conducting electricity" }

def three_points_plane : Event :=
  { description := "Three non-collinear points determine a plane" }

def seed_without_water : Event :=
  { description := "A seed germinates without water" }

def consecutive_lottery : Event :=
  { description := "Someone wins the lottery for two consecutive weeks" }

theorem seed_without_water_impossible :
  Event.impossible seed_without_water ∧
  ¬Event.impossible conductor_heating ∧
  ¬Event.impossible three_points_plane ∧
  ¬Event.impossible consecutive_lottery :=
by sorry

end seed_without_water_impossible_l40_4062


namespace line_through_circle_center_parallel_to_line_l40_4068

/-- The equation of a line passing through the center of a circle and parallel to another line -/
theorem line_through_circle_center_parallel_to_line :
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}
  let center : ℝ × ℝ := (2, 0)
  let parallel_line : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + 1 = 0}
  let result_line : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 - 4 = 0}
  (center ∈ circle) →
  (∀ p ∈ result_line, ∃ q ∈ parallel_line, (p.2 - q.2) / (p.1 - q.1) = (center.2 - q.2) / (center.1 - q.1)) →
  (center ∈ result_line) :=
by
  sorry

end line_through_circle_center_parallel_to_line_l40_4068


namespace circle_B_radius_l40_4031

/-- The configuration of four circles A, B, C, and D with specific properties -/
structure CircleConfig where
  /-- Radius of circle A -/
  radius_A : ℝ
  /-- Radius of circle B -/
  radius_B : ℝ
  /-- Radius of circle C -/
  radius_C : ℝ
  /-- Radius of circle D -/
  radius_D : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externally_tangent : radius_A + radius_B + radius_C = radius_D
  /-- Circles B and C are congruent -/
  B_C_congruent : radius_B = radius_C
  /-- Circle A passes through the center of D -/
  A_through_D_center : radius_A = radius_D / 2
  /-- Circle A has a radius of 2 -/
  A_radius_2 : radius_A = 2

/-- The main theorem stating that given the circle configuration, the radius of circle B is approximately 0.923 -/
theorem circle_B_radius (config : CircleConfig) : 
  0.922 < config.radius_B ∧ config.radius_B < 0.924 := by
  sorry


end circle_B_radius_l40_4031


namespace solution_of_functional_equation_l40_4057

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, x + f x = f (f x)

/-- The theorem stating that the only solution to f(f(x)) = 0 is x = 0 -/
theorem solution_of_functional_equation (f : ℝ → ℝ) (h : FunctionalEquation f) :
  {x : ℝ | f (f x) = 0} = {0} := by
  sorry

end solution_of_functional_equation_l40_4057


namespace math_club_team_probability_l40_4018

theorem math_club_team_probability :
  let total_girls : ℕ := 8
  let total_boys : ℕ := 6
  let team_size : ℕ := 4
  let girls_in_team : ℕ := 2
  let boys_in_team : ℕ := 2

  (Nat.choose total_girls girls_in_team * Nat.choose total_boys boys_in_team) /
  Nat.choose (total_girls + total_boys) team_size = 60 / 143 :=
by sorry

end math_club_team_probability_l40_4018


namespace bus_delay_l40_4075

/-- Proves that walking at 4/5 of usual speed results in a 5-minute delay -/
theorem bus_delay (usual_time : ℝ) (h : usual_time = 20) : 
  usual_time * (5/4) - usual_time = 5 := by
  sorry

end bus_delay_l40_4075


namespace divisibility_of_subset_products_l40_4077

def P (A : Finset Nat) : Nat := A.prod id

theorem divisibility_of_subset_products :
  let S : Finset Nat := Finset.range 2010
  let n : Nat := Nat.choose 2010 99
  let subsets : Finset (Finset Nat) := S.powerset.filter (fun A => A.card = 99)
  2010 ∣ subsets.sum P := by sorry

end divisibility_of_subset_products_l40_4077


namespace shaded_probability_l40_4026

/-- Represents a triangle in the diagram -/
structure Triangle where
  shaded : Bool

/-- Represents the diagram with triangles -/
structure Diagram where
  triangles : List Triangle
  shaded_count : Nat
  total_count : Nat

/-- The probability of selecting a shaded triangle -/
def probability_shaded (d : Diagram) : ℚ :=
  d.shaded_count / d.total_count

theorem shaded_probability (d : Diagram) 
  (h1 : d.total_count > 4)
  (h2 : d.shaded_count = d.total_count / 2)
  (h3 : d.shaded_count = (d.triangles.filter Triangle.shaded).length)
  (h4 : d.total_count = d.triangles.length) :
  probability_shaded d = 1 / 2 := by
  sorry

#check shaded_probability

end shaded_probability_l40_4026


namespace complex_fraction_equality_l40_4035

theorem complex_fraction_equality : 
  let a := 3 + 1/3 + 2.5
  let b := 2.5 - (1 + 1/3)
  let c := 4.6 - (2 + 1/3)
  let d := 4.6 + (2 + 1/3)
  let e := 5.2
  let f := 0.05 / (1/7 - 0.125) + 5.7
  (a / b * c / d * e) / f = 5/34 := by sorry

end complex_fraction_equality_l40_4035


namespace hydrochloric_acid_solution_l40_4023

/-- Represents the volume of pure hydrochloric acid needed to be added -/
def x : ℝ := sorry

/-- The initial volume of the solution in liters -/
def initial_volume : ℝ := 60

/-- The initial concentration of hydrochloric acid as a decimal -/
def initial_concentration : ℝ := 0.10

/-- The target concentration of hydrochloric acid as a decimal -/
def target_concentration : ℝ := 0.15

theorem hydrochloric_acid_solution :
  initial_concentration * initial_volume + x = target_concentration * (initial_volume + x) := by
  sorry

end hydrochloric_acid_solution_l40_4023


namespace vector_loop_closure_l40_4067

variable {V : Type*} [AddCommGroup V]

theorem vector_loop_closure (A B C : V) :
  (B - A) - (B - C) + (A - C) = (0 : V) := by
  sorry

end vector_loop_closure_l40_4067


namespace hyperbola_focal_length_l40_4032

/-- The focal length of a hyperbola with equation x²/m - y² = 1 (m > 0) 
    and asymptote √3x + my = 0 is 4 -/
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / m - y^2 = 1
  let asymptote : ℝ → ℝ → Prop := λ x y => Real.sqrt 3 * x + m * y = 0
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    (∀ x y, C x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∀ x y, asymptote x y ↔ y = -(Real.sqrt 3 / m) * x) ∧
    c^2 = a^2 + b^2 ∧
    2 * c = 4 :=
by sorry

end hyperbola_focal_length_l40_4032


namespace trailing_zeroes_of_sum_factorials_l40_4050

/-- The number of trailing zeroes in a natural number -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The main theorem: the number of trailing zeroes in 70! + 140! is 16 -/
theorem trailing_zeroes_of_sum_factorials :
  trailingZeroes (factorial 70 + factorial 140) = 16 := by sorry

end trailing_zeroes_of_sum_factorials_l40_4050


namespace slope_intercept_sum_horizontal_line_l40_4036

/-- Given two points with the same y-coordinate and different x-coordinates,
    the sum of the slope and y-intercept of the line containing both points is 20. -/
theorem slope_intercept_sum_horizontal_line (C D : ℝ × ℝ) :
  C.2 = 20 →
  D.2 = 20 →
  C.1 ≠ D.1 →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 20 := by
  sorry

end slope_intercept_sum_horizontal_line_l40_4036


namespace exercise_band_resistance_l40_4010

/-- The resistance added by each exercise band -/
def band_resistance : ℝ := sorry

/-- The number of exercise bands -/
def num_bands : ℕ := 2

/-- The weight of the dumbbell in pounds -/
def dumbbell_weight : ℝ := 10

/-- The total squat weight with both sets of bands doubled and the dumbbell -/
def total_squat_weight : ℝ := 30

/-- Theorem stating that each band adds 10 pounds of resistance -/
theorem exercise_band_resistance :
  band_resistance = 10 :=
by sorry

end exercise_band_resistance_l40_4010


namespace arithmetic_sequence_12th_term_l40_4048

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_12th_term 
  (seq : ArithmeticSequence) 
  (sum7 : sum_n seq 7 = 7)
  (term79 : seq.a 7 + seq.a 9 = 16) : 
  seq.a 12 = 15 := by
sorry

end arithmetic_sequence_12th_term_l40_4048


namespace min_fence_posts_is_22_l40_4005

/-- Calculates the number of fence posts needed for a rectangular grazing area -/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side_posts := (length / post_spacing) + 1
  let short_side_posts := (width / post_spacing) + 1
  (2 * long_side_posts) + short_side_posts - 2

/-- The minimum number of fence posts for the given dimensions is 22 -/
theorem min_fence_posts_is_22 :
  fence_posts 80 50 10 = 22 :=
by sorry

end min_fence_posts_is_22_l40_4005


namespace max_sum_given_constraints_l40_4074

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end max_sum_given_constraints_l40_4074


namespace no_positive_integer_solution_l40_4089

theorem no_positive_integer_solution :
  ¬ ∃ (a b c d : ℕ+), a + b + c + d - 3 = a * b + c * d := by
  sorry

end no_positive_integer_solution_l40_4089


namespace product_first_10000_trailing_zeros_l40_4093

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The product of the first 10000 natural numbers has 2499 trailing zeros -/
theorem product_first_10000_trailing_zeros :
  trailingZeros 10000 = 2499 := by
  sorry

end product_first_10000_trailing_zeros_l40_4093


namespace time_to_reach_destination_l40_4014

/-- Calculates the time needed to reach a destination given initial movement and remaining distance -/
theorem time_to_reach_destination (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance_yards : ℝ) : 
  initial_distance > 0 ∧ initial_time > 0 ∧ remaining_distance_yards > 0 →
  (remaining_distance_yards * 3) / (initial_distance / initial_time) = 75 :=
by
  sorry

#check time_to_reach_destination 80 20 100

end time_to_reach_destination_l40_4014


namespace arithmetic_sequence_property_l40_4033

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 2 + 4 * a 7 + a 12 = 96

/-- Theorem stating the relationship in the arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : 
  2 * seq.a 3 + seq.a 15 = 48 := by
  sorry

end arithmetic_sequence_property_l40_4033


namespace trapezoid_area_l40_4098

/-- Represents a trapezoid ABCD with point E at the intersection of diagonals -/
structure Trapezoid :=
  (A B C D E : ℝ × ℝ)

/-- The area of a triangle given its vertices -/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_area (ABCD : Trapezoid) : 
  (ABCD.A.1 = ABCD.B.1) ∧  -- AB is parallel to CD (same x-coordinate)
  (ABCD.C.1 = ABCD.D.1) ∧
  (triangle_area ABCD.A ABCD.B ABCD.E = 60) ∧  -- Area of ABE is 60
  (triangle_area ABCD.A ABCD.D ABCD.E = 30) →  -- Area of ADE is 30
  (triangle_area ABCD.A ABCD.B ABCD.C) + 
  (triangle_area ABCD.A ABCD.C ABCD.D) = 135 := by
  sorry

end trapezoid_area_l40_4098


namespace pie_apples_ratio_l40_4003

def total_apples : ℕ := 62
def refrigerator_apples : ℕ := 25
def muffin_apples : ℕ := 6

def pie_apples : ℕ := total_apples - refrigerator_apples - muffin_apples

theorem pie_apples_ratio :
  (pie_apples : ℚ) / total_apples = 1 / 2 := by sorry

end pie_apples_ratio_l40_4003


namespace store_opening_cost_l40_4051

/-- The cost to open Kim's store -/
def openingCost (monthlyRevenue : ℕ) (monthlyExpenses : ℕ) (monthsToPayback : ℕ) : ℕ :=
  (monthlyRevenue - monthlyExpenses) * monthsToPayback

/-- Theorem stating the cost to open Kim's store -/
theorem store_opening_cost : openingCost 4000 1500 10 = 25000 := by
  sorry

end store_opening_cost_l40_4051


namespace engine_problem_solution_l40_4087

/-- Represents the fuel consumption and operation time of two engines -/
structure EnginePair where
  first_consumption : ℝ
  second_consumption : ℝ
  time_difference : ℝ
  consumption_difference : ℝ

/-- Determines if the given fuel consumption rates satisfy the conditions for the two engines -/
def is_valid_solution (pair : EnginePair) (first_rate second_rate : ℝ) : Prop :=
  first_rate > 0 ∧
  second_rate > 0 ∧
  first_rate = second_rate + pair.consumption_difference ∧
  pair.first_consumption / first_rate - pair.second_consumption / second_rate = pair.time_difference

/-- Theorem stating that the given solution satisfies the engine problem conditions -/
theorem engine_problem_solution (pair : EnginePair) 
    (h1 : pair.first_consumption = 300)
    (h2 : pair.second_consumption = 192)
    (h3 : pair.time_difference = 2)
    (h4 : pair.consumption_difference = 6) :
    is_valid_solution pair 30 24 := by
  sorry


end engine_problem_solution_l40_4087


namespace tank_capacity_l40_4007

theorem tank_capacity (x : ℝ) (h : 0.5 * x = 75) : x = 150 := by
  sorry

end tank_capacity_l40_4007


namespace intersection_line_equation_l40_4042

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x + 4)^2 + (y + 3)^2 = 8

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 3*y + 13 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line x y :=
by sorry

end intersection_line_equation_l40_4042


namespace root_sum_quotient_l40_4061

theorem root_sum_quotient (p q r s t : ℝ) (hp : p ≠ 0) 
  (h1 : p * 6^4 + q * 6^3 + r * 6^2 + s * 6 + t = 0)
  (h2 : p * (-4)^4 + q * (-4)^3 + r * (-4)^2 + s * (-4) + t = 0)
  (h3 : t = 0) :
  (q + s) / p = 48 := by
  sorry

end root_sum_quotient_l40_4061


namespace math_city_intersections_l40_4088

/-- Represents a city with streets and intersections -/
structure City where
  num_streets : ℕ
  num_non_intersections : ℕ

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  Nat.choose c.num_streets 2 - c.num_non_intersections

/-- Theorem: A city with 10 streets and 2 non-intersections has 43 intersections -/
theorem math_city_intersections :
  let c : City := { num_streets := 10, num_non_intersections := 2 }
  num_intersections c = 43 := by sorry

end math_city_intersections_l40_4088


namespace candy_box_price_increase_l40_4012

theorem candy_box_price_increase (new_price : ℝ) (increase_rate : ℝ) (original_price : ℝ) :
  new_price = 20 ∧ increase_rate = 0.25 ∧ new_price = original_price * (1 + increase_rate) →
  original_price = 16 := by
  sorry

end candy_box_price_increase_l40_4012


namespace fraction_1991_1949_position_l40_4001

/-- Represents a fraction in the table -/
structure Fraction where
  numerator : ℕ
  denominator : ℕ

/-- Represents a row in the table -/
def Row := List Fraction

/-- Generates a row of the table given its index -/
def generateRow (n : ℕ) : Row :=
  sorry

/-- Checks if a fraction appears in a given row -/
def appearsInRow (f : Fraction) (r : Row) : Prop :=
  sorry

/-- The row number where 1991/1949 appears -/
def targetRow : ℕ := 3939

/-- The position of 1991/1949 in its row -/
def targetPosition : ℕ := 1949

theorem fraction_1991_1949_position : 
  let f := Fraction.mk 1991 1949
  let r := generateRow targetRow
  appearsInRow f r ∧ 
  (∃ (l1 l2 : List Fraction), r = l1 ++ [f] ++ l2 ∧ l1.length = targetPosition - 1) :=
sorry

end fraction_1991_1949_position_l40_4001


namespace product_of_squares_l40_4081

theorem product_of_squares (x : ℝ) : 
  (Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8) → 
  ((6 + x) * (21 - x) = 1369 / 4) := by
sorry

end product_of_squares_l40_4081


namespace cards_in_play_l40_4076

/-- The number of cards in a standard deck --/
def standard_deck : ℕ := 52

/-- The number of cards kept away --/
def cards_kept_away : ℕ := 2

/-- Theorem: The number of cards being played with is 50 --/
theorem cards_in_play (deck : ℕ) (kept_away : ℕ) 
  (h1 : deck = standard_deck) (h2 : kept_away = cards_kept_away) : 
  deck - kept_away = 50 := by
  sorry

end cards_in_play_l40_4076


namespace first_month_sales_l40_4008

def sales_month_2 : ℕ := 5744
def sales_month_3 : ℕ := 5864
def sales_month_4 : ℕ := 6122
def sales_month_5 : ℕ := 6588
def sales_month_6 : ℕ := 4916
def average_sale : ℕ := 5750

theorem first_month_sales :
  sales_month_2 + sales_month_3 + sales_month_4 + sales_month_5 + sales_month_6 + 5266 = 6 * average_sale :=
by sorry

end first_month_sales_l40_4008


namespace smallest_k_for_zero_l40_4016

def f (a b M : ℕ) (n : ℤ) : ℤ :=
  if n ≤ M then n + a else n - b

def iterate_f (a b M : ℕ) : ℕ → ℤ → ℤ
  | 0, n => n
  | k+1, n => f a b M (iterate_f a b M k n)

theorem smallest_k_for_zero (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let M := (a + b) / 2
  ∃ k : ℕ, (∀ j < k, iterate_f a b M j 0 ≠ 0) ∧ 
            iterate_f a b M k 0 = 0 ∧
            k = (a + b) / Nat.gcd a b :=
  sorry

end smallest_k_for_zero_l40_4016


namespace minimum_pigs_on_farm_l40_4080

theorem minimum_pigs_on_farm (total : ℕ) (pigs : ℕ) : 
  (pigs : ℝ) / total ≥ 0.54 ∧ (pigs : ℝ) / total ≤ 0.57 → pigs ≥ 5 :=
by sorry

end minimum_pigs_on_farm_l40_4080


namespace hidden_faces_sum_l40_4006

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 4, 5, 6]

/-- The sum of visible numbers -/
def visible_sum : ℕ := visible_numbers.sum

theorem hidden_faces_sum :
  (num_dice * die_sum) - visible_sum = 58 := by
  sorry

end hidden_faces_sum_l40_4006


namespace dot_product_range_l40_4094

/-- Given points A and B in a 2D Cartesian coordinate system,
    and P on the curve y = √(1-x²), prove that the dot product
    BP · BA is bounded by 0 and 1+√2. -/
theorem dot_product_range (x y : ℝ) :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, -1)
  let P : ℝ × ℝ := (x, y)
  y = Real.sqrt (1 - x^2) →
  0 ≤ ((P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2)) ∧
  ((P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2)) ≤ 1 + Real.sqrt 2 :=
by sorry

end dot_product_range_l40_4094


namespace solve_chicken_problem_l40_4040

/-- Represents the problem of calculating the number of chickens sold -/
def chicken_problem (selling_price feed_cost feed_weight feed_per_chicken total_profit : ℚ) : Prop :=
  let cost_per_chicken := (feed_per_chicken / feed_weight) * feed_cost
  let profit_per_chicken := selling_price - cost_per_chicken
  let num_chickens := total_profit / profit_per_chicken
  num_chickens = 50

/-- Theorem stating the solution to the chicken problem -/
theorem solve_chicken_problem :
  chicken_problem 1.5 2 20 2 65 := by
  sorry

#check solve_chicken_problem

end solve_chicken_problem_l40_4040


namespace resistor_value_l40_4046

/-- Given two identical resistors connected in series to a DC voltage source,
    if the voltage across one resistor is 2 V and the current through the circuit is 4 A,
    then the resistance of each resistor is 0.5 Ω. -/
theorem resistor_value (R₀ : ℝ) (U V I : ℝ) : 
  U = 2 → -- Voltage across one resistor
  V = 2 * U → -- Total voltage
  I = 4 → -- Current through the circuit
  V = I * (2 * R₀) → -- Ohm's law
  R₀ = 0.5 := by
  sorry

#check resistor_value

end resistor_value_l40_4046


namespace complex_number_in_fourth_quadrant_l40_4063

theorem complex_number_in_fourth_quadrant (a b : ℝ) : 
  let z : ℂ := (a^2 - 6*a + 10) + (-b^2 + 4*b - 5)*I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l40_4063


namespace sum_min_period_length_l40_4064

def min_period_length (x : ℚ) : ℕ :=
  sorry

theorem sum_min_period_length (A B : ℚ) :
  min_period_length A = 6 →
  min_period_length B = 12 →
  min_period_length (A + B) = 12 ∨ min_period_length (A + B) = 4 :=
by sorry

end sum_min_period_length_l40_4064


namespace s_128_eq_one_half_l40_4083

/-- Best decomposition of a positive integer -/
def BestDecomposition (n : ℕ+) : ℕ+ × ℕ+ :=
  sorry

/-- S function for a positive integer -/
def S (n : ℕ+) : ℚ :=
  let (p, q) := BestDecomposition n
  p.val / q.val

/-- Theorem: S(128) = 1/2 -/
theorem s_128_eq_one_half : S 128 = 1/2 := by
  sorry

end s_128_eq_one_half_l40_4083


namespace car_speed_problem_l40_4047

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) 
  (h1 : speed_second_hour = 55)
  (h2 : average_speed = 72.5) : 
  ∃ speed_first_hour : ℝ, 
    speed_first_hour = 90 ∧ 
    (speed_first_hour + speed_second_hour) / 2 = average_speed :=
by
  sorry

end car_speed_problem_l40_4047


namespace determinant_transformation_l40_4095

theorem determinant_transformation (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = 10 →
  Matrix.det !![a + 2*c, b + 3*d; c, d] = 10 - c*d :=
by sorry

end determinant_transformation_l40_4095


namespace sons_age_l40_4011

/-- Given a father and son, where the father is 46 years older than the son,
    and in two years the father's age will be twice the son's age,
    prove that the son's current age is 44 years. -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 46 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 44 := by sorry

end sons_age_l40_4011


namespace square_perimeter_ratio_l40_4066

theorem square_perimeter_ratio (d D s S : ℝ) : 
  d > 0 → s > 0 → 
  d = s * Real.sqrt 2 → 
  D = S * Real.sqrt 2 → 
  D = 11 * d → 
  (4 * S) / (4 * s) = 11 := by
sorry

end square_perimeter_ratio_l40_4066


namespace juliet_younger_than_ralph_l40_4091

/-- Represents the ages of three siblings -/
structure SiblingAges where
  juliet : ℕ
  maggie : ℕ
  ralph : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : SiblingAges) : Prop :=
  ages.juliet = ages.maggie + 3 ∧
  ages.juliet < ages.ralph ∧
  ages.juliet = 10 ∧
  ages.maggie + ages.ralph = 19

/-- The theorem to be proved -/
theorem juliet_younger_than_ralph (ages : SiblingAges) 
  (h : problem_conditions ages) : ages.ralph - ages.juliet = 2 := by
  sorry


end juliet_younger_than_ralph_l40_4091


namespace g_one_equals_three_l40_4085

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem g_one_equals_three (f g : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_even : is_even_function g) 
  (h_eq1 : f (-1) + g 1 = 2) 
  (h_eq2 : f 1 + g (-1) = 4) : 
  g 1 = 3 := by sorry

end g_one_equals_three_l40_4085


namespace difference_theorem_l40_4086

def difference (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

theorem difference_theorem :
  (difference (-2) (-4) 1 = -5/3) ∧
  (2/3 = max
    (max (difference (-2) (-4) 1) (difference (-2) 1 (-4)))
    (max (difference (-4) (-2) 1) (max (difference (-4) 1 (-2)) (max (difference 1 (-4) (-2)) (difference 1 (-2) (-4)))))) ∧
  (∀ x : ℚ, difference (-1) 6 x = 2 ↔ (x = -7 ∨ x = 8)) :=
by sorry

end difference_theorem_l40_4086


namespace remainder_2673_base12_div_9_l40_4092

/-- Converts a base-12 integer to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of 2673 --/
def base12_2673 : List Nat := [2, 6, 7, 3]

theorem remainder_2673_base12_div_9 :
  (base12ToDecimal base12_2673) % 9 = 8 := by sorry

end remainder_2673_base12_div_9_l40_4092


namespace orange_beads_count_l40_4027

/-- Represents the number of beads of each color in a necklace -/
structure NecklaceComposition where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Represents the total number of beads available for each color -/
def TotalBeads : ℕ := 45

/-- The composition of beads in each necklace -/
def necklace : NecklaceComposition := {
  green := 9,
  white := 6,
  orange := 9  -- This is what we want to prove
}

/-- The maximum number of necklaces that can be made -/
def maxNecklaces : ℕ := 5

theorem orange_beads_count :
  necklace.orange = 9 ∧
  necklace.green * maxNecklaces = TotalBeads ∧
  necklace.white * maxNecklaces ≤ TotalBeads ∧
  necklace.orange * maxNecklaces = TotalBeads :=
by sorry

end orange_beads_count_l40_4027


namespace congruence_product_l40_4052

theorem congruence_product (a b c d m : ℤ) : 
  a ≡ b [ZMOD m] → c ≡ d [ZMOD m] → (a * c) ≡ (b * d) [ZMOD m] := by
  sorry

end congruence_product_l40_4052


namespace line_plane_perpendicularity_l40_4043

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) (h : subset n α) :
  (perp_line m n → perp_plane m α) ∧ 
  ¬(perp_line m n ↔ perp_plane m α) :=
sorry

end line_plane_perpendicularity_l40_4043


namespace min_coins_is_four_l40_4045

/-- The minimum number of coins Ana can have -/
def min_coins : ℕ :=
  let initial_coins := 22
  let operations := [6, 18, -12]
  sorry

/-- Theorem: The minimum number of coins Ana can have is 4 -/
theorem min_coins_is_four : min_coins = 4 := by
  sorry

end min_coins_is_four_l40_4045
