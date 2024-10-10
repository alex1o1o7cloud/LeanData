import Mathlib

namespace range_of_g_l2967_296720

theorem range_of_g (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  π/4 ≤ Real.arcsin x + Real.arccos x - Real.arctan x ∧ 
  Real.arcsin x + Real.arccos x - Real.arctan x ≤ 3*π/4 := by
  sorry

end range_of_g_l2967_296720


namespace number_added_to_product_l2967_296777

theorem number_added_to_product (a b : Int) (h1 : a = -2) (h2 : b = -3) :
  ∃ x : Int, a * b + x = 65 ∧ x = 59 := by
sorry

end number_added_to_product_l2967_296777


namespace consecutive_odd_integers_sum_l2967_296701

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n + (n + 4) = 150) → (n + (n + 2) + (n + 4) = 225) := by
  sorry

end consecutive_odd_integers_sum_l2967_296701


namespace concentric_circles_area_ratio_l2967_296754

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let A₁ : ℝ := Real.pi * r₁ ^ 2  -- area of smaller circle
  let A₂ : ℝ := Real.pi * r₂ ^ 2  -- area of larger circle
  (A₂ - A₁) / A₁ = 8 :=
by sorry

end concentric_circles_area_ratio_l2967_296754


namespace min_ellipse_eccentricity_l2967_296726

/-- Given an ellipse C: x²/a² + y²/b² = 1 where a > b > 0, with foci F₁ and F₂,
    and right vertex A. A line l passing through F₁ intersects C at P and Q.
    AP · AQ = (1/2)(a+c)². The minimum eccentricity of C is 1 - √2/2. -/
theorem min_ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := c / a
  ∀ P Q : ℝ × ℝ,
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →
  (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) →
  ∃ m : ℝ, (P.1 = m * P.2 - c) ∧ (Q.1 = m * Q.2 - c) →
  ((P.1 - a) * (Q.1 - a) + P.2 * Q.2 = (1/2) * (a + c)^2) →
  e ≥ 1 - Real.sqrt 2 / 2 :=
sorry

end min_ellipse_eccentricity_l2967_296726


namespace school_fee_calculation_l2967_296769

def mother_money : ℚ :=
  2 * 100 + 1 * 50 + 5 * 20 + 3 * 10 + 4 * 5 + 6 * 0.25 + 10 * 0.1 + 5 * 0.05

def father_money : ℚ :=
  3 * 100 + 4 * 50 + 2 * 20 + 1 * 10 + 6 * 5 + 8 * 0.25 + 7 * 0.1 + 3 * 0.05

def school_fee : ℚ := mother_money + father_money

theorem school_fee_calculation : school_fee = 985.60 := by
  sorry

end school_fee_calculation_l2967_296769


namespace sum_of_three_numbers_l2967_296797

theorem sum_of_three_numbers : 1.35 + 0.123 + 0.321 = 1.794 := by
  sorry

end sum_of_three_numbers_l2967_296797


namespace covering_recurrence_l2967_296708

/-- Number of ways to cover a 2 × n rectangle with 1 × 2 pieces -/
def coveringWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => coveringWays (n + 1) + coveringWays n

/-- The recurrence relation for covering a 2 × n rectangle with 1 × 2 pieces -/
theorem covering_recurrence (n : ℕ) (h : n ≥ 2) :
  coveringWays n = coveringWays (n - 1) + coveringWays (n - 2) := by
  sorry

end covering_recurrence_l2967_296708


namespace range_of_m_l2967_296776

/-- Given points A(1,0) and B(4,0) in the Cartesian plane, and a point P on the line x-y+m=0 
    such that 2PA = PB, the range of possible values for m is [-2√2, 2√2]. -/
theorem range_of_m (m : ℝ) : 
  ∃ (x y : ℝ), 
    (x - y + m = 0) ∧ 
    (2 * ((x - 1)^2 + y^2) = (x - 4)^2 + y^2) →
    -2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2 := by
  sorry

end range_of_m_l2967_296776


namespace quadratic_function_properties_l2967_296710

/-- Quadratic function passing through (2,3) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 2) * x + 3

theorem quadratic_function_properties :
  ∃ a : ℝ,
  (f a 2 = 3) ∧
  (∀ x : ℝ, 0 < x → x < 3 → 2 ≤ f 0 x ∧ f 0 x < 6) ∧
  (∀ m y₁ y₂ : ℝ, f 0 (m - 1) = y₁ → f 0 m = y₂ → y₁ > y₂ → m < 3/2) :=
sorry

end quadratic_function_properties_l2967_296710


namespace sum_of_xyz_l2967_296736

theorem sum_of_xyz (x y z : ℕ+) (h : x + 2*x*y + 3*x*y*z = 115) : x + y + z = 10 := by
  sorry

end sum_of_xyz_l2967_296736


namespace circle_equation_l2967_296703

/-- A circle C with points A and B, and a chord intercepted by a line --/
structure CircleWithPointsAndChord where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- Point A on the circle
  pointA : ℝ × ℝ
  -- Point B on the circle
  pointB : ℝ × ℝ
  -- Length of the chord intercepted by the line x-y-2=0
  chordLength : ℝ
  -- Ensure A and B are on the circle
  h_pointA_on_circle : (pointA.1 - center.1)^2 + (pointA.2 - center.2)^2 = radius^2
  h_pointB_on_circle : (pointB.1 - center.1)^2 + (pointB.2 - center.2)^2 = radius^2
  -- Ensure the chord length is correct
  h_chord_length : chordLength = Real.sqrt 2

/-- The theorem stating that the circle satisfying the given conditions has the equation (x-1)² + y² = 1 --/
theorem circle_equation (c : CircleWithPointsAndChord) 
  (h_pointA : c.pointA = (1, 1)) 
  (h_pointB : c.pointB = (2, 0)) :
  c.center = (1, 0) ∧ c.radius = 1 :=
sorry

end circle_equation_l2967_296703


namespace six_solved_only_b_l2967_296779

/-- Represents the number of students who solved specific combinations of problems -/
structure ProblemSolvers where
  a : ℕ  -- only A
  b : ℕ  -- only B
  c : ℕ  -- only C
  ab : ℕ  -- A and B
  bc : ℕ  -- B and C
  ca : ℕ  -- C and A
  abc : ℕ  -- all three

/-- The conditions of the math competition problem -/
def competition_conditions (s : ProblemSolvers) : Prop :=
  -- Total number of students is 25
  s.a + s.b + s.c + s.ab + s.bc + s.ca + s.abc = 25 ∧
  -- Among students who didn't solve A, those who solved B is twice those who solved C
  s.b + s.bc = 2 * (s.c + s.bc) ∧
  -- Among students who solved A, those who solved only A is one more than those who solved A and others
  s.a = (s.ab + s.ca + s.abc) + 1 ∧
  -- Among students who solved only one problem, half didn't solve A
  s.a = s.b + s.c

/-- The theorem stating that 6 students solved only problem B -/
theorem six_solved_only_b :
  ∃ (s : ProblemSolvers), competition_conditions s ∧ s.b = 6 :=
sorry

end six_solved_only_b_l2967_296779


namespace circle_diameter_from_area_l2967_296740

theorem circle_diameter_from_area :
  ∀ (A r d : ℝ),
  A = 225 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 30 := by sorry

end circle_diameter_from_area_l2967_296740


namespace charles_journey_l2967_296728

/-- Represents the distance traveled by Charles -/
def total_distance : ℝ := 1800

/-- Represents the speed for the first half of the journey -/
def speed1 : ℝ := 90

/-- Represents the speed for the second half of the journey -/
def speed2 : ℝ := 180

/-- Represents the total time of the journey -/
def total_time : ℝ := 30

/-- Theorem stating that given the conditions of Charles' journey, the total distance is 1800 miles -/
theorem charles_journey :
  (total_distance / 2 / speed1 + total_distance / 2 / speed2 = total_time) →
  total_distance = 1800 :=
by sorry

end charles_journey_l2967_296728


namespace traffic_light_color_change_probability_l2967_296719

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the total time when color changes occur -/
def colorChangeDuration (cycle : TrafficLightCycle) : ℕ :=
  3 * 5  -- 5 seconds for each color change

/-- Theorem: The probability of observing a color change is 3/20 -/
theorem traffic_light_color_change_probability
  (cycle : TrafficLightCycle)
  (h1 : cycle.green = 45)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 50)
  (h4 : colorChangeDuration cycle = 15) :
  (colorChangeDuration cycle : ℚ) / (cycleDuration cycle : ℚ) = 3 / 20 := by
  sorry

end traffic_light_color_change_probability_l2967_296719


namespace smallest_m_for_inequality_l2967_296786

theorem smallest_m_for_inequality : 
  ∃ (m : ℝ), (∀ (a b c : ℕ+), a + b + c = 1 → 
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) ∧ 
  (∀ (m' : ℝ), m' < m → 
    ∃ (a b c : ℕ+), a + b + c = 1 ∧ 
    m' * (a^3 + b^3 + c^3) < 6 * (a^2 + b^2 + c^2) + 1) ∧
  m = 27 := by
sorry

end smallest_m_for_inequality_l2967_296786


namespace geometric_relations_l2967_296756

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the theorem
theorem geometric_relations 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular_plane m α ∧ perpendicular_plane n β ∧ perpendicular m n → perpendicular_planes α β) ∧
  (perpendicular_plane m α ∧ parallel n β ∧ parallel_planes α β → perpendicular m n) := by
  sorry

end geometric_relations_l2967_296756


namespace product_divisible_by_twelve_l2967_296745

theorem product_divisible_by_twelve (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) := by
  sorry

end product_divisible_by_twelve_l2967_296745


namespace orange_juice_percentage_l2967_296772

def pear_juice_yield : ℚ := 10 / 2
def orange_juice_yield : ℚ := 6 / 3
def pears_used : ℕ := 4
def oranges_used : ℕ := 6

theorem orange_juice_percentage : 
  (oranges_used * orange_juice_yield) / ((pears_used * pear_juice_yield) + (oranges_used * orange_juice_yield)) = 375 / 1000 :=
by sorry

end orange_juice_percentage_l2967_296772


namespace certain_number_proof_l2967_296748

theorem certain_number_proof : ∃ x : ℝ, (0.80 * x = 0.50 * 960) ∧ (x = 600) := by
  sorry

end certain_number_proof_l2967_296748


namespace binomial_square_coefficient_l2967_296700

theorem binomial_square_coefficient (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end binomial_square_coefficient_l2967_296700


namespace yarn_length_difference_l2967_296704

theorem yarn_length_difference (green_length red_length : ℝ) : 
  green_length = 156 →
  red_length > 3 * green_length →
  green_length + red_length = 632 →
  red_length - 3 * green_length = 8 := by
sorry

end yarn_length_difference_l2967_296704


namespace kylie_coins_to_laura_l2967_296750

/-- The number of coins Kylie collected from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie collected from her brother -/
def brother_coins : ℕ := 13

/-- The number of coins Kylie collected from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie had left after giving some to Laura -/
def coins_left : ℕ := 15

/-- The total number of coins Kylie collected -/
def total_coins : ℕ := piggy_bank_coins + brother_coins + father_coins

/-- The number of coins Kylie gave to Laura -/
def coins_given_to_laura : ℕ := total_coins - coins_left

theorem kylie_coins_to_laura : coins_given_to_laura = 21 := by
  sorry

end kylie_coins_to_laura_l2967_296750


namespace simplify_trig_expression_simplify_trig_product_l2967_296784

-- Part 1
theorem simplify_trig_expression (α : Real) :
  (Real.sin (α - π/2) + Real.cos (3*π/2 + α)) / (Real.sin (π - α) + Real.cos (3*π + α)) =
  1 / 0 := by sorry

-- Part 2
theorem simplify_trig_product :
  Real.sin (40 * π/180) * (Real.tan (10 * π/180) - Real.sqrt 3) =
  -Real.sin (80 * π/180) / Real.cos (10 * π/180) := by sorry

end simplify_trig_expression_simplify_trig_product_l2967_296784


namespace balloon_permutations_count_l2967_296792

/-- The number of distinct permutations of a 7-letter word with two pairs of repeated letters -/
def balloon_permutations : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "balloon" is 1260 -/
theorem balloon_permutations_count : balloon_permutations = 1260 := by
  sorry

end balloon_permutations_count_l2967_296792


namespace following_pierre_better_than_guessing_l2967_296732

-- Define the probability of Pierre giving correct information
def pierre_correct_prob : ℚ := 3/4

-- Define the probability of Pierre giving incorrect information
def pierre_incorrect_prob : ℚ := 1/4

-- Define the probability of Jean guessing correctly for one event
def jean_guess_prob : ℚ := 1/2

-- Define the probability of Jean getting both dates correct when following Pierre's advice
def jean_correct_following_pierre : ℚ :=
  pierre_correct_prob * (pierre_correct_prob * pierre_correct_prob) +
  pierre_incorrect_prob * (pierre_incorrect_prob * pierre_incorrect_prob)

-- Define the probability of Jean getting both dates correct when guessing randomly
def jean_correct_guessing : ℚ := jean_guess_prob * jean_guess_prob

-- Theorem stating that following Pierre's advice is better than guessing randomly
theorem following_pierre_better_than_guessing :
  jean_correct_following_pierre > jean_correct_guessing :=
by sorry

end following_pierre_better_than_guessing_l2967_296732


namespace time_for_order_l2967_296749

/-- Represents the time it takes to make one shirt -/
def shirt_time : ℝ := 1

/-- Represents the time it takes to make one pair of pants -/
def pants_time : ℝ := 2 * shirt_time

/-- Represents the time it takes to make one jacket -/
def jacket_time : ℝ := 3 * shirt_time

/-- The total time to make 2 shirts, 3 pairs of pants, and 4 jackets is 10 hours -/
axiom total_time_10 : 2 * shirt_time + 3 * pants_time + 4 * jacket_time = 10

/-- Theorem: It takes 20 working hours to make 14 shirts, 10 pairs of pants, and 2 jackets -/
theorem time_for_order : 14 * shirt_time + 10 * pants_time + 2 * jacket_time = 20 := by
  sorry

end time_for_order_l2967_296749


namespace E_parity_l2967_296715

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 1) + E n

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem E_parity : is_odd (E 2021) ∧ is_even (E 2022) ∧ is_odd (E 2023) := by sorry

end E_parity_l2967_296715


namespace remove_six_maximizes_probability_l2967_296789

def original_list : List Int := List.range 15 |>.map (λ x => x - 2)

def remove_number (list : List Int) (n : Int) : List Int :=
  list.filter (λ x => x ≠ n)

def count_pairs_sum_11 (list : List Int) : Nat :=
  list.filterMap (λ x => 
    if x < 11 ∧ list.contains (11 - x) ∧ x ≠ 11 - x
    then some (x, 11 - x)
    else none
  ) |>.length

theorem remove_six_maximizes_probability :
  ∀ n ∈ original_list, n ≠ 6 →
    count_pairs_sum_11 (remove_number original_list 6) ≥ 
    count_pairs_sum_11 (remove_number original_list n) :=
by sorry

end remove_six_maximizes_probability_l2967_296789


namespace quadratic_roots_arithmetic_sequence_l2967_296731

theorem quadratic_roots_arithmetic_sequence (a b : ℚ) : 
  a ≠ b →
  (∃ x₁ x₂ x₃ x₄ : ℚ, 
    (x₁^2 - x₁ + a = 0 ∧ x₂^2 - x₂ + a = 0) ∧ 
    (x₃^2 - x₃ + b = 0 ∧ x₄^2 - x₄ + b = 0) ∧
    (∃ d : ℚ, x₁ = 1/4 ∧ x₂ = x₁ + d ∧ x₃ = x₂ + d ∧ x₄ = x₃ + d)) →
  a + b = 31/72 := by
sorry

end quadratic_roots_arithmetic_sequence_l2967_296731


namespace cup_sales_problem_l2967_296788

/-- Proves that the number of additional days is 11, given the conditions of the cup sales problem -/
theorem cup_sales_problem (first_day_sales : ℕ) (daily_sales : ℕ) (average_sales : ℚ) : 
  first_day_sales = 86 →
  daily_sales = 50 →
  average_sales = 53 →
  ∃ d : ℕ, 
    (first_day_sales + d * daily_sales : ℚ) / (d + 1 : ℚ) = average_sales ∧
    d = 11 := by
  sorry


end cup_sales_problem_l2967_296788


namespace fraction_sum_integer_implies_fractions_integer_l2967_296787

theorem fraction_sum_integer_implies_fractions_integer 
  (x y : ℕ+) 
  (h : ∃ (k : ℤ), (x.val^2 - 1 : ℤ) / (y.val + 1) + (y.val^2 - 1 : ℤ) / (x.val + 1) = k) :
  (∃ (m : ℤ), (x.val^2 - 1 : ℤ) / (y.val + 1) = m) ∧ 
  (∃ (n : ℤ), (y.val^2 - 1 : ℤ) / (x.val + 1) = n) :=
by sorry

end fraction_sum_integer_implies_fractions_integer_l2967_296787


namespace ship_length_l2967_296758

/-- The length of a ship given its speed and time to cross a lighthouse -/
theorem ship_length (speed : ℝ) (time : ℝ) : 
  speed = 18 → time = 20 → speed * time * (1000 / 3600) = 100 := by
  sorry

#check ship_length

end ship_length_l2967_296758


namespace largest_prime_factor_of_sum_factorials_l2967_296721

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 5 + factorial 6

theorem largest_prime_factor_of_sum_factorials :
  (Nat.factors sum_of_factorials).maximum? = some 7 := by
  sorry

end largest_prime_factor_of_sum_factorials_l2967_296721


namespace kitchen_floor_theorem_l2967_296706

/-- Calculates the area of the kitchen floor given the total mopping time,
    mopping rate, and bathroom floor area. -/
def kitchen_floor_area (total_time : ℕ) (mopping_rate : ℕ) (bathroom_area : ℕ) : ℕ :=
  total_time * mopping_rate - bathroom_area

/-- Proves that the kitchen floor area is 80 square feet given the specified conditions. -/
theorem kitchen_floor_theorem :
  kitchen_floor_area 13 8 24 = 80 := by
  sorry

#eval kitchen_floor_area 13 8 24

end kitchen_floor_theorem_l2967_296706


namespace triangle_similarity_FC_length_l2967_296781

theorem triangle_similarity_FC_length
  (DC : ℝ) (CB : ℝ) (AD : ℝ) (AB : ℝ) (ED : ℝ) (FC : ℝ)
  (h1 : DC = 10)
  (h2 : CB = 5)
  (h3 : AB = (1/3) * AD)
  (h4 : ED = (4/5) * AD)
  : FC = 10 := by
  sorry

end triangle_similarity_FC_length_l2967_296781


namespace mrs_hilt_pencil_purchase_l2967_296738

/-- Given Mrs. Hilt's purchases at the school store, prove the number of pencils she bought. -/
theorem mrs_hilt_pencil_purchase
  (total_spent : ℕ)
  (notebook_cost : ℕ)
  (ruler_cost : ℕ)
  (pencil_cost : ℕ)
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : ruler_cost = 18)
  (h4 : pencil_cost = 7)
  : (total_spent - notebook_cost - ruler_cost) / pencil_cost = 3 :=
by sorry

end mrs_hilt_pencil_purchase_l2967_296738


namespace expression_evaluation_l2967_296795

theorem expression_evaluation : (2.1 * (49.7 + 0.3) + 15 : ℝ) = 120 := by sorry

end expression_evaluation_l2967_296795


namespace competition_selection_count_l2967_296791

def male_count : ℕ := 5
def female_count : ℕ := 3
def selection_size : ℕ := 3

def selection_count : ℕ := 45

theorem competition_selection_count :
  (Nat.choose female_count 2 * Nat.choose male_count 1) +
  (Nat.choose female_count 1 * Nat.choose male_count 2) = selection_count :=
by sorry

end competition_selection_count_l2967_296791


namespace smallest_a_for_two_zeros_in_unit_interval_l2967_296793

theorem smallest_a_for_two_zeros_in_unit_interval :
  ∃ (a b c : ℤ), 
    a = 5 ∧
    (∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
      a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) ∧
    (∀ (a' b' c' : ℤ), a' > 0 ∧ a' < 5 →
      ¬(∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
        a' * x^2 - b' * x + c' = 0 ∧ a' * y^2 - b' * y + c' = 0)) :=
sorry

end smallest_a_for_two_zeros_in_unit_interval_l2967_296793


namespace rectangle_square_area_ratio_l2967_296730

/-- Given a square S and a rectangle R, where the longer side of R is 20% more than
    the side of S, the shorter side of R is 20% less than the side of S, and the
    diagonal of R is 10% longer than the diagonal of S, prove that the ratio of
    the area of R to the area of S is 24/25. -/
theorem rectangle_square_area_ratio 
  (S : Real) -- Side length of square S
  (R_long : Real) -- Longer side of rectangle R
  (R_short : Real) -- Shorter side of rectangle R
  (R_diag : Real) -- Diagonal of rectangle R
  (h1 : R_long = 1.2 * S) -- Longer side of R is 20% more than side of S
  (h2 : R_short = 0.8 * S) -- Shorter side of R is 20% less than side of S
  (h3 : R_diag = 1.1 * S * Real.sqrt 2) -- Diagonal of R is 10% longer than diagonal of S
  : (R_long * R_short) / (S * S) = 24 / 25 := by
  sorry

end rectangle_square_area_ratio_l2967_296730


namespace circle_line_segments_l2967_296718

/-- The number of line segments formed by joining each pair of n distinct points on a circle -/
def lineSegments (n : ℕ) : ℕ := n.choose 2

/-- There are 8 distinct points on a circle -/
def numPoints : ℕ := 8

theorem circle_line_segments :
  lineSegments numPoints = 28 := by
  sorry

end circle_line_segments_l2967_296718


namespace parabola_c_value_l2967_296759

/-- A parabola with equation y = 2x^2 + bx + c passing through (-2, 20) and (2, 28) has c = 16 -/
theorem parabola_c_value (b c : ℝ) : 
  (∀ x y : ℝ, y = 2 * x^2 + b * x + c → 
    ((x = -2 ∧ y = 20) ∨ (x = 2 ∧ y = 28))) → 
  c = 16 := by sorry

end parabola_c_value_l2967_296759


namespace hyperbola_eccentricity_l2967_296778

theorem hyperbola_eccentricity (a b e : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = Real.sqrt (2 * e - 1) * x) →
  e = 2 := by
sorry

end hyperbola_eccentricity_l2967_296778


namespace complex_absolute_value_product_l2967_296711

theorem complex_absolute_value_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end complex_absolute_value_product_l2967_296711


namespace unique_congruent_integer_l2967_296773

theorem unique_congruent_integer : ∃! n : ℤ, 5 ≤ n ∧ n ≤ 10 ∧ n ≡ 12345 [ZMOD 6] ∧ n = 9 := by
  sorry

end unique_congruent_integer_l2967_296773


namespace inequality_solution_range_l2967_296757

theorem inequality_solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∃ m : ℝ, (4 / (x + 1) + 1 / y < m^2 + 3/2 * m)) →
  (∃ m : ℝ, m < -3 ∨ m > 3/2) :=
by sorry

end inequality_solution_range_l2967_296757


namespace regions_for_twenty_points_l2967_296733

/-- The number of regions created by chords in a circle --/
def num_regions (n : ℕ) : ℕ :=
  let vertices := n + (n.choose 4)
  let edges := (n * (n - 1) + 2 * (n.choose 4)) / 2
  edges - vertices + 1

/-- Theorem stating the number of regions for 20 points --/
theorem regions_for_twenty_points :
  num_regions 20 = 5036 := by
  sorry

end regions_for_twenty_points_l2967_296733


namespace fraction_of_task_completed_l2967_296707

theorem fraction_of_task_completed (total_time minutes : ℕ) (h : total_time = 60) (h2 : minutes = 15) :
  (minutes : ℚ) / total_time = 1 / 4 := by
  sorry

end fraction_of_task_completed_l2967_296707


namespace monomial_count_l2967_296798

/-- A function that determines if an expression is a monomial -/
def isMonomial (expr : String) : Bool :=
  match expr with
  | "-1" => true
  | "-1/2*a^2" => true
  | "2/3*x^2*y" => true
  | "a*b^2/π" => true
  | "ab/c" => false
  | "3a-b" => false
  | "0" => true
  | "(x-1)/2" => false
  | _ => false

/-- The list of expressions to check -/
def expressions : List String :=
  ["-1", "-1/2*a^2", "2/3*x^2*y", "a*b^2/π", "ab/c", "3a-b", "0", "(x-1)/2"]

/-- Counts the number of monomials in the list of expressions -/
def countMonomials (exprs : List String) : Nat :=
  exprs.filter isMonomial |>.length

theorem monomial_count :
  countMonomials expressions = 5 := by sorry

end monomial_count_l2967_296798


namespace reflection_across_x_axis_l2967_296771

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (2, -1)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (2, 1)

theorem reflection_across_x_axis :
  reflect_x original_point = reflected_point := by sorry

end reflection_across_x_axis_l2967_296771


namespace trig_simplification_l2967_296764

theorem trig_simplification (x y z : ℝ) :
  Real.sin (x - y + z) * Real.cos y - Real.cos (x - y + z) * Real.sin y = Real.sin (x - 2*y + z) := by
  sorry

end trig_simplification_l2967_296764


namespace tom_initial_investment_l2967_296743

/-- Represents the investment scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_initial : ℕ  -- Tom's initial investment
  jose_investment : ℕ := 4500  -- Jose's investment
  total_profit : ℕ := 5400  -- Total profit after one year
  jose_profit : ℕ := 3000  -- Jose's share of the profit
  tom_months : ℕ := 12  -- Months Tom invested
  jose_months : ℕ := 10  -- Months Jose invested

/-- Theorem stating that Tom's initial investment was 3000 --/
theorem tom_initial_investment (shop : ShopInvestment) : shop.tom_initial = 3000 := by
  sorry

end tom_initial_investment_l2967_296743


namespace eight_percent_of_fifty_l2967_296763

theorem eight_percent_of_fifty : ∃ x : ℝ, x = 50 * 0.08 ∧ x = 4 := by
  sorry

end eight_percent_of_fifty_l2967_296763


namespace correct_delivery_probability_l2967_296746

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- Probability of exactly k out of n packages being delivered to the correct houses -/
def prob_correct_delivery (n k : ℕ) : ℚ :=
  (Nat.choose n k * (Nat.factorial k) * (Nat.factorial (n - k) / Nat.factorial n)) / Nat.factorial n

theorem correct_delivery_probability :
  prob_correct_delivery n k = 1 / 12 :=
sorry

end correct_delivery_probability_l2967_296746


namespace fraction_addition_l2967_296735

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l2967_296735


namespace peter_soda_purchase_l2967_296709

/-- The cost of soda per ounce in dollars -/
def soda_cost_per_ounce : ℚ := 25 / 100

/-- The amount Peter brought in dollars -/
def initial_amount : ℚ := 2

/-- The amount Peter left with in dollars -/
def remaining_amount : ℚ := 1 / 2

/-- The number of ounces of soda Peter bought -/
def soda_ounces : ℚ := (initial_amount - remaining_amount) / soda_cost_per_ounce

theorem peter_soda_purchase : soda_ounces = 6 := by
  sorry

end peter_soda_purchase_l2967_296709


namespace value_of_a_l2967_296799

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^2 + 14
def g (x : ℝ) : ℝ := x^3 - 4

-- State the theorem
theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 18) :
  a = (4 + 2 * Real.sqrt 3 / 3) ^ (1/3) := by
  sorry


end value_of_a_l2967_296799


namespace sum_fraction_bounds_l2967_296790

theorem sum_fraction_bounds (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let S := (a / (a + b + d)) + (b / (b + c + a)) + (c / (c + d + b)) + (d / (d + a + c))
  1 < S ∧ S < 2 := by
  sorry

end sum_fraction_bounds_l2967_296790


namespace exists_equal_area_split_line_l2967_296744

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the four circles
def circles : List Circle := [
  { center := (14, 92), radius := 5 },
  { center := (17, 76), radius := 5 },
  { center := (19, 84), radius := 5 },
  { center := (25, 90), radius := 5 }
]

-- Define a line passing through a point with a given slope
structure Line where
  point : ℝ × ℝ
  slope : ℝ

-- Function to calculate the area of a circle segment cut by a line
def circleSegmentArea (c : Circle) (l : Line) : ℝ := sorry

-- Function to calculate the total area of circle segments on one side of the line
def totalSegmentArea (cs : List Circle) (l : Line) : ℝ := sorry

-- Theorem statement
theorem exists_equal_area_split_line :
  ∃ m : ℝ, let l := { point := (17, 76), slope := m }
    totalSegmentArea circles l = (1/2) * (List.sum (circles.map (fun c => π * c.radius^2))) :=
sorry

end exists_equal_area_split_line_l2967_296744


namespace triangle_arithmetic_sequence_triangle_area_l2967_296727

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  3 * t.b^2 = 2 * t.a * t.c * (1 + Real.cos t.B)

-- Define arithmetic sequence property
def isArithmeticSequence (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Theorem 1
theorem triangle_arithmetic_sequence (t : Triangle) 
  (h : satisfiesCondition t) : isArithmeticSequence t.a t.b t.c := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : t.a = 3) (h2 : t.b = 5) (h3 : satisfiesCondition t) : 
  (1/2 * t.a * t.b * Real.sin t.C) = 15 * Real.sqrt 3 / 4 := by
  sorry

end triangle_arithmetic_sequence_triangle_area_l2967_296727


namespace covering_circles_highest_point_covered_l2967_296722

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The highest point of a circle -/
def highestPoint (c : Circle) : ℝ × ℝ :=
  (c.center.1, c.center.2 + c.radius)

/-- Check if a point is inside or on a circle -/
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

/-- A set of 101 unit circles where the first 100 cover the 101st -/
structure CoveringCircles where
  circles : Fin 101 → Circle
  all_unit : ∀ i, (circles i).radius = 1
  last_covered : ∀ p, isInside p (circles 100) → ∃ i < 100, isInside p (circles i)
  all_distinct : ∀ i j, i ≠ j → circles i ≠ circles j

theorem covering_circles_highest_point_covered (cc : CoveringCircles) :
  ∃ i j, i < 100 ∧ j < 100 ∧ i ≠ j ∧
    isInside (highestPoint (cc.circles j)) (cc.circles i) :=
  sorry

end covering_circles_highest_point_covered_l2967_296722


namespace factorization_of_difference_of_squares_l2967_296739

theorem factorization_of_difference_of_squares (a b : ℝ) :
  3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by sorry

end factorization_of_difference_of_squares_l2967_296739


namespace x_intercepts_count_l2967_296724

theorem x_intercepts_count : 
  (⌊(100000 : ℝ) / Real.pi⌋ : ℤ) - (⌊(10000 : ℝ) / Real.pi⌋ : ℤ) = 28648 := by
  sorry

end x_intercepts_count_l2967_296724


namespace total_visits_equals_39_l2967_296767

/-- Calculates the total number of doctor visits in a year -/
def total_visits (visits_per_month_doc1 : ℕ) 
                 (months_between_visits_doc2 : ℕ) 
                 (visits_per_period_doc3 : ℕ) 
                 (months_per_period_doc3 : ℕ) : ℕ :=
  let months_in_year := 12
  let visits_doc1 := visits_per_month_doc1 * months_in_year
  let visits_doc2 := months_in_year / months_between_visits_doc2
  let periods_in_year := months_in_year / months_per_period_doc3
  let visits_doc3 := visits_per_period_doc3 * periods_in_year
  visits_doc1 + visits_doc2 + visits_doc3

/-- Theorem stating that the total visits in a year is 39 -/
theorem total_visits_equals_39 : 
  total_visits 2 2 3 4 = 39 := by
  sorry

end total_visits_equals_39_l2967_296767


namespace train_journey_time_l2967_296761

/-- Proves that if a train moving at 6/7 of its usual speed arrives 30 minutes late, then its usual journey time is 3 hours. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : (6 / 7 * usual_speed) * (usual_time + 1/2) = usual_speed * usual_time) : 
  usual_time = 3 := by
  sorry

end train_journey_time_l2967_296761


namespace coin_collection_problem_l2967_296742

theorem coin_collection_problem :
  ∀ (n d q : ℕ),
    n + d + q = 30 →
    d = n + 4 →
    5 * n + 10 * d + 25 * q = 410 →
    q = n + 2 :=
by
  sorry

end coin_collection_problem_l2967_296742


namespace g_solution_set_m_range_l2967_296782

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x - 8
def g (x : ℝ) : ℝ := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem g_solution_set :
  {x : ℝ | g x < 0} = {x : ℝ | -2 < x ∧ x < 4} := by sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 2 := by sorry

end g_solution_set_m_range_l2967_296782


namespace function_composition_problem_l2967_296714

/-- Given a function f(x) = ax - b where a > 0 and f(f(x)) = 4x - 3, prove that f(2) = 3 -/
theorem function_composition_problem (a b : ℝ) (h1 : a > 0) :
  (∀ x, ∃ f : ℝ → ℝ, f x = a * x - b) →
  (∀ x, ∃ f : ℝ → ℝ, f (f x) = 4 * x - 3) →
  ∃ f : ℝ → ℝ, f 2 = 3 :=
by sorry

end function_composition_problem_l2967_296714


namespace savings_difference_is_250_l2967_296734

def window_price : ℕ := 125
def offer_purchase : ℕ := 6
def offer_free : ℕ := 2
def dave_windows : ℕ := 9
def doug_windows : ℕ := 11

def calculate_cost (num_windows : ℕ) : ℕ :=
  let sets := num_windows / (offer_purchase + offer_free)
  let remainder := num_windows % (offer_purchase + offer_free)
  (sets * offer_purchase + min remainder offer_purchase) * window_price

def savings_difference : ℕ :=
  let separate_cost := calculate_cost dave_windows + calculate_cost doug_windows
  let combined_cost := calculate_cost (dave_windows + doug_windows)
  let separate_savings := dave_windows * window_price + doug_windows * window_price - separate_cost
  let combined_savings := (dave_windows + doug_windows) * window_price - combined_cost
  combined_savings - separate_savings

theorem savings_difference_is_250 : savings_difference = 250 := by
  sorry

end savings_difference_is_250_l2967_296734


namespace half_percent_to_decimal_l2967_296705

theorem half_percent_to_decimal : (1 / 2 : ℚ) / 100 = (0.005 : ℚ) := by
  sorry

end half_percent_to_decimal_l2967_296705


namespace count_solutions_2x_3y_763_l2967_296770

theorem count_solutions_2x_3y_763 : 
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 3 * p.2 = 763 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 764) (Finset.range 764))).card = 127 := by
  sorry

end count_solutions_2x_3y_763_l2967_296770


namespace greatest_integer_gcd_30_is_10_l2967_296753

theorem greatest_integer_gcd_30_is_10 : 
  ∃ n : ℕ, n < 100 ∧ Nat.gcd n 30 = 10 ∧ ∀ m : ℕ, m < 100 → Nat.gcd m 30 = 10 → m ≤ n :=
by
  -- The proof goes here
  sorry

end greatest_integer_gcd_30_is_10_l2967_296753


namespace tan_equality_solution_l2967_296751

theorem tan_equality_solution (n : ℤ) (h1 : -180 < n) (h2 : n < 180) 
  (h3 : Real.tan (n * π / 180) = Real.tan (123 * π / 180)) : 
  n = 123 ∨ n = -57 := by
  sorry

end tan_equality_solution_l2967_296751


namespace opposite_sign_sum_l2967_296780

theorem opposite_sign_sum (x y : ℝ) :
  (|x + 2| + |y - 4| = 0) → (x + y - 3 = -1) := by
  sorry

end opposite_sign_sum_l2967_296780


namespace sugar_measurement_l2967_296752

theorem sugar_measurement (sugar_needed : ℚ) (cup_capacity : ℚ) : 
  sugar_needed = 5/2 ∧ cup_capacity = 1/4 → sugar_needed / cup_capacity = 10 := by
  sorry

end sugar_measurement_l2967_296752


namespace smallest_integer_half_square_third_cube_l2967_296774

theorem smallest_integer_half_square_third_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (a : ℕ), n / 2 = a * a) ∧ 
  (∃ (b : ℕ), n / 3 = b * b * b) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), m / 2 = x * x) → 
    (∃ (y : ℕ), m / 3 = y * y * y) → 
    m ≥ n) ∧
  n = 648 :=
sorry

end smallest_integer_half_square_third_cube_l2967_296774


namespace complex_number_in_first_quadrant_l2967_296775

theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I / (1 + Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end complex_number_in_first_quadrant_l2967_296775


namespace sum_of_cubes_over_product_l2967_296760

theorem sum_of_cubes_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hsum : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := by
sorry

end sum_of_cubes_over_product_l2967_296760


namespace vector_calculation_l2967_296716

def a : Fin 2 → ℚ := ![1, 1]
def b : Fin 2 → ℚ := ![1, -1]

theorem vector_calculation : (1/2 : ℚ) • a - (3/2 : ℚ) • b = ![-1, 2] := by sorry

end vector_calculation_l2967_296716


namespace sum_of_possible_x_values_l2967_296747

/-- An isosceles triangle with two angles of 50° and x° --/
structure IsoscelesTriangle where
  x : ℝ
  is_isosceles : Bool
  has_50_degree_angle : Bool
  has_x_degree_angle : Bool

/-- The sum of angles in a triangle is 180° --/
axiom angle_sum (t : IsoscelesTriangle) : t.x + 50 + (180 - t.x - 50) = 180

/-- In an isosceles triangle, at least two angles are equal --/
axiom isosceles_equal_angles (t : IsoscelesTriangle) : t.is_isosceles → 
  (t.x = 50 ∨ t.x = (180 - 50) / 2 ∨ t.x = 180 - 2 * 50)

/-- The theorem to be proved --/
theorem sum_of_possible_x_values : 
  ∀ t : IsoscelesTriangle, t.is_isosceles ∧ t.has_50_degree_angle ∧ t.has_x_degree_angle → 
    50 + (180 - 50) / 2 + (180 - 2 * 50) = 195 := by
  sorry

end sum_of_possible_x_values_l2967_296747


namespace largest_remainder_l2967_296762

theorem largest_remainder (A B : ℕ) : 
  (A / 13 = 33) → (A % 13 = B) → (∀ C : ℕ, (C / 13 = 33) → (C % 13 ≤ B)) → A = 441 :=
by sorry

end largest_remainder_l2967_296762


namespace mothers_day_discount_l2967_296765

theorem mothers_day_discount (original_price : ℝ) (final_price : ℝ) 
  (additional_discount : ℝ) (h1 : original_price = 125) 
  (h2 : final_price = 108) (h3 : additional_discount = 0.04) : 
  ∃ (initial_discount : ℝ), 
    final_price = (1 - additional_discount) * (original_price * (1 - initial_discount)) ∧ 
    initial_discount = 0.1 := by
  sorry

end mothers_day_discount_l2967_296765


namespace class_size_l2967_296723

theorem class_size (poor_vision_percentage : ℝ) (glasses_percentage : ℝ) (glasses_count : ℕ) :
  poor_vision_percentage = 0.4 →
  glasses_percentage = 0.7 →
  glasses_count = 21 →
  ∃ total_students : ℕ, 
    (poor_vision_percentage * glasses_percentage * total_students : ℝ) = glasses_count ∧
    total_students = 75 :=
by sorry

end class_size_l2967_296723


namespace rational_equation_solution_l2967_296725

theorem rational_equation_solution : 
  ∃ (x : ℚ), (x + 11) / (x - 4) = (x - 3) / (x + 7) ∧ x = -13/5 := by
  sorry

end rational_equation_solution_l2967_296725


namespace factorial_difference_l2967_296702

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end factorial_difference_l2967_296702


namespace range_of_s_l2967_296737

/-- A decreasing function with central symmetry property -/
def DecreasingSymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧
  (∀ x, f (x - 1) = -f (2 - x))

/-- The main theorem -/
theorem range_of_s (f : ℝ → ℝ) (h : DecreasingSymmetricFunction f) :
  ∀ s : ℝ, f (s^2 - 2*s) + f (2 - s) ≤ 0 → s ≤ 1 ∨ s ≥ 2 := by
  sorry

end range_of_s_l2967_296737


namespace xiaoming_home_most_precise_l2967_296768

-- Define the possible descriptions of location
inductive LocationDescription
  | RightSide
  | Distance (d : ℝ)
  | WestSide
  | WestSideWithDistance (d : ℝ)

-- Define a function to check if a description is complete (has both direction and distance)
def isCompleteDescription (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.WestSideWithDistance _ => True
  | _ => False

-- Define Xiao Ming's home location
def xiaomingHome : LocationDescription := LocationDescription.WestSideWithDistance 900

-- Theorem: Xiao Ming's home location is the most precise description
theorem xiaoming_home_most_precise :
  isCompleteDescription xiaomingHome ∧
  ∀ (desc : LocationDescription), isCompleteDescription desc → desc = xiaomingHome :=
sorry

end xiaoming_home_most_precise_l2967_296768


namespace quadratic_inequality_l2967_296785

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 14 ≤ 0 ↔ 2 ≤ y ∧ y ≤ 7 := by
  sorry

end quadratic_inequality_l2967_296785


namespace f_minimum_value_l2967_296755

def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

theorem f_minimum_value :
  (∀ x y : ℝ, f x y ≥ 21.2) ∧ f 8 1 = 21.2 := by
  sorry

end f_minimum_value_l2967_296755


namespace moms_ice_cream_scoops_pierre_ice_cream_problem_l2967_296712

/-- Given the cost of ice cream scoops and the total bill, calculate the number of scoops Pierre's mom gets. -/
theorem moms_ice_cream_scoops (cost_per_scoop : ℕ) (pierres_scoops : ℕ) (total_bill : ℕ) : ℕ :=
  let moms_scoops := (total_bill - cost_per_scoop * pierres_scoops) / cost_per_scoop
  moms_scoops

/-- Prove that given the specific conditions, Pierre's mom gets 4 scoops of ice cream. -/
theorem pierre_ice_cream_problem :
  moms_ice_cream_scoops 2 3 14 = 4 := by
  sorry

end moms_ice_cream_scoops_pierre_ice_cream_problem_l2967_296712


namespace two_digit_primes_with_digit_sum_10_l2967_296766

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem two_digit_primes_with_digit_sum_10 :
  ∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ satisfies_condition n ∧ s.card = 3 :=
sorry

end two_digit_primes_with_digit_sum_10_l2967_296766


namespace simplify_expression_l2967_296729

/-- For all real numbers z, (2-3z) - (3+4z) = -1-7z -/
theorem simplify_expression (z : ℝ) : (2 - 3*z) - (3 + 4*z) = -1 - 7*z := by
  sorry

end simplify_expression_l2967_296729


namespace b_absolute_value_l2967_296794

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the polynomial g(x)
def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^5 + b * x^4 + c * x^3 + b * x + a

-- State the theorem
theorem b_absolute_value (a b c : ℤ) : 
  (g a b c (3 + i) = 0) →  -- Condition 1
  (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c) = 1) →  -- Condition 2 and 3
  (Int.natAbs b = 60) :=  -- Conclusion
by sorry

end b_absolute_value_l2967_296794


namespace fourth_person_height_l2967_296783

/-- Proves that given four people with heights in increasing order, where the difference
    between consecutive heights is 2, 2, and 6 inches respectively, and the average
    height is 77 inches, the height of the fourth person is 83 inches. -/
theorem fourth_person_height
  (h₁ h₂ h₃ h₄ : ℝ)
  (height_order : h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄)
  (diff_1_2 : h₂ - h₁ = 2)
  (diff_2_3 : h₃ - h₂ = 2)
  (diff_3_4 : h₄ - h₃ = 6)
  (avg_height : (h₁ + h₂ + h₃ + h₄) / 4 = 77) :
  h₄ = 83 := by
  sorry

end fourth_person_height_l2967_296783


namespace mean_of_numbers_l2967_296717

def numbers : List ℝ := [13, 8, 13, 21, 7, 23]

theorem mean_of_numbers : (numbers.sum / numbers.length : ℝ) = 14.1666667 := by
  sorry

end mean_of_numbers_l2967_296717


namespace cats_sold_l2967_296796

theorem cats_sold (ratio : ℚ) (dogs : ℕ) (cats : ℕ) : 
  ratio = 2 / 1 → dogs = 8 → cats = 16 := by
  sorry

end cats_sold_l2967_296796


namespace power_two_33_mod_9_l2967_296741

theorem power_two_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end power_two_33_mod_9_l2967_296741


namespace sum_of_digits_879_times_492_l2967_296713

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The theorem stating that the sum of digits in the product of 879 and 492 is 27 -/
theorem sum_of_digits_879_times_492 :
  sum_of_digits (879 * 492) = 27 := by
  sorry

#eval sum_of_digits (879 * 492)  -- This line is optional, for verification

end sum_of_digits_879_times_492_l2967_296713
