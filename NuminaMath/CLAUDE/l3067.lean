import Mathlib

namespace NUMINAMATH_CALUDE_boat_distance_calculation_l3067_306716

theorem boat_distance_calculation
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (total_time : ℝ)
  (h1 : boat_speed = 8)
  (h2 : stream_speed = 2)
  (h3 : total_time = 56)
  : ∃ (distance : ℝ),
    distance = 210 ∧
    total_time = distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_distance_calculation_l3067_306716


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l3067_306736

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides : 
  ∀ n : ℕ, 
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l3067_306736


namespace NUMINAMATH_CALUDE_shoes_lost_example_l3067_306746

/-- Given an initial number of shoe pairs and a maximum number of remaining pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (max_remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * max_remaining_pairs

/-- Theorem: Given 27 initial pairs of shoes and 22 maximum remaining pairs,
    the number of individual shoes lost is 10. -/
theorem shoes_lost_example : shoes_lost 27 22 = 10 := by
  sorry

end NUMINAMATH_CALUDE_shoes_lost_example_l3067_306746


namespace NUMINAMATH_CALUDE_problem_solution_l3067_306773

theorem problem_solution (x : ℝ) (h : x + Real.sqrt (x^2 - 1) + 1 / (x + Real.sqrt (x^2 - 1)) = 12) :
  x^3 + Real.sqrt (x^6 - 1) + 1 / (x^3 + Real.sqrt (x^6 - 1)) = 432 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3067_306773


namespace NUMINAMATH_CALUDE_income_of_M_l3067_306778

theorem income_of_M (M N O : ℝ) 
  (avg_MN : (M + N) / 2 = 5050)
  (avg_NO : (N + O) / 2 = 6250)
  (avg_MO : (M + O) / 2 = 5200) :
  M = 2666.67 := by
  sorry

end NUMINAMATH_CALUDE_income_of_M_l3067_306778


namespace NUMINAMATH_CALUDE_round_trip_speed_l3067_306760

theorem round_trip_speed (x : ℝ) : 
  x > 0 →
  (2 : ℝ) / ((1 / x) + (1 / 3)) = 5 →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_round_trip_speed_l3067_306760


namespace NUMINAMATH_CALUDE_complex_conjugate_roots_imply_real_coefficients_l3067_306756

theorem complex_conjugate_roots_imply_real_coefficients (a b : ℝ) :
  (∃ x y : ℝ, y ≠ 0 ∧ 
    (Complex.I * y + x) ^ 2 + (6 + Complex.I * a) * (Complex.I * y + x) + (13 + Complex.I * b) = 0 ∧
    (Complex.I * -y + x) ^ 2 + (6 + Complex.I * a) * (Complex.I * -y + x) + (13 + Complex.I * b) = 0) →
  a = 0 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_conjugate_roots_imply_real_coefficients_l3067_306756


namespace NUMINAMATH_CALUDE_lcm_is_perfect_square_l3067_306714

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) :
  ∃ k : ℕ, Nat.lcm a b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_is_perfect_square_l3067_306714


namespace NUMINAMATH_CALUDE_mike_marbles_l3067_306749

theorem mike_marbles (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 8 → given = 4 → remaining = initial - given → remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_marbles_l3067_306749


namespace NUMINAMATH_CALUDE_track_circumference_l3067_306726

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (v1 v2 t : ℝ) (h1 : v1 = 4.5) (h2 : v2 = 3.75) (h3 : t = 4.8 / 60) :
  2 * (v1 * t + v2 * t) = 1.32 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l3067_306726


namespace NUMINAMATH_CALUDE_chord_length_polar_circle_l3067_306739

/-- The length of the chord intercepted by the line tan θ = 1/2 on the circle ρ = 4sin θ is 16/5 -/
theorem chord_length_polar_circle (θ : Real) (ρ : Real) : 
  ρ = 4 * Real.sin θ → Real.tan θ = 1 / 2 → 
  2 * ρ * Real.sin θ = 16 / 5 := by sorry

end NUMINAMATH_CALUDE_chord_length_polar_circle_l3067_306739


namespace NUMINAMATH_CALUDE_student_arrangement_count_l3067_306788

/-- The number of ways to arrange students among attractions -/
def arrange_students (n_students : ℕ) (n_attractions : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange students among attractions when two specific students are at the same attraction -/
def arrange_students_with_pair (n_students : ℕ) (n_attractions : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of arrangements under given conditions -/
theorem student_arrangement_count :
  let n_students : ℕ := 4
  let n_attractions : ℕ := 3
  arrange_students n_students n_attractions - arrange_students_with_pair n_students n_attractions = 30 :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l3067_306788


namespace NUMINAMATH_CALUDE_temperature_at_noon_l3067_306792

/-- Given the lowest temperature of a day and the fact that the temperature at noon
    is 10°C higher, this theorem proves the temperature at noon. -/
theorem temperature_at_noon (a : ℝ) : 
  let lowest_temp := a
  let temp_diff := 10
  lowest_temp + temp_diff = a + 10 := by sorry

end NUMINAMATH_CALUDE_temperature_at_noon_l3067_306792


namespace NUMINAMATH_CALUDE_total_cost_for_all_puppies_l3067_306718

/-- Represents the cost calculation for a dog breed -/
structure BreedCost where
  mothers : Nat
  puppiesPerLitter : Nat
  shotCost : Nat
  shotsPerPuppy : Nat
  additionalCosts : Nat

/-- Calculates the total cost for a breed -/
def totalBreedCost (breed : BreedCost) : Nat :=
  breed.mothers * breed.puppiesPerLitter * 
  (breed.shotCost * breed.shotsPerPuppy + breed.additionalCosts)

/-- Theorem stating the total cost for all puppies -/
theorem total_cost_for_all_puppies :
  let goldenRetrievers : BreedCost := {
    mothers := 3,
    puppiesPerLitter := 4,
    shotCost := 5,
    shotsPerPuppy := 2,
    additionalCosts := 72  -- 6 months of vitamins at $12 per month
  }
  let germanShepherds : BreedCost := {
    mothers := 2,
    puppiesPerLitter := 5,
    shotCost := 8,
    shotsPerPuppy := 3,
    additionalCosts := 40  -- microchip ($25) + special toy ($15)
  }
  let bulldogs : BreedCost := {
    mothers := 4,
    puppiesPerLitter := 3,
    shotCost := 10,
    shotsPerPuppy := 4,
    additionalCosts := 38  -- customized collar ($20) + exclusive chew toy ($18)
  }
  totalBreedCost goldenRetrievers + totalBreedCost germanShepherds + totalBreedCost bulldogs = 2560 :=
by
  sorry


end NUMINAMATH_CALUDE_total_cost_for_all_puppies_l3067_306718


namespace NUMINAMATH_CALUDE_inequality_square_l3067_306786

theorem inequality_square (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_square_l3067_306786


namespace NUMINAMATH_CALUDE_tan_600_degrees_equals_sqrt_3_l3067_306752

theorem tan_600_degrees_equals_sqrt_3 : Real.tan (600 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_600_degrees_equals_sqrt_3_l3067_306752


namespace NUMINAMATH_CALUDE_infinite_nested_sqrt_l3067_306767

/-- Given that y is a non-negative real number satisfying y = √(2 - y), prove that y = 1 -/
theorem infinite_nested_sqrt (y : ℝ) (hy : y ≥ 0) (h : y = Real.sqrt (2 - y)) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_nested_sqrt_l3067_306767


namespace NUMINAMATH_CALUDE_exclusive_multiples_of_6_or_8_less_than_151_l3067_306712

def count_multiples (n m : ℕ) : ℕ := (n - 1) / m

def count_exclusive_multiples (upper bound1 bound2 : ℕ) : ℕ :=
  let lcm := Nat.lcm bound1 bound2
  (count_multiples upper bound1) + (count_multiples upper bound2) - 2 * (count_multiples upper lcm)

theorem exclusive_multiples_of_6_or_8_less_than_151 :
  count_exclusive_multiples 151 6 8 = 31 := by sorry

end NUMINAMATH_CALUDE_exclusive_multiples_of_6_or_8_less_than_151_l3067_306712


namespace NUMINAMATH_CALUDE_no_prime_sum_power_four_l3067_306703

theorem no_prime_sum_power_four (n : ℕ+) : ¬ Prime (4^(n : ℕ) + (n : ℕ)^4) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_power_four_l3067_306703


namespace NUMINAMATH_CALUDE_chocolate_count_l3067_306797

/-- The number of chocolates in each bag -/
def chocolates_per_bag : ℕ := 156

/-- The number of bags bought -/
def bags_bought : ℕ := 20

/-- The total number of chocolates -/
def total_chocolates : ℕ := chocolates_per_bag * bags_bought

theorem chocolate_count : total_chocolates = 3120 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l3067_306797


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3067_306722

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3067_306722


namespace NUMINAMATH_CALUDE_course_length_proof_l3067_306771

/-- Proves that the length of a course is 45 miles given the conditions of two cyclists --/
theorem course_length_proof (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 14)
  (h2 : speed2 = 16)
  (h3 : time = 1.5)
  : speed1 * time + speed2 * time = 45 := by
  sorry

#check course_length_proof

end NUMINAMATH_CALUDE_course_length_proof_l3067_306771


namespace NUMINAMATH_CALUDE_prob_all_blue_is_one_twelfth_l3067_306705

/-- The number of balls in the urn -/
def total_balls : ℕ := 10

/-- The number of blue balls in the urn -/
def blue_balls : ℕ := 5

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- Combination function -/
def C (n k : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The probability of drawing all blue balls -/
def prob_all_blue : ℚ := C blue_balls drawn_balls / C total_balls drawn_balls

theorem prob_all_blue_is_one_twelfth : 
  prob_all_blue = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_prob_all_blue_is_one_twelfth_l3067_306705


namespace NUMINAMATH_CALUDE_product_of_square_roots_l3067_306762

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p) * Real.sqrt (10 * p^3) * Real.sqrt (14 * p^5) = 10 * p^4 * Real.sqrt (21 * p) :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l3067_306762


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l3067_306733

/-- The time required for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9 * 5 / 18) -- Convert 9 kmph to m/s
  (h2 : train_speed = 45 * 5 / 18) -- Convert 45 kmph to m/s
  (h3 : train_length = 120)
  (h4 : initial_distance = 270) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 39 := by
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l3067_306733


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3067_306731

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 6) :
  (1 + 2 / (x + 1)) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3067_306731


namespace NUMINAMATH_CALUDE_x_value_l3067_306775

/-- The equation that defines x -/
def x_equation (x : ℝ) : Prop := x = Real.sqrt (2 + x)

/-- Theorem stating that the solution to the equation is 2 -/
theorem x_value : ∃ x : ℝ, x_equation x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_x_value_l3067_306775


namespace NUMINAMATH_CALUDE_total_weight_in_kg_l3067_306725

-- Define the weight of one envelope in grams
def envelope_weight : ℝ := 8.5

-- Define the number of envelopes
def num_envelopes : ℕ := 850

-- Define the conversion factor from grams to kilograms
def grams_to_kg : ℝ := 1000

-- Theorem statement
theorem total_weight_in_kg :
  (envelope_weight * num_envelopes) / grams_to_kg = 7.225 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_in_kg_l3067_306725


namespace NUMINAMATH_CALUDE_voice_area_greater_than_ground_area_l3067_306777

/-- The side length of the square ground in meters -/
def ground_side : ℝ := 25

/-- The maximum distance the trainer's voice can be heard in meters -/
def voice_range : ℝ := 140

/-- The area of the ground where the trainer's voice can be heard is greater than the area of the square ground -/
theorem voice_area_greater_than_ground_area : π * voice_range^2 > ground_side^2 := by
  sorry

end NUMINAMATH_CALUDE_voice_area_greater_than_ground_area_l3067_306777


namespace NUMINAMATH_CALUDE_parabola_max_sum_l3067_306779

/-- Given a parabola y = -x^2 - 3x + 3 and a point P(m, n) on this parabola,
    the maximum value of m + n is 4. -/
theorem parabola_max_sum (m n : ℝ) : 
  n = -m^2 - 3*m + 3 → (∀ x y : ℝ, y = -x^2 - 3*x + 3 → m + n ≥ x + y) → m + n = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_sum_l3067_306779


namespace NUMINAMATH_CALUDE_inequality_solution_l3067_306764

theorem inequality_solution (x : ℝ) : 
  x + |2*x + 3| ≥ 2 ↔ x ≤ -5 ∨ x ≥ -1/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3067_306764


namespace NUMINAMATH_CALUDE_projectile_height_time_l3067_306734

theorem projectile_height_time (t : ℝ) : 
  (∃ t₁ t₂ : ℝ, t₁ < t₂ ∧ -4.9 * t₁^2 + 30 * t₁ = 35 ∧ -4.9 * t₂^2 + 30 * t₂ = 35) → 
  (∀ t' : ℝ, -4.9 * t'^2 + 30 * t' = 35 → t' ≥ 10/7) ∧
  -4.9 * (10/7)^2 + 30 * (10/7) = 35 :=
sorry

end NUMINAMATH_CALUDE_projectile_height_time_l3067_306734


namespace NUMINAMATH_CALUDE_smallest_angle_CBD_l3067_306711

theorem smallest_angle_CBD (ABC : ℝ) (ABD : ℝ) (CBD : ℝ) 
  (h1 : ABC = 40)
  (h2 : ABD = 15)
  (h3 : CBD = ABC - ABD) :
  CBD = 25 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_CBD_l3067_306711


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l3067_306768

/-- Given two points M₁(x₁, y₁, z₁) and M₂(x₂, y₂, z₂), this theorem proves that the x-coordinate
    of the point P(x, 0, 0) on the Ox axis that is equidistant from M₁ and M₂ is given by
    x = (x₂² - x₁² + y₂² - y₁² + z₂² - z₁²) / (2(x₂ - x₁)) -/
theorem equidistant_point_on_x_axis 
  (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) 
  (h : x₁ ≠ x₂) : 
  ∃ x : ℝ, x = (x₂^2 - x₁^2 + y₂^2 - y₁^2 + z₂^2 - z₁^2) / (2 * (x₂ - x₁)) ∧ 
  (x - x₁)^2 + y₁^2 + z₁^2 = (x - x₂)^2 + y₂^2 + z₂^2 :=
by sorry

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l3067_306768


namespace NUMINAMATH_CALUDE_binomial_20_10_l3067_306770

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 31824) 
                        (h2 : Nat.choose 18 9 = 48620) 
                        (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 172822 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l3067_306770


namespace NUMINAMATH_CALUDE_prob_heads_tails_heads_l3067_306729

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of a sequence of independent events is the product of their individual probabilities -/
def prob_independent_events (p q r : ℝ) : ℝ := p * q * r

/-- The probability of getting heads, tails, then heads when flipping a fair coin three times is 1/8 -/
theorem prob_heads_tails_heads :
  ∀ (p : ℝ), fair_coin p →
  prob_independent_events p p p = 1/8 :=
sorry

end NUMINAMATH_CALUDE_prob_heads_tails_heads_l3067_306729


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_24_with_cube_root_between_9_and_9_1_l3067_306741

theorem exists_number_divisible_by_24_with_cube_root_between_9_and_9_1 :
  ∃ n : ℕ+,
    (∃ k : ℕ, n = 24 * k) ∧
    (9 < (n : ℝ) ^ (1/3 : ℝ)) ∧
    ((n : ℝ) ^ (1/3 : ℝ) < 9.1) ∧
    n = 744 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_24_with_cube_root_between_9_and_9_1_l3067_306741


namespace NUMINAMATH_CALUDE_range_of_a_l3067_306781

open Set

/-- The statement p: √(2x-1) ≤ 1 -/
def p (x : ℝ) : Prop := Real.sqrt (2 * x - 1) ≤ 1

/-- The statement q: (x-a)(x-(a+1)) ≤ 0 -/
def q (x a : ℝ) : Prop := (x - a) * (x - (a + 1)) ≤ 0

/-- The set of x satisfying statement p -/
def P : Set ℝ := {x | p x}

/-- The set of x satisfying statement q -/
def Q (a : ℝ) : Set ℝ := {x | q x a}

/-- p is a sufficient but not necessary condition for q -/
def sufficient_not_necessary (a : ℝ) : Prop := P ⊂ Q a ∧ P ≠ Q a

theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ∈ Icc 0 (1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3067_306781


namespace NUMINAMATH_CALUDE_fraction_equality_l3067_306713

theorem fraction_equality (a b : ℚ) (h : (a - b) / b = 3 / 7) : a / b = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3067_306713


namespace NUMINAMATH_CALUDE_computer_purchase_cost_l3067_306745

/-- Calculates the total cost of John's computer purchase --/
theorem computer_purchase_cost (computer_cost : ℝ) (base_video_card_cost : ℝ) 
  (monitor_foreign_cost : ℝ) (exchange_rate : ℝ) :
  computer_cost = 1500 →
  base_video_card_cost = 300 →
  monitor_foreign_cost = 200 →
  exchange_rate = 1.25 →
  ∃ total_cost : ℝ,
    total_cost = 
      (computer_cost + 
       (0.25 * computer_cost) + 
       (2.5 * base_video_card_cost * 0.88) + 
       ((0.25 * computer_cost) * 1.05) - 
       (0.07 * (computer_cost + (0.25 * computer_cost) + (2.5 * base_video_card_cost * 0.88))) + 
       (monitor_foreign_cost / exchange_rate)) ∧
    total_cost = 2536.30 := by
  sorry

end NUMINAMATH_CALUDE_computer_purchase_cost_l3067_306745


namespace NUMINAMATH_CALUDE_rachel_theorem_l3067_306758

def rachel_problem (initial_amount lunch_fraction dvd_fraction : ℚ) : Prop :=
  let lunch_expense := initial_amount * lunch_fraction
  let dvd_expense := initial_amount * dvd_fraction
  let remaining_amount := initial_amount - lunch_expense - dvd_expense
  remaining_amount = 50

theorem rachel_theorem :
  rachel_problem 200 (1/4) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_rachel_theorem_l3067_306758


namespace NUMINAMATH_CALUDE_stratified_sampling_total_l3067_306720

theorem stratified_sampling_total (senior junior freshman sampled_freshman : ℕ) 
  (h1 : senior = 1000)
  (h2 : junior = 1200)
  (h3 : freshman = 1500)
  (h4 : sampled_freshman = 75) :
  (senior + junior + freshman) * sampled_freshman / freshman = 185 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_total_l3067_306720


namespace NUMINAMATH_CALUDE_ellipse_intersection_l3067_306706

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-4)^2) + Real.sqrt ((x-6)^2 + y^2) = 10

-- Define the foci
def F1 : ℝ × ℝ := (0, 4)
def F2 : ℝ × ℝ := (6, 0)

-- Theorem statement
theorem ellipse_intersection :
  ∃ (x : ℝ), x ≠ 0 ∧ ellipse x 0 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_l3067_306706


namespace NUMINAMATH_CALUDE_parabola_directrix_a_value_l3067_306754

/-- A parabola with equation y² = ax and directrix x = 1 has a = -4 -/
theorem parabola_directrix_a_value :
  ∀ (a : ℝ),
  (∀ (x y : ℝ), y^2 = a*x → (∃ (p : ℝ), x = -p ∧ x = 1)) →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_a_value_l3067_306754


namespace NUMINAMATH_CALUDE_C₁_is_unit_circle_intersection_point_C₁_k4_equation_l3067_306763

-- Define the curves C₁ and C₂
def C₁ (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = Real.cos t ^ k ∧ p.2 = Real.sin t ^ k}

def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 - 16 * p.2 + 3 = 0}

-- Part 1: Prove that C₁ when k = 1 is a unit circle
theorem C₁_is_unit_circle :
  C₁ 1 = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} := by sorry

-- Part 2: Prove that (1/4, 1/4) is an intersection point of C₁ and C₂ when k = 4
theorem intersection_point :
  (1/4, 1/4) ∈ C₁ 4 ∧ (1/4, 1/4) ∈ C₂ := by sorry

-- Helper theorem: The equation of C₁ when k = 4 can be written as √x + √y = 1
theorem C₁_k4_equation (p : ℝ × ℝ) :
  p ∈ C₁ 4 ↔ Real.sqrt p.1 + Real.sqrt p.2 = 1 := by sorry

end NUMINAMATH_CALUDE_C₁_is_unit_circle_intersection_point_C₁_k4_equation_l3067_306763


namespace NUMINAMATH_CALUDE_abs_negative_two_thousand_l3067_306730

theorem abs_negative_two_thousand : |(-2000 : ℤ)| = 2000 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_thousand_l3067_306730


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3067_306782

theorem complex_equation_sum (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  Complex.mk a b = Complex.mk 1 1 * Complex.mk 2 (-1) →
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3067_306782


namespace NUMINAMATH_CALUDE_sixth_power_sum_l3067_306732

/-- Given real numbers a, b, x, and y satisfying certain conditions, 
    prove that ax^6 + by^6 = 1531.25 -/
theorem sixth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 12)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^6 + b * y^6 = 1531.25 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l3067_306732


namespace NUMINAMATH_CALUDE_x_squared_plus_7x_plus_12_bounds_l3067_306719

theorem x_squared_plus_7x_plus_12_bounds 
  (x : ℝ) (h : x^2 - 7*x + 12 < 0) : 
  48 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_7x_plus_12_bounds_l3067_306719


namespace NUMINAMATH_CALUDE_evelyn_found_caps_l3067_306708

/-- The number of bottle caps Evelyn started with -/
def starting_caps : ℕ := 18

/-- The number of bottle caps Evelyn ended up with -/
def total_caps : ℕ := 81

/-- The number of bottle caps Evelyn found -/
def found_caps : ℕ := total_caps - starting_caps

theorem evelyn_found_caps : found_caps = 63 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_found_caps_l3067_306708


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l3067_306769

theorem greatest_integer_fraction (x : ℤ) : (5 : ℚ) / 8 > (x : ℚ) / 15 ↔ x ≤ 9 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l3067_306769


namespace NUMINAMATH_CALUDE_rate_percent_calculation_l3067_306740

/-- Given that the simple interest on Rs. 25,000 amounts to Rs. 5,500 in 7 years,
    prove that the rate percent is equal to (5500 * 100) / (25000 * 7) -/
theorem rate_percent_calculation (principal : ℝ) (interest : ℝ) (time : ℝ) 
    (h1 : principal = 25000)
    (h2 : interest = 5500)
    (h3 : time = 7)
    (h4 : interest = principal * (rate_percent / 100) * time) :
  rate_percent = (interest * 100) / (principal * time) := by
  sorry

end NUMINAMATH_CALUDE_rate_percent_calculation_l3067_306740


namespace NUMINAMATH_CALUDE_jennifer_pears_l3067_306766

/-- Proves that Jennifer initially had 10 pears given the problem conditions -/
theorem jennifer_pears : ∃ P : ℕ, 
  (P + 20 + 2*P) - 6 = 44 ∧ P = 10 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_pears_l3067_306766


namespace NUMINAMATH_CALUDE_unique_solution_for_B_l3067_306757

theorem unique_solution_for_B : ∃! B : ℕ, ∃ A : ℕ, 
  (A < 10 ∧ B < 10) ∧ (100 * A + 78 - (20 * B + B) = 364) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_B_l3067_306757


namespace NUMINAMATH_CALUDE_wendy_makeup_time_l3067_306780

/-- Calculates the time spent on make-up given the number of facial products,
    waiting time between products, and total time for the full face routine. -/
def makeupTime (numProducts : ℕ) (waitingTime : ℕ) (totalTime : ℕ) : ℕ :=
  totalTime - (numProducts - 1) * waitingTime

/-- Proves that given 5 facial products, 5 minutes waiting time between each product,
    and a total of 55 minutes for the "full face," the time spent on make-up is 35 minutes. -/
theorem wendy_makeup_time :
  makeupTime 5 5 55 = 35 := by
  sorry

#eval makeupTime 5 5 55

end NUMINAMATH_CALUDE_wendy_makeup_time_l3067_306780


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3067_306761

theorem greatest_divisor_with_remainders : Nat.gcd (1442 - 12) (1816 - 6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3067_306761


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3067_306743

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 9}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3067_306743


namespace NUMINAMATH_CALUDE_range_of_m_l3067_306715

-- Define propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the set A (negation of q)
def A (m : ℝ) : Set ℝ := {x | x > 1 + m ∨ x < 1 - m}

-- Define the set B (negation of p)
def B : Set ℝ := {x | x > 10 ∨ x < -2}

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, x ∈ A m → x ∈ B) →
  (∃ x, x ∈ B ∧ x ∉ A m) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3067_306715


namespace NUMINAMATH_CALUDE_sugar_calculation_l3067_306704

theorem sugar_calculation (recipe_sugar : ℕ) (additional_sugar : ℕ) 
  (h1 : recipe_sugar = 7)
  (h2 : additional_sugar = 3) :
  recipe_sugar - additional_sugar = 4 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l3067_306704


namespace NUMINAMATH_CALUDE_gizmos_produced_75_workers_2_hours_l3067_306735

/-- Represents the production rates and worker information for a manufacturing plant. -/
structure ProductionData where
  gadget_rate : ℝ  -- Gadgets produced per worker per hour
  gizmo_rate : ℝ   -- Gizmos produced per worker per hour
  workers : ℕ      -- Number of workers
  hours : ℝ        -- Number of hours worked

/-- Calculates the number of gizmos produced given production data. -/
def gizmos_produced (data : ProductionData) : ℝ :=
  data.gizmo_rate * data.workers * data.hours

/-- States that the number of gizmos produced by 75 workers in 2 hours is 450. -/
theorem gizmos_produced_75_workers_2_hours :
  let data : ProductionData := {
    gadget_rate := 2,
    gizmo_rate := 3,
    workers := 75,
    hours := 2
  }
  gizmos_produced data = 450 := by sorry

end NUMINAMATH_CALUDE_gizmos_produced_75_workers_2_hours_l3067_306735


namespace NUMINAMATH_CALUDE_triangle_third_side_l3067_306721

theorem triangle_third_side (a b c : ℝ) : 
  a = 1 → b = 5 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l3067_306721


namespace NUMINAMATH_CALUDE_folding_theorem_l3067_306753

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a line segment -/
structure Segment where
  length : ℝ

/-- Represents the folding problem -/
def FoldingProblem (rect : Rectangle) : Prop :=
  ∃ (CC' EF : Segment),
    rect.width = 240 ∧
    rect.height = 288 ∧
    CC'.length = 312 ∧
    EF.length = 260

/-- The main theorem -/
theorem folding_theorem (rect : Rectangle) :
  FoldingProblem rect :=
sorry

end NUMINAMATH_CALUDE_folding_theorem_l3067_306753


namespace NUMINAMATH_CALUDE_parabola_focus_l3067_306765

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x = 4 * y^2

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : (ℝ × ℝ → Prop)) : Prop :=
  ∃ (a : ℝ), parabola = λ (x, y) => y^2 = a * (x - p.1) ∧ p.2 = 0

-- Theorem statement
theorem parabola_focus :
  focus (1/16, 0) (λ (x, y) => parabola_equation x y) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l3067_306765


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l3067_306702

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l3067_306702


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3067_306700

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3067_306700


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_l3067_306772

theorem x_squared_plus_inverse (x : ℝ) (h : 47 = x^6 + 1 / x^6) : x^2 + 1 / x^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_l3067_306772


namespace NUMINAMATH_CALUDE_tangent_line_proof_l3067_306750

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 1

/-- The proposed tangent line function -/
def g (x : ℝ) : ℝ := x + 1

theorem tangent_line_proof :
  (∃ x₀ : ℝ, f x₀ = g x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → f x < g x ∨ f x > g x)) ∧
  g (-1) = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l3067_306750


namespace NUMINAMATH_CALUDE_bus_driver_worked_54_hours_l3067_306794

/-- Represents the bus driver's compensation structure and work details for a week --/
structure BusDriverWeek where
  regularRate : ℝ
  overtimeRateMultiplier : ℝ
  regularHoursLimit : ℕ
  bonusPerPassenger : ℝ
  totalCompensation : ℝ
  passengersTransported : ℕ

/-- Calculates the total hours worked by the bus driver --/
def totalHoursWorked (week : BusDriverWeek) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, the bus driver worked 54 hours --/
theorem bus_driver_worked_54_hours :
  let week : BusDriverWeek := {
    regularRate := 14,
    overtimeRateMultiplier := 1.75,
    regularHoursLimit := 40,
    bonusPerPassenger := 0.25,
    totalCompensation := 998,
    passengersTransported := 350
  }
  totalHoursWorked week = 54 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_worked_54_hours_l3067_306794


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3067_306748

/-- Given that (z - 2i)(2 - i) = 5, prove that z = 2 + 3i -/
theorem complex_equation_solution (z : ℂ) (h : (z - 2*Complex.I)*(2 - Complex.I) = 5) :
  z = 2 + 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3067_306748


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l3067_306787

theorem not_all_perfect_squares (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(∃ x y z : ℕ, (2 * a^2 + b^2 + 3 = x^2) ∧ (2 * b^2 + c^2 + 3 = y^2) ∧ (2 * c^2 + a^2 + 3 = z^2)) :=
by sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l3067_306787


namespace NUMINAMATH_CALUDE_avery_egg_cartons_l3067_306710

/-- Calculates the number of full egg cartons given the number of chickens,
    eggs per chicken, and eggs per carton. -/
def full_egg_cartons (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / eggs_per_carton

/-- Proves that Avery can fill 10 egg cartons with the given conditions. -/
theorem avery_egg_cartons :
  full_egg_cartons 20 6 12 = 10 := by
  sorry

#eval full_egg_cartons 20 6 12

end NUMINAMATH_CALUDE_avery_egg_cartons_l3067_306710


namespace NUMINAMATH_CALUDE_expression_evaluation_l3067_306774

theorem expression_evaluation :
  1 - 1 / (1 + Real.sqrt (2 + Real.sqrt 3)) + 1 / (1 - Real.sqrt (2 - Real.sqrt 3)) =
  1 + (Real.sqrt (2 - Real.sqrt 3) + Real.sqrt (2 + Real.sqrt 3)) / (-1 - Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3067_306774


namespace NUMINAMATH_CALUDE_problem_statement_l3067_306707

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem problem_statement (a b : ℝ) 
  (h1 : 0 < a ∧ a < 1/2) 
  (h2 : 0 < b ∧ b < 1/2) 
  (h3 : f (1/a) + f (2/b) = 10) : 
  a + b/2 ≥ 2/7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3067_306707


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3067_306709

theorem expression_simplification_and_evaluation (a : ℝ) 
  (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4*a + 4) / (a + 1)) = -(a + 2) / (a - 2) ∧
  (-(1 + 2) / (1 - 2) = 3) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3067_306709


namespace NUMINAMATH_CALUDE_lottery_probabilities_l3067_306791

/-- Represents the probability of drawing a red ball from Box A -/
def prob_red_A : ℚ := 4 / 10

/-- Represents the probability of drawing a red ball from Box B -/
def prob_red_B : ℚ := 1 / 2

/-- Represents the probability of winning the first prize in one draw -/
def prob_first_prize : ℚ := prob_red_A * prob_red_B

/-- Represents the probability of winning the second prize in one draw -/
def prob_second_prize : ℚ := prob_red_A * (1 - prob_red_B) + (1 - prob_red_A) * prob_red_B

/-- Represents the probability of winning a prize in one draw -/
def prob_win_prize : ℚ := prob_first_prize + prob_second_prize

/-- Represents the number of independent lottery draws -/
def num_draws : ℕ := 3

theorem lottery_probabilities :
  (prob_win_prize = 7 / 10) ∧
  (1 - prob_first_prize ^ num_draws = 124 / 125) := by
  sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l3067_306791


namespace NUMINAMATH_CALUDE_spinner_direction_l3067_306789

-- Define the possible directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation : Rat := 7/4
  let counterclockwise_rotation : Rat := 5/2
  rotate initial_direction clockwise_rotation counterclockwise_rotation = Direction.East :=
by sorry

end NUMINAMATH_CALUDE_spinner_direction_l3067_306789


namespace NUMINAMATH_CALUDE_one_ton_equals_2200_pounds_l3067_306790

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 1760

/-- Represents the capacity of the gunny bag in tons -/
def bag_capacity : ℕ := 13

/-- Theorem stating that one ton equals 2200 pounds -/
theorem one_ton_equals_2200_pounds :
  (num_packets : ℚ) * packet_weight / (bag_capacity : ℚ) = 2200 := by
  sorry

end NUMINAMATH_CALUDE_one_ton_equals_2200_pounds_l3067_306790


namespace NUMINAMATH_CALUDE_rectangle_area_l3067_306783

theorem rectangle_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * (Real.sqrt (diagonal^2 - side^2)) → area = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3067_306783


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l3067_306728

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, x = 2 → (x - 2) * (x + 5) = 0) ∧
  (∃ x : ℝ, (x - 2) * (x + 5) = 0 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l3067_306728


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3067_306796

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : 4 * a - 2 * b + c > 0) 
  (h2 : a + b + c < 0) : 
  b^2 > a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3067_306796


namespace NUMINAMATH_CALUDE_binomial_20_4_l3067_306738

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_4_l3067_306738


namespace NUMINAMATH_CALUDE_intersection_length_range_l3067_306755

def interval_length (a b : ℝ) := b - a

theorem intersection_length_range :
  ∀ (a b : ℝ),
  (∀ x ∈ {x | a ≤ x ∧ x ≤ a+1}, -1 ≤ x ∧ x ≤ 1) →
  (∀ x ∈ {x | b-3/2 ≤ x ∧ x ≤ b}, -1 ≤ x ∧ x ≤ 1) →
  ∃ (l : ℝ), l = interval_length (max a (b-3/2)) (min (a+1) b) ∧
  1/2 ≤ l ∧ l ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_range_l3067_306755


namespace NUMINAMATH_CALUDE_total_turtles_received_l3067_306701

theorem total_turtles_received (martha_turtles : ℕ) (marion_extra_turtles : ℕ) : 
  martha_turtles = 40 → 
  marion_extra_turtles = 20 → 
  martha_turtles + (martha_turtles + marion_extra_turtles) = 100 := by
sorry

end NUMINAMATH_CALUDE_total_turtles_received_l3067_306701


namespace NUMINAMATH_CALUDE_inequality_proof_l3067_306747

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b / a > b - a / b) ∧ (1 / a + c < 1 / b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3067_306747


namespace NUMINAMATH_CALUDE_ratio_to_eight_l3067_306795

theorem ratio_to_eight : ∃ x : ℚ, (5 : ℚ) / 1 = x / 8 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_eight_l3067_306795


namespace NUMINAMATH_CALUDE_selection_theorem_l3067_306742

def number_of_lenovo : ℕ := 4
def number_of_crsc : ℕ := 5
def total_to_select : ℕ := 3

def ways_to_select : ℕ := 
  (number_of_lenovo.choose 1 * number_of_crsc.choose 2) +
  (number_of_lenovo.choose 2 * number_of_crsc.choose 1)

theorem selection_theorem : ways_to_select = 70 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l3067_306742


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l3067_306744

theorem reciprocal_of_negative_half : ((-1/2 : ℚ)⁻¹ : ℚ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l3067_306744


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_ellipse_standard_equation_l3067_306717

-- Problem 1: Hyperbola
def hyperbola_equation (e : ℝ) (vertex_distance : ℝ) : Prop :=
  e = 5/3 ∧ vertex_distance = 6 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/9 - y^2/16 = 1)

theorem hyperbola_standard_equation :
  hyperbola_equation (5/3) 6 :=
sorry

-- Problem 2: Ellipse
def ellipse_equation (major_minor_ratio : ℝ) (point : ℝ × ℝ) : Prop :=
  major_minor_ratio = 3 ∧ point = (3, 0) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/9 + y^2 = 1)) ∨
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, y^2/a^2 + x^2/b^2 = 1 ↔ y^2/81 + x^2/9 = 1))

theorem ellipse_standard_equation :
  ellipse_equation 3 (3, 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_ellipse_standard_equation_l3067_306717


namespace NUMINAMATH_CALUDE_mri_to_xray_ratio_l3067_306723

/-- The cost of an x-ray in dollars -/
def x_ray_cost : ℝ := 250

/-- The cost of an MRI as a multiple of the x-ray cost -/
def mri_cost (k : ℝ) : ℝ := k * x_ray_cost

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.8

/-- The amount Mike paid in dollars -/
def mike_payment : ℝ := 200

/-- The theorem stating the ratio of MRI cost to x-ray cost -/
theorem mri_to_xray_ratio :
  ∃ k : ℝ,
    (1 - insurance_coverage) * (x_ray_cost + mri_cost k) = mike_payment ∧
    k = 3 :=
sorry

end NUMINAMATH_CALUDE_mri_to_xray_ratio_l3067_306723


namespace NUMINAMATH_CALUDE_berry_farm_kept_fraction_l3067_306784

/-- Given a berry farm scenario, prove that half of the fresh berries need to be kept. -/
theorem berry_farm_kept_fraction (total_berries : ℕ) (rotten_fraction : ℚ) (berries_to_sell : ℕ) :
  total_berries = 60 →
  rotten_fraction = 1/3 →
  berries_to_sell = 20 →
  (total_berries - (rotten_fraction * total_berries).num - berries_to_sell) / 
  (total_berries - (rotten_fraction * total_berries).num) = 1/2 := by
  sorry

#check berry_farm_kept_fraction

end NUMINAMATH_CALUDE_berry_farm_kept_fraction_l3067_306784


namespace NUMINAMATH_CALUDE_floor_tile_equations_l3067_306776

/-- Represents the floor tile purchase scenario -/
structure FloorTilePurchase where
  x : ℕ  -- number of colored floor tiles
  y : ℕ  -- number of single-color floor tiles
  colored_cost : ℕ := 24  -- cost of colored tiles in yuan
  single_cost : ℕ := 12   -- cost of single-color tiles in yuan
  total_cost : ℕ := 2220  -- total cost in yuan

/-- The system of equations correctly represents the floor tile purchase scenario -/
theorem floor_tile_equations (purchase : FloorTilePurchase) : 
  (purchase.colored_cost * purchase.x + purchase.single_cost * purchase.y = purchase.total_cost) ∧
  (purchase.y = 2 * purchase.x - 15) := by
  sorry

end NUMINAMATH_CALUDE_floor_tile_equations_l3067_306776


namespace NUMINAMATH_CALUDE_kim_water_consumption_l3067_306798

/-- Proves that the total amount of water Kim drinks is 60 ounces -/
theorem kim_water_consumption (quart_to_ounce : ℚ) (bottle_volume : ℚ) (can_volume : ℚ) :
  quart_to_ounce = 32 →
  bottle_volume = 3/2 →
  can_volume = 12 →
  bottle_volume * quart_to_ounce + can_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_kim_water_consumption_l3067_306798


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l3067_306793

theorem right_triangle_ratio (a d : ℝ) (h_d_pos : d > 0) (h_d_odd : ∃ k : ℤ, d = 2 * k + 1) :
  (a + 4 * d)^2 = a^2 + (a + 2 * d)^2 → a / d = 1 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l3067_306793


namespace NUMINAMATH_CALUDE_weight_relationship_and_total_l3067_306737

/-- Given the weights of Haley, Verna, and Sherry, prove their relationship and total weight -/
theorem weight_relationship_and_total (haley verna sherry : ℕ) : 
  verna = haley + 17 →
  verna * 2 = sherry →
  haley = 103 →
  verna + sherry = 360 := by
sorry

end NUMINAMATH_CALUDE_weight_relationship_and_total_l3067_306737


namespace NUMINAMATH_CALUDE_unique_fraction_representation_l3067_306724

theorem unique_fraction_representation (p : ℕ) (hp : p > 2) (hprime : Nat.Prime p) :
  ∃! (x y : ℕ), x ≠ y ∧ (2 : ℚ) / p = 1 / x + 1 / y :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_representation_l3067_306724


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l3067_306785

/-- Given a parabola y² = 4x with focus F(1,0) and a point P on the parabola
    such that the distance PF = 5, the area of triangle PFO
    (where O is the origin) is 2. -/
theorem parabola_triangle_area :
  ∀ (P : ℝ × ℝ),
  (P.2)^2 = 4 * P.1 →  -- P is on the parabola y² = 4x
  (P.1 - 1)^2 + P.2^2 = 25 →  -- distance PF = 5
  (1/2) * P.1 * P.2 = 2 :=  -- area of triangle PFO
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l3067_306785


namespace NUMINAMATH_CALUDE_students_taking_no_subjects_l3067_306727

theorem students_taking_no_subjects (total : ℕ) (math physics chemistry : ℕ) 
  (math_physics math_chemistry physics_chemistry : ℕ) (all_three : ℕ) 
  (h1 : total = 60)
  (h2 : math = 40)
  (h3 : physics = 30)
  (h4 : chemistry = 25)
  (h5 : math_physics = 18)
  (h6 : physics_chemistry = 10)
  (h7 : math_chemistry = 12)
  (h8 : all_three = 5) :
  total - ((math + physics + chemistry) - (math_physics + physics_chemistry + math_chemistry) + all_three) = 5 := by
sorry


end NUMINAMATH_CALUDE_students_taking_no_subjects_l3067_306727


namespace NUMINAMATH_CALUDE_f_even_and_decreasing_l3067_306759

-- Define the function f(x) = -x² + 1
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_decreasing_l3067_306759


namespace NUMINAMATH_CALUDE_unique_sundaes_count_l3067_306799

/-- The number of flavors available -/
def n : ℕ := 8

/-- The number of flavors in each sundae -/
def k : ℕ := 2

/-- The number of unique two scoop sundaes -/
def unique_sundaes : ℕ := Nat.choose n k

theorem unique_sundaes_count : unique_sundaes = 28 := by
  sorry

end NUMINAMATH_CALUDE_unique_sundaes_count_l3067_306799


namespace NUMINAMATH_CALUDE_body_temperature_survey_most_suitable_for_census_l3067_306751

/-- Represents a survey option -/
inductive SurveyOption
| HeightSurvey
| TrafficRegulationsSurvey
| BodyTemperatureSurvey
| MovieViewershipSurvey

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  requiresCompleteData : Bool
  impactsSafety : Bool
  populationSize : Nat

/-- Defines what makes a survey suitable for a census -/
def suitableForCensus (c : SurveyCharacteristics) : Prop :=
  c.requiresCompleteData ∧ c.impactsSafety ∧ c.populationSize > 0

/-- Assigns characteristics to each survey option -/
def getSurveyCharacteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.HeightSurvey => ⟨false, false, 1000⟩
| SurveyOption.TrafficRegulationsSurvey => ⟨false, false, 10000⟩
| SurveyOption.BodyTemperatureSurvey => ⟨true, true, 500⟩
| SurveyOption.MovieViewershipSurvey => ⟨false, false, 2000⟩

theorem body_temperature_survey_most_suitable_for_census :
  suitableForCensus (getSurveyCharacteristics SurveyOption.BodyTemperatureSurvey) ∧
  ∀ (s : SurveyOption), s ≠ SurveyOption.BodyTemperatureSurvey →
    ¬(suitableForCensus (getSurveyCharacteristics s)) :=
  sorry

end NUMINAMATH_CALUDE_body_temperature_survey_most_suitable_for_census_l3067_306751
