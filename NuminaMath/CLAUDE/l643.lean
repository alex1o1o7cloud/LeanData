import Mathlib

namespace candidate_vote_percentage_l643_64361

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 285600) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 60 / 100 := by
  sorry

end candidate_vote_percentage_l643_64361


namespace point_symmetry_l643_64347

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- The point (3,4) -/
def point1 : ℝ × ℝ := (3, 4)

/-- The point (3,-4) -/
def point2 : ℝ × ℝ := (3, -4)

/-- Theorem stating that point1 and point2 are symmetric with respect to the x-axis -/
theorem point_symmetry : symmetric_wrt_x_axis point1 point2 := by sorry

end point_symmetry_l643_64347


namespace pyramid_sculpture_surface_area_l643_64309

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat

/-- Calculates the painted surface area of a cube sculpture -/
def painted_surface_area (sculpture : CubeSculpture) : Nat :=
  sorry

/-- The specific sculpture described in the problem -/
def pyramid_sculpture : CubeSculpture :=
  { top_layer := 1
  , middle_layer := 5
  , bottom_layer := 13 }

theorem pyramid_sculpture_surface_area :
  painted_surface_area pyramid_sculpture = 31 := by
  sorry

end pyramid_sculpture_surface_area_l643_64309


namespace danny_initial_caps_l643_64389

/-- The number of bottle caps Danny found at the park -/
def found_caps : ℕ := 7

/-- The total number of bottle caps Danny has after adding the found ones -/
def total_caps : ℕ := 32

/-- The number of bottle caps Danny had before finding the ones at the park -/
def initial_caps : ℕ := total_caps - found_caps

theorem danny_initial_caps : initial_caps = 25 := by
  sorry

end danny_initial_caps_l643_64389


namespace ngo_wage_problem_l643_64391

/-- The NGO wage problem -/
theorem ngo_wage_problem (illiterate_count : ℕ) (literate_count : ℕ) 
  (initial_illiterate_wage : ℚ) (average_decrease : ℚ) :
  illiterate_count = 20 →
  literate_count = 10 →
  initial_illiterate_wage = 25 →
  average_decrease = 10 →
  ∃ (new_illiterate_wage : ℚ),
    new_illiterate_wage = 10 ∧
    illiterate_count * (initial_illiterate_wage - new_illiterate_wage) = 
      (illiterate_count + literate_count) * average_decrease :=
by sorry

end ngo_wage_problem_l643_64391


namespace fraction_sum_zero_l643_64376

theorem fraction_sum_zero (a b c : ℤ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (avg : b = (a + c) / 2)
  (sum_zero : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end fraction_sum_zero_l643_64376


namespace total_chickens_and_ducks_prove_total_chickens_and_ducks_l643_64398

theorem total_chickens_and_ducks : ℕ → ℕ → ℕ → Prop :=
  fun (chickens ducks total : ℕ) =>
    chickens = 45 ∧ 
    chickens = ducks + 8 ∧ 
    total = chickens + ducks → 
    total = 82

-- Proof
theorem prove_total_chickens_and_ducks : 
  ∃ (chickens ducks total : ℕ), total_chickens_and_ducks chickens ducks total :=
by
  sorry

end total_chickens_and_ducks_prove_total_chickens_and_ducks_l643_64398


namespace range_of_function_l643_64386

theorem range_of_function (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 2) :
  0 < 2 * a + b ∧ 2 * a + b < 10 := by
  sorry

end range_of_function_l643_64386


namespace gcd_lcm_product_l643_64345

theorem gcd_lcm_product (a b : ℕ) (h : a = 90 ∧ b = 135) : 
  (Nat.gcd a b) * (Nat.lcm a b) = 12150 := by
  sorry

end gcd_lcm_product_l643_64345


namespace tangent_line_and_extreme_values_l643_64394

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 2 / x

theorem tangent_line_and_extreme_values (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = Real.log x - a * x + 2 / x) →
  (a = 1 → ∀ x y : ℝ, y = f 1 x → (x = 1 ∧ y = 1) → 2 * x + y - 3 = 0) ∧
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → f a x ≤ f a x₁ ∧ f a x ≤ f a x₂) ↔ 0 < a ∧ a < 1/8) :=
sorry

end tangent_line_and_extreme_values_l643_64394


namespace largest_two_decimal_rounding_to_five_l643_64306

-- Define a two-decimal number that rounds to 5.0
def is_valid_number (x : ℚ) : Prop :=
  (x ≥ 4.95) ∧ (x < 5.05) ∧ (∃ n : ℤ, x = n / 100)

-- Define the largest possible value
def largest_value : ℚ := 5.04

-- Theorem statement
theorem largest_two_decimal_rounding_to_five :
  ∀ x : ℚ, is_valid_number x → x ≤ largest_value :=
by sorry

end largest_two_decimal_rounding_to_five_l643_64306


namespace video_count_l643_64382

theorem video_count (video_length : ℝ) (lila_speed : ℝ) (roger_speed : ℝ) (total_time : ℝ) :
  video_length = 100 →
  lila_speed = 2 →
  roger_speed = 1 →
  total_time = 900 →
  ∃ n : ℕ, (n : ℝ) * (video_length / lila_speed + video_length / roger_speed) = total_time ∧ n = 6 := by
  sorry

end video_count_l643_64382


namespace exists_unstudied_planet_l643_64307

/-- Represents a planet in the solar system -/
structure Planet where
  id : ℕ

/-- Represents the solar system with its properties -/
structure SolarSystem where
  planets : Finset Planet
  distance : Planet → Planet → ℝ
  closest_planet : Planet → Planet
  odd_num_planets : Odd planets.card
  distinct_distances : ∀ p1 p2 p3 p4 : Planet, p1 ≠ p2 → p3 ≠ p4 → (p1, p2) ≠ (p3, p4) → distance p1 p2 ≠ distance p3 p4
  closest_is_closest : ∀ p1 p2 : Planet, p1 ≠ p2 → distance p1 (closest_planet p1) ≤ distance p1 p2
  not_self_study : ∀ p : Planet, closest_planet p ≠ p

/-- There exists a planet not being studied by any astronomer -/
theorem exists_unstudied_planet (s : SolarSystem) : 
  ∃ p : Planet, p ∈ s.planets ∧ ∀ q : Planet, q ∈ s.planets → s.closest_planet q ≠ p :=
sorry

end exists_unstudied_planet_l643_64307


namespace rectangle_area_l643_64323

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + 4 * W = 34) (h2 : 4 * L + 2 * W = 38) :
  L * W = 35 := by
sorry

end rectangle_area_l643_64323


namespace original_fraction_l643_64311

theorem original_fraction (n : ℚ) : 
  (n + 1) / (n + 6) = 7 / 12 → n / (n + 5) = 6 / 11 := by
  sorry

end original_fraction_l643_64311


namespace student_journal_pages_l643_64395

/-- Calculates the total number of journal pages written by a student over a given number of weeks. -/
def total_pages (sessions_per_week : ℕ) (pages_per_session : ℕ) (weeks : ℕ) : ℕ :=
  sessions_per_week * pages_per_session * weeks

/-- Theorem stating that given the specific conditions, a student writes 72 pages in 6 weeks. -/
theorem student_journal_pages :
  total_pages 3 4 6 = 72 := by
  sorry

end student_journal_pages_l643_64395


namespace identity_proof_l643_64369

theorem identity_proof (A B C A₁ B₁ C₁ : ℝ) :
  (A^2 + B^2 + C^2) * (A₁^2 + B₁^2 + C₁^2) - (A*A₁ + B*B₁ + C*C₁)^2 =
  (A*B₁ + A₁*B)^2 + (A*C₁ + A₁*C)^2 + (B*C₁ + B₁*C)^2 := by
  sorry

end identity_proof_l643_64369


namespace time_to_produce_one_item_l643_64362

/-- Given a machine that can produce 300 items in 2 hours, 
    prove that it takes 0.4 minutes to produce one item. -/
theorem time_to_produce_one_item 
  (total_time : ℝ) 
  (total_items : ℕ) 
  (h1 : total_time = 2) 
  (h2 : total_items = 300) : 
  (total_time / total_items) * 60 = 0.4 := by
  sorry

end time_to_produce_one_item_l643_64362


namespace mollys_age_l643_64380

/-- Given that the ratio of Sandy's age to Molly's age is 4:3,
    and Sandy will be 34 years old in 6 years,
    prove that Molly's current age is 21 years. -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 34 →
  molly_age = 21 := by
  sorry

end mollys_age_l643_64380


namespace divisible_by_seven_l643_64326

theorem divisible_by_seven (x y : ℕ+) (a b : ℕ) 
  (h1 : 3 * x.val + 4 * y.val = a ^ 2)
  (h2 : 4 * x.val + 3 * y.val = b ^ 2) : 
  7 ∣ x.val ∧ 7 ∣ y.val := by
sorry

end divisible_by_seven_l643_64326


namespace flag_arrangements_l643_64346

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def N : ℕ := 858

/-- The number of red flags -/
def red_flags : ℕ := 12

/-- The number of yellow flags -/
def yellow_flags : ℕ := 11

/-- The total number of flags -/
def total_flags : ℕ := red_flags + yellow_flags

/-- Theorem stating that N is the correct number of distinguishable arrangements -/
theorem flag_arrangements :
  N = (red_flags - 1) * (Nat.choose (red_flags + 1) yellow_flags) :=
by sorry

end flag_arrangements_l643_64346


namespace arithmetic_sequence_third_term_l643_64320

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first 5 terms of a sequence. -/
def SumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5

theorem arithmetic_sequence_third_term 
  (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SumFirstFive a = 20) : 
  a 3 = 4 := by
  sorry

end arithmetic_sequence_third_term_l643_64320


namespace sine_ratio_in_triangle_l643_64338

theorem sine_ratio_in_triangle (a b c : ℝ) (A B C : ℝ) :
  (b + c) / (c + a) = 4 / 5 ∧
  (c + a) / (a + b) = 5 / 6 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  Real.sin A / Real.sin B = 7 / 5 ∧
  Real.sin B / Real.sin C = 5 / 3 :=
by sorry

end sine_ratio_in_triangle_l643_64338


namespace min_value_inequality_l643_64332

theorem min_value_inequality (a : ℝ) (h : a > 1) :
  a + 2 / (a - 1) ≥ 1 + 2 * Real.sqrt 2 ∧
  ∃ a₀ > 1, a₀ + 2 / (a₀ - 1) = 1 + 2 * Real.sqrt 2 :=
by sorry

end min_value_inequality_l643_64332


namespace car_motorcycle_transaction_loss_l643_64334

theorem car_motorcycle_transaction_loss : 
  ∀ (car_cost motorcycle_cost : ℝ),
  car_cost * (1 - 0.25) = 16000 →
  motorcycle_cost * (1 + 0.25) = 16000 →
  car_cost + motorcycle_cost - 2 * 16000 = 2133.33 := by
sorry

end car_motorcycle_transaction_loss_l643_64334


namespace rational_function_sum_l643_64330

-- Define p(x) and q(x) as functions
variable (p q : ℝ → ℝ)

-- State the conditions
variable (h1 : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c)
variable (h2 : q 2 = 0 ∧ q 4 = 0)
variable (h3 : p 1 = 2)
variable (h4 : q 3 = 3)

-- State the theorem
theorem rational_function_sum :
  ∃ f : ℝ → ℝ, (∀ x, f x = p x + q x) ∧ (∀ x, f x = -3 * x^2 + 18 * x - 22) :=
sorry

end rational_function_sum_l643_64330


namespace number_comparison_l643_64397

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 2 to base 10 -/
def base2ToBase10 (n : ℕ) : ℕ := sorry

theorem number_comparison :
  let a : ℕ := 33
  let b : ℕ := base6ToBase10 52
  let c : ℕ := base2ToBase10 11111
  a > b ∧ b > c := by sorry

end number_comparison_l643_64397


namespace train_journey_properties_l643_64350

/-- Represents the properties of a train's journey -/
structure TrainJourney where
  duration : Real
  hourly_distance : Real

/-- Defines the concept of constant speed -/
def constant_speed (journey : TrainJourney) : Prop :=
  ∀ t : Real, 0 < t → t ≤ journey.duration → 
    (t * journey.hourly_distance) / t = journey.hourly_distance

/-- Calculates the total distance traveled -/
def total_distance (journey : TrainJourney) : Real :=
  journey.duration * journey.hourly_distance

/-- Main theorem about the train's journey -/
theorem train_journey_properties (journey : TrainJourney) 
  (h1 : journey.duration = 5.5)
  (h2 : journey.hourly_distance = 100) : 
  constant_speed journey ∧ total_distance journey = 550 := by
  sorry

#check train_journey_properties

end train_journey_properties_l643_64350


namespace fractional_method_experiments_l643_64336

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The number of experimental points -/
def num_points : ℕ := 12

/-- The maximum number of additional experiments needed -/
def max_additional_experiments : ℕ := 5

/-- Theorem: Given 12 experimental points and using the fractional method
    to find the optimal point of a unimodal function, the maximum number
    of additional experiments needed is 5. -/
theorem fractional_method_experiments :
  ∃ k : ℕ, num_points = fib (k + 1) - 1 ∧ max_additional_experiments = k :=
sorry

end fractional_method_experiments_l643_64336


namespace product_quality_comparison_l643_64341

structure MachineData where
  first_class : ℕ
  second_class : ℕ
  total : ℕ

def machine_a : MachineData := ⟨150, 50, 200⟩
def machine_b : MachineData := ⟨120, 80, 200⟩

def total_products : ℕ := 400

def k_squared (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_comparison :
  (machine_a.first_class : ℚ) / machine_a.total = 3/4 ∧
  (machine_b.first_class : ℚ) / machine_b.total = 3/5 ∧
  6.635 < k_squared total_products machine_a.first_class machine_a.second_class
    machine_b.first_class machine_b.second_class ∧
  k_squared total_products machine_a.first_class machine_a.second_class
    machine_b.first_class machine_b.second_class < 10.828 := by
  sorry

end product_quality_comparison_l643_64341


namespace remainder_sum_l643_64383

theorem remainder_sum (c d : ℤ) :
  (∃ p : ℤ, c = 84 * p + 76) →
  (∃ q : ℤ, d = 126 * q + 117) →
  (c + d) % 42 = 25 := by
  sorry

end remainder_sum_l643_64383


namespace cube_side_length_l643_64339

/-- Given a cube where the length of its space diagonal is 6.92820323027551 m,
    prove that the side length of the cube is 4 m. -/
theorem cube_side_length (d : ℝ) (h : d = 6.92820323027551) : 
  ∃ (a : ℝ), a * Real.sqrt 3 = d ∧ a = 4 := by
  sorry

end cube_side_length_l643_64339


namespace average_score_is_correct_rounded_average_score_l643_64324

/-- Represents the score distribution for a class test --/
structure ScoreDistribution where
  score_100 : Nat
  score_95  : Nat
  score_85  : Nat
  score_75  : Nat
  score_65  : Nat
  score_55  : Nat
  score_45  : Nat

/-- Calculates the average score given a score distribution --/
def calculateAverageScore (dist : ScoreDistribution) : Rat :=
  let totalStudents := dist.score_100 + dist.score_95 + dist.score_85 + 
                       dist.score_75 + dist.score_65 + dist.score_55 + dist.score_45
  let totalScore := 100 * dist.score_100 + 95 * dist.score_95 + 85 * dist.score_85 +
                    75 * dist.score_75 + 65 * dist.score_65 + 55 * dist.score_55 +
                    45 * dist.score_45
  totalScore / totalStudents

/-- The main theorem stating that the average score is approximately 76.3333 --/
theorem average_score_is_correct (dist : ScoreDistribution) 
  (h1 : dist.score_100 = 10)
  (h2 : dist.score_95 = 20)
  (h3 : dist.score_85 = 40)
  (h4 : dist.score_75 = 30)
  (h5 : dist.score_65 = 25)
  (h6 : dist.score_55 = 15)
  (h7 : dist.score_45 = 10) :
  calculateAverageScore dist = 11450 / 150 := by
  sorry

/-- The rounded average score is 76 --/
theorem rounded_average_score (dist : ScoreDistribution)
  (h : calculateAverageScore dist = 11450 / 150) :
  Int.floor (calculateAverageScore dist + 1/2) = 76 := by
  sorry

end average_score_is_correct_rounded_average_score_l643_64324


namespace unique_integer_product_digits_l643_64331

/-- ProductOfDigits calculates the product of digits for a given natural number -/
def ProductOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem states that 84 is the unique positive integer k such that 
    the product of its digits is equal to (11k/4) - 199 -/
theorem unique_integer_product_digits : 
  ∃! (k : ℕ), k > 0 ∧ ProductOfDigits k = (11 * k) / 4 - 199 := by sorry

end unique_integer_product_digits_l643_64331


namespace time_spent_on_activities_l643_64358

theorem time_spent_on_activities (hours_A hours_B : ℕ) : 
  hours_A = 6 → hours_A = hours_B + 3 → hours_B = 3 := by
  sorry

end time_spent_on_activities_l643_64358


namespace constraint_implies_equality_and_minimum_value_l643_64378

open Real

-- Define the constraint function
def constraint (a b c : ℝ) : Prop :=
  exp (a - c) + b * exp (c + 1) ≤ a + log b + 3

-- Define the objective function
def objective (a b c : ℝ) : ℝ :=
  a + b + 2 * c

-- Theorem statement
theorem constraint_implies_equality_and_minimum_value
  (a b c : ℝ) (h : constraint a b c) :
  a = c ∧ ∀ x y z, constraint x y z → objective a b c ≤ objective x y z ∧ objective a b c = -3 * log 3 :=
sorry

end constraint_implies_equality_and_minimum_value_l643_64378


namespace symmetric_reflection_theorem_l643_64359

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point across the xOy plane -/
def reflectXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Reflect a point across the z axis -/
def reflectZ (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

theorem symmetric_reflection_theorem :
  let P : Point3D := { x := 1, y := 1, z := 1 }
  let R₁ : Point3D := reflectXOY P
  let p₂ : Point3D := reflectZ R₁
  p₂ = { x := -1, y := -1, z := -1 } :=
by sorry

end symmetric_reflection_theorem_l643_64359


namespace units_digit_of_n_l643_64372

/-- Given two natural numbers m and n, returns true if m has a units digit of 3 -/
def has_units_digit_3 (m : ℕ) : Prop :=
  m % 10 = 3

/-- Given a natural number n, returns its units digit -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^5) (h2 : has_units_digit_3 m) :
  units_digit n = 7 := by
sorry

end units_digit_of_n_l643_64372


namespace sqrt_difference_equals_eight_thirds_sqrt_three_l643_64396

theorem sqrt_difference_equals_eight_thirds_sqrt_three :
  Real.sqrt 27 - Real.sqrt (1/3) = (8/3) * Real.sqrt 3 := by
  sorry

end sqrt_difference_equals_eight_thirds_sqrt_three_l643_64396


namespace purple_four_leaved_clovers_l643_64314

theorem purple_four_leaved_clovers (total_clovers : ℕ) (four_leaf_percentage : ℚ) (purple_fraction : ℚ) : 
  total_clovers = 500 →
  four_leaf_percentage = 1/5 →
  purple_fraction = 1/4 →
  (total_clovers : ℚ) * four_leaf_percentage * purple_fraction = 25 := by
sorry

end purple_four_leaved_clovers_l643_64314


namespace dad_borrowed_quarters_l643_64316

/-- The number of quarters borrowed by Sara's dad -/
def quarters_borrowed (initial_quarters current_quarters : ℕ) : ℕ :=
  initial_quarters - current_quarters

/-- Proof that Sara's dad borrowed 271 quarters -/
theorem dad_borrowed_quarters : quarters_borrowed 783 512 = 271 := by
  sorry

end dad_borrowed_quarters_l643_64316


namespace sequence_representation_l643_64368

def is_valid_sequence (q : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → q n < q m ∧ q n < 2 * n

theorem sequence_representation (q : ℕ → ℕ) (h : is_valid_sequence q) :
  ∀ m : ℕ, (∃ i : ℕ, q i = m) ∨ (∃ j k : ℕ, q j - q k = m) :=
by sorry

end sequence_representation_l643_64368


namespace quadratic_root_sum_l643_64388

theorem quadratic_root_sum (b c : ℝ) (h : c ≠ 0) : 
  (c^2 + 2*b*c - 5*c = 0) → (2*b + c = 5) := by
  sorry

end quadratic_root_sum_l643_64388


namespace decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l643_64317

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.47

/-- The fraction representation of the repeating decimal -/
def fraction : ℚ := 47 / 99

/-- Theorem stating that the repeating decimal equals the fraction -/
theorem decimal_equals_fraction : repeating_decimal = fraction := by sorry

/-- Theorem stating that the fraction is in lowest terms -/
theorem fraction_is_lowest_terms : 
  ∀ (a b : ℕ), a / b = fraction → b ≠ 0 → a.gcd b = 1 := by sorry

/-- The main theorem to prove -/
theorem sum_of_numerator_and_denominator : 
  ∃ (n d : ℕ), n / d = fraction ∧ n.gcd d = 1 ∧ n + d = 146 := by sorry

end decimal_equals_fraction_fraction_is_lowest_terms_sum_of_numerator_and_denominator_l643_64317


namespace f_max_value_l643_64377

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + Real.sin x) + Real.sin (x - Real.sin x) + (Real.pi / 2 - 2) * Real.sin (Real.sin x)

theorem f_max_value : 
  ∃ (M : ℝ), M = (Real.pi - 2) / Real.sqrt 2 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end f_max_value_l643_64377


namespace isosceles_right_triangle_relation_l643_64384

/-- In an isosceles right triangle ABC with right angle at A, 
    if CB = CA = h, BM + MA = 2(BC + CA), and MB = x, then x = 7h/5 -/
theorem isosceles_right_triangle_relation (h x : ℝ) : 
  h > 0 → 
  x > 0 → 
  x + Real.sqrt ((x + h)^2 + h^2) = 4 * h → 
  x = (7 * h) / 5 := by
sorry

end isosceles_right_triangle_relation_l643_64384


namespace abs_x_leq_2_necessary_not_sufficient_l643_64335

theorem abs_x_leq_2_necessary_not_sufficient :
  (∃ x : ℝ, |x + 1| ≤ 1 ∧ ¬(|x| ≤ 2)) = False ∧
  (∃ x : ℝ, |x| ≤ 2 ∧ ¬(|x + 1| ≤ 1)) = True :=
by sorry

end abs_x_leq_2_necessary_not_sufficient_l643_64335


namespace perfect_square_trinomial_l643_64385

theorem perfect_square_trinomial (m n : ℝ) :
  (4 / 9) * m^2 + (4 / 3) * m * n + n^2 = ((2 / 3) * m + n)^2 := by sorry

end perfect_square_trinomial_l643_64385


namespace largest_certain_divisor_l643_64313

def is_valid_selection (s : Finset Nat) : Prop :=
  s.card = 6 ∧ s ⊆ Finset.range 8

def Q (s : Finset Nat) : Nat :=
  s.prod id

theorem largest_certain_divisor :
  ∀ (s : Finset Nat), is_valid_selection s →
  (2 ∣ Q s) ∧ 
  ∀ (n : Nat), n > 2 → (∃ (t : Finset Nat), is_valid_selection t ∧ ¬(n ∣ Q t)) :=
by sorry

end largest_certain_divisor_l643_64313


namespace absolute_value_inequality_solution_set_l643_64337

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |3*x + 1| - |x - 1| < 0} = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end absolute_value_inequality_solution_set_l643_64337


namespace no_valid_triples_l643_64353

theorem no_valid_triples :
  ¬ ∃ (x y z : ℕ),
    (1 ≤ x) ∧ (x ≤ y) ∧ (y ≤ z) ∧
    (x * y * z + 2 * (x * y + y * z + z * x) = 2 * (2 * (x * y + y * z + z * x)) + 12) :=
by sorry


end no_valid_triples_l643_64353


namespace snake_diet_decade_l643_64370

/-- The number of mice a snake eats in a decade -/
def mice_eaten_in_decade (weeks_per_mouse : ℕ) (weeks_per_year : ℕ) (years_per_decade : ℕ) : ℕ :=
  (weeks_per_year / weeks_per_mouse) * years_per_decade

/-- Theorem: A snake eating one mouse every 4 weeks will eat 130 mice in a decade -/
theorem snake_diet_decade : 
  mice_eaten_in_decade 4 52 10 = 130 := by
  sorry

#eval mice_eaten_in_decade 4 52 10

end snake_diet_decade_l643_64370


namespace number_division_remainders_l643_64333

theorem number_division_remainders (N : ℤ) (h : N % 1554 = 131) : 
  (N % 37 = 20) ∧ (N % 73 = 58) := by
  sorry

end number_division_remainders_l643_64333


namespace greatest_power_of_two_l643_64390

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (12^603 - 8^402) ∧ 
   ∀ m : ℕ, 2^m ∣ (12^603 - 8^402) → m ≤ k) → 
  n = 1209 :=
sorry

end greatest_power_of_two_l643_64390


namespace race_length_l643_64300

/-- The race between Nicky and Cristina -/
def race (cristina_speed nicky_speed : ℝ) (head_start catch_up_time : ℝ) : Prop :=
  let nicky_distance := nicky_speed * catch_up_time
  let cristina_time := catch_up_time - head_start
  let cristina_distance := cristina_speed * cristina_time
  nicky_distance = cristina_distance ∧ nicky_distance = 90

/-- The race length is 90 meters -/
theorem race_length :
  race 5 3 12 30 :=
by
  sorry

end race_length_l643_64300


namespace average_difference_l643_64360

theorem average_difference (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 45)
  (avg_bc : (b + c) / 2 = 60) : 
  c - a = 30 := by
sorry

end average_difference_l643_64360


namespace quadratic_equation_solution_l643_64379

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) :
  let x := (a^2 - b^2) / (2*a)
  x^2 + b^2 = (a - x)^2 := by
  sorry

end quadratic_equation_solution_l643_64379


namespace calculate_expression_l643_64349

theorem calculate_expression : (8^5 / 8^2) * 3^6 = 373248 := by
  sorry

end calculate_expression_l643_64349


namespace solve_class_problem_l643_64328

def class_problem (num_girls : ℕ) (total_books : ℕ) (girls_books : ℕ) : Prop :=
  ∃ (num_boys : ℕ),
    num_boys = 10 ∧
    num_girls = 15 ∧
    total_books = 375 ∧
    girls_books = 225 ∧
    ∃ (books_per_student : ℕ),
      books_per_student * (num_girls + num_boys) = total_books ∧
      books_per_student * num_girls = girls_books

theorem solve_class_problem :
  class_problem 15 375 225 := by
  sorry

end solve_class_problem_l643_64328


namespace profit_percentage_proof_l643_64392

/-- Given that the cost price of 20 articles equals the selling price of 16 articles,
    prove that the profit percentage is 25%. -/
theorem profit_percentage_proof (C S : ℝ) (h : 20 * C = 16 * S) :
  (S - C) / C * 100 = 25 :=
sorry

end profit_percentage_proof_l643_64392


namespace third_degree_equation_roots_l643_64344

theorem third_degree_equation_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 8 * x^3 - 4 * x^2 - 4 * x - 1
  let root1 := Real.sin (π / 14)
  let root2 := Real.sin (5 * π / 14)
  let root3 := Real.sin (-3 * π / 14)
  (f root1 = 0) ∧ (f root2 = 0) ∧ (f root3 = 0) := by
  sorry

end third_degree_equation_roots_l643_64344


namespace finite_selector_existence_l643_64357

theorem finite_selector_existence
  (A B C : ℕ → Set ℕ)
  (h_finite : ∀ i, (A i).Finite ∧ (B i).Finite ∧ (C i).Finite)
  (h_disjoint : ∀ i, Disjoint (A i) (B i) ∧ Disjoint (A i) (C i) ∧ Disjoint (B i) (C i))
  (h_cover : ∀ X Y Z : Set ℕ, Disjoint X Y ∧ Disjoint X Z ∧ Disjoint Y Z → X ∪ Y ∪ Z = univ →
    ∃ i, A i ⊆ X ∧ B i ⊆ Y ∧ C i ⊆ Z) :
  ∃ S : Finset ℕ, ∀ X Y Z : Set ℕ, Disjoint X Y ∧ Disjoint X Z ∧ Disjoint Y Z → X ∪ Y ∪ Z = univ →
    ∃ i ∈ S, A i ⊆ X ∧ B i ⊆ Y ∧ C i ⊆ Z :=
by sorry

end finite_selector_existence_l643_64357


namespace joshua_bottles_count_l643_64305

theorem joshua_bottles_count (bottles_per_crate : ℕ) (num_crates : ℕ) (extra_bottles : ℕ) : 
  bottles_per_crate = 12 → 
  num_crates = 10 → 
  extra_bottles = 10 → 
  bottles_per_crate * num_crates + extra_bottles = 130 := by
sorry

end joshua_bottles_count_l643_64305


namespace coefficient_x_squared_in_f_prime_l643_64315

def f (x : ℝ) : ℝ := (1 - 2*x)^10

theorem coefficient_x_squared_in_f_prime : 
  ∃ (g : ℝ → ℝ), (∀ x, deriv f x = g x) ∧ 
  (∃ (a b c : ℝ), ∀ x, g x = a*x^2 + b*x + c) ∧
  (∃ (a b c : ℝ), (∀ x, g x = a*x^2 + b*x + c) ∧ a = -2880) :=
sorry

end coefficient_x_squared_in_f_prime_l643_64315


namespace percentage_red_cars_chennai_l643_64329

/-- Percentage of red cars in the total car population -/
def percentage_red_cars (total_cars : ℕ) (honda_cars : ℕ) (honda_red_ratio : ℚ) (non_honda_red_ratio : ℚ) : ℚ :=
  let non_honda_cars := total_cars - honda_cars
  let red_honda_cars := honda_red_ratio * honda_cars
  let red_non_honda_cars := non_honda_red_ratio * non_honda_cars
  let total_red_cars := red_honda_cars + red_non_honda_cars
  (total_red_cars / total_cars) * 100

/-- The percentage of red cars in Chennai -/
theorem percentage_red_cars_chennai :
  percentage_red_cars 900 500 (90/100) (225/1000) = 60 := by
  sorry

end percentage_red_cars_chennai_l643_64329


namespace smallest_sum_of_roots_l643_64366

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + 3*a*x + 4*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 4*b*x + 3*a = 0) :
  a + b ≥ 7/3 := by
  sorry

end smallest_sum_of_roots_l643_64366


namespace sandy_age_l643_64325

/-- Given that Molly is 20 years older than Sandy and their ages are in the ratio 7:9, prove that Sandy is 70 years old. -/
theorem sandy_age (sandy molly : ℕ) 
  (h1 : molly = sandy + 20) 
  (h2 : sandy * 9 = molly * 7) : 
  sandy = 70 := by sorry

end sandy_age_l643_64325


namespace image_of_4_neg2_l643_64302

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (xy, x+y) -/
def f : ℝ × ℝ → ℝ × ℝ := λ (x, y) => (x * y, x + y)

/-- The theorem stating that the image of (4, -2) under f is (-8, 2) -/
theorem image_of_4_neg2 : f (4, -2) = (-8, 2) := by sorry

end image_of_4_neg2_l643_64302


namespace round_201949_to_two_sig_figs_l643_64352

/-- Rounds a number to a specified number of significant figures in scientific notation -/
def roundToSignificantFigures (x : ℝ) (sigFigs : ℕ) : ℝ := sorry

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

theorem round_201949_to_two_sig_figs :
  let number : ℝ := 201949
  let rounded := roundToSignificantFigures number 2
  ∃ (sn : ScientificNotation), 
    sn.coefficient = 2.0 ∧ 
    sn.exponent = 5 ∧ 
    rounded = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end round_201949_to_two_sig_figs_l643_64352


namespace chess_team_girls_l643_64365

theorem chess_team_girls (total : ℕ) (boys girls : ℕ) 
  (h1 : total = boys + girls)
  (h2 : total = 26)
  (h3 : 3 * boys / 4 + girls / 4 = 13) : 
  girls = 13 := by
sorry

end chess_team_girls_l643_64365


namespace right_triangle_hypotenuse_l643_64374

theorem right_triangle_hypotenuse (area : ℝ) (leg : ℝ) (hypotenuse : ℝ) :
  area = 320 →
  leg = 16 →
  area = (1 / 2) * leg * (area / (1 / 2 * leg)) →
  hypotenuse^2 = leg^2 + (area / (1 / 2 * leg))^2 →
  hypotenuse = 4 * Real.sqrt 116 :=
by sorry

end right_triangle_hypotenuse_l643_64374


namespace largest_sum_largest_sum_proof_l643_64322

theorem largest_sum : ℝ → ℝ → ℝ → Prop :=
  fun A B C => 
    let A := 2010 / 2009 + 2010 / 2011
    let B := 2010 / 2011 + 2012 / 2011
    let C := 2011 / 2010 + 2011 / 2012 + 1 / 2011
    C > A ∧ C > B

-- The proof is omitted
theorem largest_sum_proof : largest_sum (2010 / 2009 + 2010 / 2011) (2010 / 2011 + 2012 / 2011) (2011 / 2010 + 2011 / 2012 + 1 / 2011) := by
  sorry

end largest_sum_largest_sum_proof_l643_64322


namespace square_plate_nails_l643_64321

/-- The number of nails on each side of the square -/
def nails_per_side : ℕ := 25

/-- The total number of unique nails used to fix the square plate -/
def total_nails : ℕ := nails_per_side * 4 - 4

theorem square_plate_nails :
  total_nails = 96 :=
by sorry

end square_plate_nails_l643_64321


namespace polyhedron_vertices_l643_64310

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for polyhedra states that V - E + F = 2, where V is the number of vertices,
    E is the number of edges, and F is the number of faces. -/
axiom eulers_formula (p : Polyhedron) : p.vertices - p.edges + p.faces = 2

/-- The theorem states that a polyhedron with 21 edges and 9 faces has 14 vertices. -/
theorem polyhedron_vertices (p : Polyhedron) (h1 : p.edges = 21) (h2 : p.faces = 9) : 
  p.vertices = 14 := by
  sorry

end polyhedron_vertices_l643_64310


namespace steves_oranges_l643_64319

/-- Steve's orange sharing problem -/
theorem steves_oranges (initial_oranges shared_oranges : ℕ) :
  initial_oranges = 46 →
  shared_oranges = 4 →
  initial_oranges - shared_oranges = 42 := by
  sorry

end steves_oranges_l643_64319


namespace cake_eating_contest_l643_64340

theorem cake_eating_contest : (7 : ℚ) / 8 - (5 : ℚ) / 6 = (1 : ℚ) / 24 := by
  sorry

end cake_eating_contest_l643_64340


namespace complex_expression_simplification_l643_64363

theorem complex_expression_simplification :
  (3/2)^0 - (1 - 0.5^(-2)) / ((27/8)^(2/3)) = 7/3 := by
  sorry

end complex_expression_simplification_l643_64363


namespace recurrence_relation_and_generating_function_l643_64367

def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

theorem recurrence_relation_and_generating_function :
  (∀ n : ℕ, a n - a (n + 1) + (1/3) * a (n + 2) - (1/27) * a (n + 3) = 0) ∧
  (∀ x : ℝ, abs x < 1/3 → ∑' (n : ℕ), a n * x^n = (1 - 3*x + 18*x^2) / (1 - 9*x + 27*x^2 - 27*x^3)) :=
by sorry

end recurrence_relation_and_generating_function_l643_64367


namespace eighteenth_term_of_equally_summed_sequence_l643_64371

/-- An Equally Summed Sequence is a sequence where the sum of each term and its subsequent term is always constant. -/
def EquallyStandardSequence (a : ℕ → ℝ) (c : ℝ) :=
  ∀ n, a n + a (n + 1) = c

theorem eighteenth_term_of_equally_summed_sequence
  (a : ℕ → ℝ)
  (h1 : EquallyStandardSequence a 5)
  (h2 : a 1 = 2) :
  a 18 = 3 := by
sorry

end eighteenth_term_of_equally_summed_sequence_l643_64371


namespace circle_area_ratio_l643_64373

theorem circle_area_ratio (r : ℝ) (h : r > 0) : (π * (3 * r)^2) / (π * r^2) = 9 := by
  sorry

end circle_area_ratio_l643_64373


namespace cauchy_schwarz_like_inequality_l643_64308

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) := by
  sorry

end cauchy_schwarz_like_inequality_l643_64308


namespace lenas_collage_glue_drops_l643_64381

/-- Calculates the total number of glue drops needed for a collage -/
def totalGlueDrops (clippings : List Nat) (gluePerClipping : Nat) : Nat :=
  (clippings.sum) * gluePerClipping

/-- Proves that the total number of glue drops for Lena's collage is 240 -/
theorem lenas_collage_glue_drops :
  let clippings := [4, 7, 5, 3, 5, 8, 2, 6]
  let gluePerClipping := 6
  totalGlueDrops clippings gluePerClipping = 240 := by
  sorry

#eval totalGlueDrops [4, 7, 5, 3, 5, 8, 2, 6] 6

end lenas_collage_glue_drops_l643_64381


namespace contrapositive_equivalence_l643_64387

theorem contrapositive_equivalence (M : Set α) (a b : α) :
  (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M) := by
  sorry

end contrapositive_equivalence_l643_64387


namespace first_player_wins_l643_64375

-- Define the chessboard as a type
def Chessboard : Type := Unit

-- Define a position on the chessboard
def Position : Type := Nat × Nat

-- Define a move as a function from one position to another
def Move : Type := Position → Position

-- Define the property of a move being valid
def ValidMove (m : Move) (visited : Set Position) : Prop :=
  ∀ p, p ∉ visited → 
    (m p).1 = p.1 ∧ ((m p).2 = p.2 + 1 ∨ (m p).2 = p.2 - 1) ∨
    (m p).2 = p.2 ∧ ((m p).1 = p.1 + 1 ∨ (m p).1 = p.1 - 1)

-- Define the game state
structure GameState :=
  (position : Position)
  (visited : Set Position)

-- Define the property of a player having a winning strategy
def HasWinningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState),
    ∃ (m : Move), ValidMove m state.visited →
      ¬∃ (m' : Move), ValidMove m' (insert (m state.position) state.visited)

-- Theorem statement
theorem first_player_wins :
  HasWinningStrategy 0 :=
sorry

end first_player_wins_l643_64375


namespace white_ball_estimate_l643_64354

/-- Represents the result of drawing balls from a bag -/
structure BagDrawResult where
  totalBalls : ℕ
  totalDraws : ℕ
  whiteDraws : ℕ

/-- Calculates the estimated number of white balls in the bag -/
def estimateWhiteBalls (result : BagDrawResult) : ℚ :=
  result.totalBalls * (result.whiteDraws : ℚ) / result.totalDraws

theorem white_ball_estimate (result : BagDrawResult) 
  (h1 : result.totalBalls = 20)
  (h2 : result.totalDraws = 100)
  (h3 : result.whiteDraws = 40) :
  estimateWhiteBalls result = 8 := by
  sorry

#eval estimateWhiteBalls { totalBalls := 20, totalDraws := 100, whiteDraws := 40 }

end white_ball_estimate_l643_64354


namespace fishing_line_length_l643_64318

/-- The original length of a fishing line can be calculated from its current length. -/
theorem fishing_line_length (current_length : ℝ) (h : current_length = 8.9) :
  (current_length + 3.1) * 3.1 * 2.1 = 78.12 := by
  sorry

#check fishing_line_length

end fishing_line_length_l643_64318


namespace sum_and_ratio_to_difference_l643_64327

theorem sum_and_ratio_to_difference (x y : ℝ) :
  x + y = 520 → x / y = 0.75 → y - x = 74 := by sorry

end sum_and_ratio_to_difference_l643_64327


namespace pigeonhole_principle_for_library_l643_64364

/-- The number of different types of books available. -/
def num_book_types : ℕ := 4

/-- The maximum number of books a student can borrow. -/
def max_books_per_student : ℕ := 3

/-- The type representing a borrowing pattern (number and types of books borrowed). -/
def BorrowingPattern := Fin num_book_types → Fin (max_books_per_student + 1)

/-- The minimum number of students required to guarantee a repeated borrowing pattern. -/
def min_students_for_repeat : ℕ := 15

theorem pigeonhole_principle_for_library :
  ∀ (students : Fin min_students_for_repeat → BorrowingPattern),
  ∃ (i j : Fin min_students_for_repeat), i ≠ j ∧ students i = students j :=
sorry

end pigeonhole_principle_for_library_l643_64364


namespace ratio_change_l643_64303

theorem ratio_change (x y : ℤ) (n : ℤ) : 
  y = 48 → x / y = 1 / 4 → (x + n) / y = 1 / 2 → n = 12 := by
  sorry

end ratio_change_l643_64303


namespace parabola_equation_correct_l643_64343

-- Define the parabola
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → ℝ

-- Define the equation of the parabola
def parabola_equation (p : Parabola) (x y : ℝ) : ℝ :=
  16 * x^2 + 25 * y^2 + 36 * x + 242 * y - 195

-- Theorem statement
theorem parabola_equation_correct (p : Parabola) :
  p.focus = (2, -1) ∧ p.directrix = (fun x y ↦ 5*x + 4*y - 20) →
  ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = ((5*x + 4*y - 20)^2) / 41 ↔
  parabola_equation p x y = 0 :=
sorry

end parabola_equation_correct_l643_64343


namespace inequality_and_equality_condition_l643_64304

theorem inequality_and_equality_condition (n : ℕ) (hn : n ≥ 1) :
  (1 / 3 : ℝ) * n^2 + (1 / 2 : ℝ) * n + (1 / 6 : ℝ) ≥ (n.factorial : ℝ)^((2 : ℝ) / n) ∧
  ((1 / 3 : ℝ) * n^2 + (1 / 2 : ℝ) * n + (1 / 6 : ℝ) = (n.factorial : ℝ)^((2 : ℝ) / n) ↔ n = 1) :=
by sorry

end inequality_and_equality_condition_l643_64304


namespace length_AB_l643_64342

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with slope 1 passing through the focus
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ line A.1 A.2 ∧ parabola B.1 B.2 ∧ line B.1 B.2

-- Theorem statement
theorem length_AB (A B : ℝ × ℝ) (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

end length_AB_l643_64342


namespace area_isosceles_right_triangle_l643_64356

/-- Given a right triangle ABC with AB = 12 and AC = 24, and points D on AC and E on BC
    forming an isosceles right triangle BDE, prove that the area of BDE is 80. -/
theorem area_isosceles_right_triangle (A B C D E : ℝ × ℝ) : 
  -- Right triangle ABC
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) →
  -- AB = 12
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 12 →
  -- AC = 24
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 24 →
  -- D is on AC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2)) →
  -- E is on BC
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (B.1 + s * (C.1 - B.1), B.2 + s * (C.2 - B.2)) →
  -- BDE is an isosceles right triangle
  (D.1 - B.1) * (E.1 - B.1) + (D.2 - B.2) * (E.2 - B.2) = 0 ∧
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = (E.1 - B.1)^2 + (E.2 - B.2)^2 →
  -- Area of BDE is 80
  (1/2) * Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) * Real.sqrt ((E.1 - B.1)^2 + (E.2 - B.2)^2) = 80 :=
by sorry


end area_isosceles_right_triangle_l643_64356


namespace budget_food_percentage_l643_64312

theorem budget_food_percentage (total_budget : ℝ) (accommodation_percent : ℝ) (entertainment_percent : ℝ) (coursework_materials : ℝ) :
  total_budget = 1000 →
  accommodation_percent = 15 →
  entertainment_percent = 25 →
  coursework_materials = 300 →
  (total_budget - (total_budget * accommodation_percent / 100 + total_budget * entertainment_percent / 100 + coursework_materials)) / total_budget * 100 = 30 := by
sorry

end budget_food_percentage_l643_64312


namespace william_shared_three_marbles_l643_64351

/-- The number of marbles William shared with Theresa -/
def marbles_shared (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

theorem william_shared_three_marbles :
  let initial := 10
  let remaining := 7
  marbles_shared initial remaining = 3 := by
  sorry

end william_shared_three_marbles_l643_64351


namespace income_change_percentage_l643_64399

theorem income_change_percentage 
  (original_payment : ℝ) 
  (original_time : ℝ) 
  (payment_increase_rate : ℝ) 
  (time_decrease_rate : ℝ) 
  (h1 : payment_increase_rate = 0.3333) 
  (h2 : time_decrease_rate = 0.3333) :
  let new_payment := original_payment * (1 + payment_increase_rate)
  let new_time := original_time * (1 - time_decrease_rate)
  let original_income := original_payment * original_time
  let new_income := new_payment * new_time
  (new_income - original_income) / original_income = -0.1111 := by
sorry

end income_change_percentage_l643_64399


namespace wednesday_profit_l643_64301

/-- The profit made by a beadshop over three days -/
def BeadshopProfit (total : ℝ) (monday : ℝ) (tuesday : ℝ) (wednesday : ℝ) : Prop :=
  total = 1200 ∧
  monday = (1/3) * total ∧
  tuesday = (1/4) * total ∧
  wednesday = total - monday - tuesday

/-- The profit made on Wednesday is $500 -/
theorem wednesday_profit (total monday tuesday wednesday : ℝ) :
  BeadshopProfit total monday tuesday wednesday →
  wednesday = 500 := by
  sorry

end wednesday_profit_l643_64301


namespace no_real_roots_for_nonzero_k_l643_64393

theorem no_real_roots_for_nonzero_k :
  ∀ k : ℝ, k ≠ 0 → ¬∃ x : ℝ, x^2 + k*x + 3*k^2 = 0 := by
sorry

end no_real_roots_for_nonzero_k_l643_64393


namespace stock_price_increase_l643_64355

/-- Proves that if a stock's price increases by 50% and closes at $15, then its opening price was $10. -/
theorem stock_price_increase (opening_price closing_price : ℝ) :
  closing_price = 15 ∧ closing_price = opening_price * 1.5 → opening_price = 10 := by
  sorry

end stock_price_increase_l643_64355


namespace bags_needed_is_17_l643_64348

/-- Calculates the number of bags of special dog food needed for a puppy's first year --/
def bags_needed : ℕ :=
  let days_in_year : ℕ := 365
  let ounces_per_pound : ℕ := 16
  let bag_size : ℕ := 5 -- in pounds
  let initial_period : ℕ := 60 -- in days
  let initial_daily_amount : ℕ := 2 -- in ounces
  let later_daily_amount : ℕ := 4 -- in ounces
  
  let initial_total : ℕ := initial_period * initial_daily_amount
  let later_period : ℕ := days_in_year - initial_period
  let later_total : ℕ := later_period * later_daily_amount
  
  let total_ounces : ℕ := initial_total + later_total
  let total_pounds : ℕ := (total_ounces + ounces_per_pound - 1) / ounces_per_pound
  (total_pounds + bag_size - 1) / bag_size

theorem bags_needed_is_17 : bags_needed = 17 := by
  sorry

end bags_needed_is_17_l643_64348
