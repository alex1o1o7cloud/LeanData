import Mathlib

namespace NUMINAMATH_CALUDE_no_solutions_for_exponential_equations_l3095_309586

theorem no_solutions_for_exponential_equations :
  (∀ n : ℕ, n > 1 → ¬∃ (p m : ℕ), Nat.Prime p ∧ Odd p ∧ m > 0 ∧ p^n + 1 = 2^m) ∧
  (∀ n : ℕ, n > 2 → ¬∃ (p m : ℕ), Nat.Prime p ∧ Odd p ∧ m > 0 ∧ p^n - 1 = 2^m) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_exponential_equations_l3095_309586


namespace NUMINAMATH_CALUDE_reciprocal_of_2024_l3095_309567

theorem reciprocal_of_2024 : (2024⁻¹ : ℚ) = 1 / 2024 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2024_l3095_309567


namespace NUMINAMATH_CALUDE_tank_capacity_correct_l3095_309521

/-- The capacity of a tank in litres. -/
def tank_capacity : ℝ := 1592

/-- The time in hours it takes for the leak to empty the full tank. -/
def leak_empty_time : ℝ := 7

/-- The rate at which the inlet pipe fills the tank in litres per minute. -/
def inlet_rate : ℝ := 6

/-- The time in hours it takes to empty the tank when both inlet and leak are open. -/
def combined_empty_time : ℝ := 12

/-- Theorem stating that the tank capacity is correct given the conditions. -/
theorem tank_capacity_correct : 
  tank_capacity = 
    (inlet_rate * 60 * combined_empty_time * leak_empty_time) / 
    (leak_empty_time - combined_empty_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_correct_l3095_309521


namespace NUMINAMATH_CALUDE_unfactorable_quadratic_l3095_309508

theorem unfactorable_quadratic : ¬ ∃ (a b c : ℝ), (∀ x : ℝ, x^2 - 10*x - 25 = (a*x + b)*(c*x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_unfactorable_quadratic_l3095_309508


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3095_309565

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 29) →
  (a 3 + a 6 + a 9 = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3095_309565


namespace NUMINAMATH_CALUDE_parallel_heaters_boiling_time_l3095_309500

/-- Given two heaters connected to the same direct current source,
    prove that the time to boil water when connected in parallel
    is (t₁ * t₂) / (t₁ + t₂), where t₁ and t₂ are the times taken
    by each heater individually. -/
theorem parallel_heaters_boiling_time
  (t₁ t₂ : ℝ)
  (h₁ : t₁ > 0)
  (h₂ : t₂ > 0)
  (boil_time : ℝ → ℝ → ℝ) :
  boil_time t₁ t₂ = t₁ * t₂ / (t₁ + t₂) :=
by sorry

end NUMINAMATH_CALUDE_parallel_heaters_boiling_time_l3095_309500


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3095_309558

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3095_309558


namespace NUMINAMATH_CALUDE_complex_repair_charge_correct_l3095_309507

/-- Calculates the charge for a complex bike repair given the following conditions:
  * Tire repair charge is $20
  * Tire repair cost is $5
  * Number of tire repairs in a month is 300
  * Number of complex repairs in a month is 2
  * Complex repair cost is $50
  * Retail profit is $2000
  * Fixed expenses are $4000
  * Total monthly profit is $3000
-/
def complex_repair_charge (
  tire_repair_charge : ℕ)
  (tire_repair_cost : ℕ)
  (num_tire_repairs : ℕ)
  (num_complex_repairs : ℕ)
  (complex_repair_cost : ℕ)
  (retail_profit : ℕ)
  (fixed_expenses : ℕ)
  (total_profit : ℕ) : ℕ :=
  let tire_repair_profit := (tire_repair_charge - tire_repair_cost) * num_tire_repairs
  let total_profit_before_complex := tire_repair_profit + retail_profit - fixed_expenses
  let complex_repair_total_profit := total_profit - total_profit_before_complex
  let complex_repair_profit := complex_repair_total_profit / num_complex_repairs
  complex_repair_profit + complex_repair_cost

theorem complex_repair_charge_correct : 
  complex_repair_charge 20 5 300 2 50 2000 4000 3000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_complex_repair_charge_correct_l3095_309507


namespace NUMINAMATH_CALUDE_z_value_proof_l3095_309564

theorem z_value_proof : 
  ∃ z : ℝ, ((2^5 : ℝ) * (9^2)) / (z * (3^5)) = 0.16666666666666666 → z = 64 := by
  sorry

end NUMINAMATH_CALUDE_z_value_proof_l3095_309564


namespace NUMINAMATH_CALUDE_horse_price_theorem_l3095_309589

/-- The sum of a geometric series with 32 terms, where the first term is 1
    and each subsequent term is twice the previous term, is 4294967295. -/
theorem horse_price_theorem :
  let n : ℕ := 32
  let a : ℕ := 1
  let r : ℕ := 2
  (a * (r^n - 1)) / (r - 1) = 4294967295 :=
by sorry

end NUMINAMATH_CALUDE_horse_price_theorem_l3095_309589


namespace NUMINAMATH_CALUDE_fourth_power_equality_l3095_309509

theorem fourth_power_equality (x : ℝ) : x^4 = (-3)^4 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_equality_l3095_309509


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3095_309540

/-- Represents the number of athletes selected in a stratified sample -/
structure StratifiedSample where
  totalMale : ℕ
  totalFemale : ℕ
  selectedMale : ℕ
  selectedFemale : ℕ

/-- Checks if the sample maintains the same ratio as the total population -/
def isProportionalSample (s : StratifiedSample) : Prop :=
  s.totalMale * s.selectedFemale = s.totalFemale * s.selectedMale

/-- Theorem: Given the conditions, the number of selected female athletes is 6 -/
theorem stratified_sample_theorem (s : StratifiedSample) :
  s.totalMale = 56 →
  s.totalFemale = 42 →
  s.selectedMale = 8 →
  isProportionalSample s →
  s.selectedFemale = 6 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l3095_309540


namespace NUMINAMATH_CALUDE_number_reading_and_approximation_l3095_309506

def number : ℕ := 60008205

def read_number (n : ℕ) : String := sorry

def approximate_to_ten_thousands (n : ℕ) : ℕ := sorry

theorem number_reading_and_approximation :
  (read_number number = "sixty million eight thousand two hundred and five") ∧
  (approximate_to_ten_thousands number = 6001) := by sorry

end NUMINAMATH_CALUDE_number_reading_and_approximation_l3095_309506


namespace NUMINAMATH_CALUDE_point_movement_l3095_309576

/-- Given a point P in a 2D Cartesian coordinate system, moving it right and down
    results in a new point Q with the expected coordinates. -/
theorem point_movement (P : ℝ × ℝ) (right down : ℝ) (Q : ℝ × ℝ) :
  P = (-1, 2) →
  right = 2 →
  down = 3 →
  Q.1 = P.1 + right →
  Q.2 = P.2 - down →
  Q = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l3095_309576


namespace NUMINAMATH_CALUDE_zero_location_l3095_309562

theorem zero_location (x y : ℝ) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) :
  x^5 < 0 ∧ 0 < y^8 := by
  sorry

end NUMINAMATH_CALUDE_zero_location_l3095_309562


namespace NUMINAMATH_CALUDE_power_of_negative_square_l3095_309568

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l3095_309568


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3095_309550

theorem fraction_subtraction (m : ℝ) (h : m ≠ 1) : m / (1 - m) - 1 / (1 - m) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3095_309550


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3095_309584

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that a + b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a + b = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3095_309584


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l3095_309583

theorem parabola_x_intercepts :
  let f (x : ℝ) := -3 * x^2 + 4 * x - 1
  (∃ a b : ℝ, a ≠ b ∧ f a = 0 ∧ f b = 0) ∧
  (∀ x y z : ℝ, f x = 0 → f y = 0 → f z = 0 → x = y ∨ x = z ∨ y = z) := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l3095_309583


namespace NUMINAMATH_CALUDE_shoe_refund_percentage_l3095_309528

/-- Given Will's shopping scenario, prove the percentage of shoe price refunded --/
theorem shoe_refund_percentage 
  (initial_amount : ℝ) 
  (sweater_cost : ℝ) 
  (tshirt_cost : ℝ) 
  (shoe_cost : ℝ) 
  (final_amount : ℝ) 
  (h1 : initial_amount = 74) 
  (h2 : sweater_cost = 9) 
  (h3 : tshirt_cost = 11) 
  (h4 : shoe_cost = 30) 
  (h5 : final_amount = 51) : 
  (final_amount - (initial_amount - (sweater_cost + tshirt_cost + shoe_cost))) / shoe_cost * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_shoe_refund_percentage_l3095_309528


namespace NUMINAMATH_CALUDE_pencil_cost_2500_l3095_309570

/-- The cost of buying a certain number of pencils with a discount applied after a threshold -/
def pencil_cost (box_size : ℕ) (box_cost : ℚ) (total_pencils : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let unit_cost := box_cost / box_size
  let regular_cost := min total_pencils discount_threshold * unit_cost
  let discounted_pencils := max (total_pencils - discount_threshold) 0
  let discounted_cost := discounted_pencils * (unit_cost * (1 - discount_rate))
  regular_cost + discounted_cost

theorem pencil_cost_2500 :
  pencil_cost 200 50 2500 1000 (1/10) = 587.5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_2500_l3095_309570


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l3095_309594

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → 
  ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l3095_309594


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_l3095_309503

theorem absolute_value_of_negative (a : ℝ) : a < 0 → |a| = -a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_l3095_309503


namespace NUMINAMATH_CALUDE_wendy_album_problem_l3095_309532

/-- Given a total number of pictures and the number of pictures in each of 5 albums,
    calculate the number of pictures in the first album. -/
def pictures_in_first_album (total : ℕ) (pics_per_album : ℕ) : ℕ :=
  total - 5 * pics_per_album

/-- Theorem stating that given 79 total pictures and 7 pictures in each of 5 albums,
    the number of pictures in the first album is 44. -/
theorem wendy_album_problem :
  pictures_in_first_album 79 7 = 44 := by
  sorry

end NUMINAMATH_CALUDE_wendy_album_problem_l3095_309532


namespace NUMINAMATH_CALUDE_bus_distance_l3095_309563

theorem bus_distance (total_distance : ℝ) (plane_fraction : ℝ) (train_bus_ratio : ℝ)
  (h1 : total_distance = 900)
  (h2 : plane_fraction = 1 / 3)
  (h3 : train_bus_ratio = 2 / 3) :
  let plane_distance := total_distance * plane_fraction
  let bus_distance := (total_distance - plane_distance) / (1 + train_bus_ratio)
  bus_distance = 360 := by
sorry

end NUMINAMATH_CALUDE_bus_distance_l3095_309563


namespace NUMINAMATH_CALUDE_sqrt_inequality_and_trig_identity_l3095_309502

theorem sqrt_inequality_and_trig_identity :
  (∀ (α : Real),
    Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 5 - Real.sqrt 3 ∧
    Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_and_trig_identity_l3095_309502


namespace NUMINAMATH_CALUDE_sector_angle_l3095_309590

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, 
    prove that its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (θ : ℝ) 
  (h_area : (1/2) * θ * r^2 = 1)
  (h_perim : 2*r + θ*r = 4) : 
  θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3095_309590


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_values_l3095_309542

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℕ
  d : ℕ
  first_term : a 1 = 1
  nth_term : ∀ n : ℕ, n ≥ 3 → a n = 70
  common_diff : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the possible values of n -/
theorem arithmetic_sequence_n_values (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≥ 3 ∧ seq.a n = 70 → n = 4 ∨ n = 24 ∨ n = 70 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_values_l3095_309542


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3095_309577

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : a 3 = 3/2) 
  (h2 : S 3 = 9/2) : 
  ∃ q : ℚ, (q = 1 ∨ q = -1/2) ∧ 
    (∀ n : ℕ, n ≥ 1 → a n = a 1 * q^(n-1)) ∧
    (∀ n : ℕ, n ≥ 1 → S n = a 1 * (1 - q^n) / (1 - q)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3095_309577


namespace NUMINAMATH_CALUDE_evaluate_expression_l3095_309535

theorem evaluate_expression (a b : ℕ+) (h : 2^(a:ℕ) * 3^(b:ℕ) = 324) : 
  2^(b:ℕ) * 3^(a:ℕ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3095_309535


namespace NUMINAMATH_CALUDE_mixed_committee_probability_l3095_309592

/-- The number of members in the Book club -/
def total_members : ℕ := 24

/-- The number of boys in the Book club -/
def num_boys : ℕ := 12

/-- The number of girls in the Book club -/
def num_girls : ℕ := 12

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- The probability of choosing a committee with at least one boy and one girl -/
def probability_mixed_committee : ℚ := 171 / 177

theorem mixed_committee_probability :
  (1 : ℚ) - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size : ℚ) / 
  (Nat.choose total_members committee_size : ℚ) = probability_mixed_committee := by
  sorry

end NUMINAMATH_CALUDE_mixed_committee_probability_l3095_309592


namespace NUMINAMATH_CALUDE_expression_evaluation_l3095_309519

theorem expression_evaluation (a b : ℝ) (h1 : a = 1) (h2 : b = -3) :
  (a - b)^2 - 2*a*(a + 3*b) + (a + 2*b)*(a - 2*b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3095_309519


namespace NUMINAMATH_CALUDE_jill_walking_time_l3095_309571

/-- The time it takes Jill to walk to school given Dave's and Jill's walking parameters -/
theorem jill_walking_time (dave_steps_per_min : ℕ) (dave_step_length : ℕ) (dave_time : ℕ)
  (jill_steps_per_min : ℕ) (jill_step_length : ℕ) 
  (h1 : dave_steps_per_min = 80) (h2 : dave_step_length = 65) (h3 : dave_time = 20)
  (h4 : jill_steps_per_min = 120) (h5 : jill_step_length = 50) :
  (dave_steps_per_min * dave_step_length * dave_time : ℚ) / (jill_steps_per_min * jill_step_length) = 52/3 :=
by sorry

end NUMINAMATH_CALUDE_jill_walking_time_l3095_309571


namespace NUMINAMATH_CALUDE_different_subjects_count_l3095_309523

/-- The number of ways to choose 2 books from different subjects -/
def choose_different_subjects (chinese_books math_books english_books : ℕ) : ℕ :=
  chinese_books * math_books + chinese_books * english_books + math_books * english_books

/-- Theorem stating that there are 143 ways to choose 2 books from different subjects -/
theorem different_subjects_count :
  choose_different_subjects 9 7 5 = 143 := by
  sorry

end NUMINAMATH_CALUDE_different_subjects_count_l3095_309523


namespace NUMINAMATH_CALUDE_average_height_is_64_inches_l3095_309539

/-- Given the heights of Parker, Daisy, and Reese, prove their average height is 64 inches. -/
theorem average_height_is_64_inches 
  (reese_height : ℕ)
  (daisy_height : ℕ)
  (parker_height : ℕ)
  (h1 : reese_height = 60)
  (h2 : daisy_height = reese_height + 8)
  (h3 : parker_height = daisy_height - 4) :
  (reese_height + daisy_height + parker_height) / 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_average_height_is_64_inches_l3095_309539


namespace NUMINAMATH_CALUDE_base6_calculation_l3095_309554

/-- Represents a number in base 6 --/
def Base6 : Type := ℕ

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : Base6 := sorry

/-- Adds two numbers in base 6 --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- Subtracts two numbers in base 6 --/
def subBase6 (a b : Base6) : Base6 := sorry

/-- Theorem: 15₆ - 4₆ + 20₆ = 31₆ in base 6 --/
theorem base6_calculation : 
  let a := toBase6 15
  let b := toBase6 4
  let c := toBase6 20
  let d := toBase6 31
  addBase6 (subBase6 a b) c = d := by sorry

end NUMINAMATH_CALUDE_base6_calculation_l3095_309554


namespace NUMINAMATH_CALUDE_food_bank_donation_ratio_l3095_309556

/-- Proves the ratio of food donations in the second week to the first week -/
theorem food_bank_donation_ratio :
  let first_week_donation : ℝ := 40
  let second_week_multiple : ℝ := x
  let total_donation : ℝ := first_week_donation + first_week_donation * second_week_multiple
  let remaining_percentage : ℝ := 0.3
  let remaining_food : ℝ := 36
  remaining_percentage * total_donation = remaining_food →
  second_week_multiple = 2 := by
  sorry

end NUMINAMATH_CALUDE_food_bank_donation_ratio_l3095_309556


namespace NUMINAMATH_CALUDE_fraction_simplification_l3095_309566

theorem fraction_simplification (x y z : ℚ) :
  x = 3 ∧ y = 4 ∧ z = 2 →
  (10 * x * y^3) / (15 * x^2 * y * z) = 16 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3095_309566


namespace NUMINAMATH_CALUDE_range_of_f_l3095_309511

def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

theorem range_of_f :
  Set.range f = Set.Icc (-8 : ℝ) 8 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3095_309511


namespace NUMINAMATH_CALUDE_tom_seashell_collection_l3095_309548

/-- The number of days Tom spent at the beach -/
def days_at_beach : ℕ := 5

/-- The number of seashells Tom found each day -/
def seashells_per_day : ℕ := 7

/-- The total number of seashells Tom found during his beach trip -/
def total_seashells : ℕ := days_at_beach * seashells_per_day

theorem tom_seashell_collection :
  total_seashells = 35 :=
by sorry

end NUMINAMATH_CALUDE_tom_seashell_collection_l3095_309548


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_six_l3095_309599

theorem factorization_a_squared_minus_six (a : ℝ) :
  a^2 - 6 = (a + Real.sqrt 6) * (a - Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_six_l3095_309599


namespace NUMINAMATH_CALUDE_birth_death_rate_interval_l3095_309574

/-- Prove that the time interval for birth and death rates is 2 seconds given the conditions --/
theorem birth_death_rate_interval (birth_rate death_rate net_increase_per_day : ℕ) 
  (h1 : birth_rate = 6)
  (h2 : death_rate = 2)
  (h3 : net_increase_per_day = 172800) :
  (24 * 60 * 60) / ((birth_rate - death_rate) * net_increase_per_day) = 2 := by
  sorry


end NUMINAMATH_CALUDE_birth_death_rate_interval_l3095_309574


namespace NUMINAMATH_CALUDE_binge_watching_duration_l3095_309552

/-- Proves that given a TV show with 90 episodes of 20 minutes each, and a viewing time of 2 hours per day, it will take 15 days to finish watching the entire show. -/
theorem binge_watching_duration (num_episodes : ℕ) (episode_length : ℕ) (daily_viewing_time : ℕ) : 
  num_episodes = 90 → 
  episode_length = 20 → 
  daily_viewing_time = 120 → 
  (num_episodes * episode_length) / daily_viewing_time = 15 := by
  sorry

#check binge_watching_duration

end NUMINAMATH_CALUDE_binge_watching_duration_l3095_309552


namespace NUMINAMATH_CALUDE_high_card_value_l3095_309531

structure CardGame where
  total_cards : Nat
  high_cards : Nat
  low_cards : Nat
  high_value : Nat
  low_value : Nat
  target_points : Nat
  target_low_cards : Nat
  ways_to_earn : Nat

def is_valid_game (game : CardGame) : Prop :=
  game.total_cards = 52 ∧
  game.high_cards = game.low_cards ∧
  game.high_cards + game.low_cards = game.total_cards ∧
  game.low_value = 1 ∧
  game.target_points = 5 ∧
  game.target_low_cards = 3 ∧
  game.ways_to_earn = 4

theorem high_card_value (game : CardGame) :
  is_valid_game game → game.high_value = 2 := by
  sorry

end NUMINAMATH_CALUDE_high_card_value_l3095_309531


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3095_309551

/-- Given two vectors a and b in ℝ², if they are parallel and a = (3, 4) and b = (x, 1/2), then x = 3/8 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (3, 4) →
  b = (x, 1/2) →
  ∃ (k : ℝ), a = k • b →
  x = 3/8 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3095_309551


namespace NUMINAMATH_CALUDE_fibonacci_special_sequence_l3095_309525

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_special_sequence (a b c : ℕ) :
  (fib c = 2 * fib b - fib a) →
  (fib c - fib a = fib a) →
  (a + c = 1700) →
  a = 849 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_special_sequence_l3095_309525


namespace NUMINAMATH_CALUDE_point_on_line_value_l3095_309544

theorem point_on_line_value (x y : ℝ) (h1 : y = x + 2) (h2 : 1 < y) (h3 : y < 3) :
  Real.sqrt (y^2 - 8*x) + Real.sqrt (y^2 + 2*x + 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_value_l3095_309544


namespace NUMINAMATH_CALUDE_distributor_cost_l3095_309518

theorem distributor_cost (commission_rate : Real) (profit_rate : Real) (observed_price : Real) :
  commission_rate = 0.20 →
  profit_rate = 0.20 →
  observed_price = 30 →
  ∃ (cost : Real),
    cost = 31.25 ∧
    observed_price = (1 - commission_rate) * (cost * (1 + profit_rate)) :=
by sorry

end NUMINAMATH_CALUDE_distributor_cost_l3095_309518


namespace NUMINAMATH_CALUDE_red_white_jelly_beans_in_fishbowl_l3095_309524

/-- The number of red jelly beans in one bag -/
def red_in_bag : ℕ := 24

/-- The number of white jelly beans in one bag -/
def white_in_bag : ℕ := 18

/-- The number of bags needed to fill the fishbowl -/
def bags_to_fill : ℕ := 3

/-- The total number of red and white jelly beans in the fishbowl -/
def total_red_white_in_bowl : ℕ := (red_in_bag + white_in_bag) * bags_to_fill

theorem red_white_jelly_beans_in_fishbowl :
  total_red_white_in_bowl = 126 :=
by sorry

end NUMINAMATH_CALUDE_red_white_jelly_beans_in_fishbowl_l3095_309524


namespace NUMINAMATH_CALUDE_min_value_theorem_l3095_309513

/-- Given positive real numbers a and b, this theorem proves that the minimum value of
    (a + 2/b)(a + 2/b - 1010) + (b + 2/a)(b + 2/a - 1010) + 101010 is -404040. -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (x + 2/y) * (x + 2/y - 1010) + (y + 2/x) * (y + 2/x - 1010) + 101010 <
  (a + 2/b) * (a + 2/b - 1010) + (b + 2/a) * (b + 2/a - 1010) + 101010) →
  (a + 2/b) * (a + 2/b - 1010) + (b + 2/a) * (b + 2/a - 1010) + 101010 ≥ -404040 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3095_309513


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3095_309561

/-- The perimeter of a semi-circle with radius 2.1 cm is π * 2.1 + 4.2 cm. -/
theorem semicircle_perimeter :
  let r : ℝ := 2.1
  (π * r + 2 * r) = π * 2.1 + 4.2 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3095_309561


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3095_309510

theorem complex_expression_simplification (x y : ℝ) :
  let i : ℂ := Complex.I
  (x^2 + i*y)^3 * (x^2 - i*y)^3 = x^12 - 9*x^8*y^2 - 9*x^4*y^4 - y^6 :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3095_309510


namespace NUMINAMATH_CALUDE_retail_price_is_1_04a_l3095_309598

/-- The retail price of a washing machine after markup and discount -/
def retail_price (a : ℝ) : ℝ :=
  a * (1 + 0.3) * (1 - 0.2)

/-- Theorem stating that the retail price is 1.04 times the initial cost -/
theorem retail_price_is_1_04a (a : ℝ) : retail_price a = 1.04 * a := by
  sorry

end NUMINAMATH_CALUDE_retail_price_is_1_04a_l3095_309598


namespace NUMINAMATH_CALUDE_triangle_abc_right_angle_l3095_309545

theorem triangle_abc_right_angle (A B C : ℝ) (h1 : A = 30) (h2 : B = 60) : C = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_right_angle_l3095_309545


namespace NUMINAMATH_CALUDE_class_payment_problem_l3095_309505

theorem class_payment_problem (total_students : ℕ) (full_payers : ℕ) (half_payers : ℕ) (total_amount : ℕ) :
  total_students = 25 →
  full_payers = 21 →
  half_payers = 4 →
  total_amount = 1150 →
  ∃ (full_payment : ℕ), 
    full_payment * full_payers + (full_payment / 2) * half_payers = total_amount ∧
    full_payment = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_class_payment_problem_l3095_309505


namespace NUMINAMATH_CALUDE_problem_1_l3095_309553

theorem problem_1 : |-3| - 2 - (-6) / (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3095_309553


namespace NUMINAMATH_CALUDE_container_emptying_l3095_309573

/-- Represents the state of the three containers -/
structure ContainerState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a valid transfer between containers -/
inductive Transfer : ContainerState → ContainerState → Prop where
  | ab {s t : ContainerState} : t.a = s.a + s.a ∧ t.b = s.b - s.a ∧ t.c = s.c → Transfer s t
  | ac {s t : ContainerState} : t.a = s.a + s.a ∧ t.b = s.b ∧ t.c = s.c - s.a → Transfer s t
  | ba {s t : ContainerState} : t.a = s.a - s.b ∧ t.b = s.b + s.b ∧ t.c = s.c → Transfer s t
  | bc {s t : ContainerState} : t.a = s.a ∧ t.b = s.b + s.b ∧ t.c = s.c - s.b → Transfer s t
  | ca {s t : ContainerState} : t.a = s.a - s.c ∧ t.b = s.b ∧ t.c = s.c + s.c → Transfer s t
  | cb {s t : ContainerState} : t.a = s.a ∧ t.b = s.b - s.c ∧ t.c = s.c + s.c → Transfer s t

/-- A sequence of transfers -/
def TransferSeq : ContainerState → ContainerState → Prop :=
  Relation.ReflTransGen Transfer

/-- The main theorem stating that it's always possible to empty a container -/
theorem container_emptying (initial : ContainerState) : 
  ∃ (final : ContainerState), TransferSeq initial final ∧ (final.a = 0 ∨ final.b = 0 ∨ final.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_container_emptying_l3095_309573


namespace NUMINAMATH_CALUDE_unique_outstanding_wins_all_l3095_309597

variable {α : Type*} [Fintype α] [DecidableEq α]

-- Define the winning relation
variable (wins : α → α → Prop)

-- Assumption: Every pair of contestants has a clear winner
axiom clear_winner (a b : α) : a ≠ b → (wins a b ∨ wins b a) ∧ ¬(wins a b ∧ wins b a)

-- Define what it means to be an outstanding contestant
def is_outstanding (a : α) : Prop :=
  ∀ b : α, b ≠ a → wins a b ∨ (∃ c : α, wins c b ∧ wins a c)

-- Theorem: If there is a unique outstanding contestant, they win against all others
theorem unique_outstanding_wins_all (a : α) :
  (is_outstanding wins a ∧ ∀ b : α, is_outstanding wins b → b = a) →
  ∀ b : α, b ≠ a → wins a b :=
by sorry

end NUMINAMATH_CALUDE_unique_outstanding_wins_all_l3095_309597


namespace NUMINAMATH_CALUDE_arithmetic_progression_term_position_l3095_309581

theorem arithmetic_progression_term_position
  (a d : ℚ)  -- first term and common difference
  (sum_two_terms : a + 11 * d + a + (x - 1) * d = 20)  -- sum of 12th and x-th term is 20
  (sum_ten_terms : 10 * a + 45 * d = 100)  -- sum of first 10 terms is 100
  (x : ℕ)  -- position of the other term
  : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_term_position_l3095_309581


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l3095_309557

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := sorry

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 32

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 2

/-- The number of extra apples -/
def extra_apples : ℕ := 73

/-- Theorem stating that the number of red apples ordered is 43 -/
theorem cafeteria_red_apples : red_apples = 43 := by sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l3095_309557


namespace NUMINAMATH_CALUDE_production_decrease_l3095_309585

/-- The number of cars originally planned for production -/
def original_plan : ℕ := 200

/-- The number of doors per car -/
def doors_per_car : ℕ := 5

/-- The total number of doors produced after reductions -/
def total_doors : ℕ := 375

/-- The reduction factor due to pandemic -/
def pandemic_reduction : ℚ := 1/2

theorem production_decrease (x : ℕ) : 
  (pandemic_reduction * (original_plan - x : ℚ)) * doors_per_car = total_doors → 
  x = 50 := by sorry

end NUMINAMATH_CALUDE_production_decrease_l3095_309585


namespace NUMINAMATH_CALUDE_percentage_five_half_years_or_more_l3095_309537

/-- Represents the number of employees in each time period -/
structure EmployeeDistribution :=
  (less_than_half_year : ℕ)
  (half_to_one_year : ℕ)
  (one_to_one_half_years : ℕ)
  (one_half_to_two_years : ℕ)
  (two_to_two_half_years : ℕ)
  (two_half_to_three_years : ℕ)
  (three_to_three_half_years : ℕ)
  (three_half_to_four_years : ℕ)
  (four_to_four_half_years : ℕ)
  (four_half_to_five_years : ℕ)
  (five_to_five_half_years : ℕ)
  (five_half_to_six_years : ℕ)
  (six_to_six_half_years : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_half_year +
  d.half_to_one_year +
  d.one_to_one_half_years +
  d.one_half_to_two_years +
  d.two_to_two_half_years +
  d.two_half_to_three_years +
  d.three_to_three_half_years +
  d.three_half_to_four_years +
  d.four_to_four_half_years +
  d.four_half_to_five_years +
  d.five_to_five_half_years +
  d.five_half_to_six_years +
  d.six_to_six_half_years

/-- Calculates the number of employees working for 5.5 years or more -/
def employees_five_half_years_or_more (d : EmployeeDistribution) : ℕ :=
  d.five_half_to_six_years + d.six_to_six_half_years

/-- Theorem stating that the percentage of employees working for 5.5 years or more is (2/38) * 100 -/
theorem percentage_five_half_years_or_more (d : EmployeeDistribution) 
  (h1 : d.less_than_half_year = 4)
  (h2 : d.half_to_one_year = 6)
  (h3 : d.one_to_one_half_years = 7)
  (h4 : d.one_half_to_two_years = 4)
  (h5 : d.two_to_two_half_years = 3)
  (h6 : d.two_half_to_three_years = 3)
  (h7 : d.three_to_three_half_years = 3)
  (h8 : d.three_half_to_four_years = 2)
  (h9 : d.four_to_four_half_years = 2)
  (h10 : d.four_half_to_five_years = 1)
  (h11 : d.five_to_five_half_years = 1)
  (h12 : d.five_half_to_six_years = 1)
  (h13 : d.six_to_six_half_years = 1) :
  (employees_five_half_years_or_more d : ℚ) / (total_employees d : ℚ) * 100 = 526 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_five_half_years_or_more_l3095_309537


namespace NUMINAMATH_CALUDE_quadratic_roots_l3095_309547

theorem quadratic_roots :
  ∃ (x₁ x₂ : ℝ), (x₁ = 2 ∧ x₂ = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3095_309547


namespace NUMINAMATH_CALUDE_triangle_sum_maximum_l3095_309533

theorem triangle_sum_maximum (arrangement : List ℕ) : 
  (arrangement.toFinset = Finset.range 9 \ {0}) →
  (∃ (side1 side2 side3 : List ℕ), 
    side1.length = 4 ∧ side2.length = 4 ∧ side3.length = 4 ∧
    (side1 ++ side2 ++ side3).toFinset = arrangement.toFinset ∧
    side1.sum = side2.sum ∧ side2.sum = side3.sum) →
  (∀ (side : List ℕ), side.toFinset ⊆ arrangement.toFinset ∧ side.length = 4 → side.sum ≤ 19) :=
by sorry

#check triangle_sum_maximum

end NUMINAMATH_CALUDE_triangle_sum_maximum_l3095_309533


namespace NUMINAMATH_CALUDE_unique_prime_square_solution_l3095_309516

theorem unique_prime_square_solution :
  ∀ (p m : ℕ), 
    Prime p → 
    m > 0 → 
    2 * p^2 + p + 9 = m^2 → 
    p = 5 ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_square_solution_l3095_309516


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3095_309534

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3095_309534


namespace NUMINAMATH_CALUDE_corn_ratio_proof_l3095_309559

theorem corn_ratio_proof (marcel_corn : ℕ) (marcel_potatoes : ℕ) (dale_potatoes : ℕ) (total_vegetables : ℕ) :
  marcel_corn = 10 →
  marcel_potatoes = 4 →
  dale_potatoes = 8 →
  total_vegetables = 27 →
  ∃ (dale_corn : ℕ), 
    marcel_corn + marcel_potatoes + dale_corn + dale_potatoes = total_vegetables ∧
    dale_corn * 2 = marcel_corn :=
by sorry

end NUMINAMATH_CALUDE_corn_ratio_proof_l3095_309559


namespace NUMINAMATH_CALUDE_inverse_matrices_product_l3095_309582

def inverse_matrices (x y z w : ℝ) : Prop :=
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![x, 3; 4, y]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, z; w, -5]
  A * B = 1

theorem inverse_matrices_product (x y z w : ℝ) 
  (h : inverse_matrices x y z w) : x * y * z * w = -5040/49 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_product_l3095_309582


namespace NUMINAMATH_CALUDE_select_four_from_fifteen_l3095_309536

theorem select_four_from_fifteen (n : ℕ) (h : n = 15) :
  (n * (n - 1) * (n - 2) * (n - 3)) = 32760 := by
  sorry

end NUMINAMATH_CALUDE_select_four_from_fifteen_l3095_309536


namespace NUMINAMATH_CALUDE_regression_equation_properties_l3095_309596

-- Define the concept of a regression equation
structure RegressionEquation where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define the property of temporality for regression equations
def has_temporality (eq : RegressionEquation) : Prop := sorry

-- Define the concept of sample values affecting applicability
def sample_values_affect_applicability (eq : RegressionEquation) : Prop := sorry

-- Theorem stating the correct properties of regression equations
theorem regression_equation_properties :
  ∀ (eq : RegressionEquation),
    (has_temporality eq) ∧
    (sample_values_affect_applicability eq) := by
  sorry

end NUMINAMATH_CALUDE_regression_equation_properties_l3095_309596


namespace NUMINAMATH_CALUDE_problem_solution_l3095_309595

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (2*a + 2*b + 2*c - d) → d = 23/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3095_309595


namespace NUMINAMATH_CALUDE_combined_gold_cost_l3095_309501

/-- The cost of Gary and Anna's combined gold -/
theorem combined_gold_cost (gary_grams anna_grams : ℕ) (gary_price anna_price : ℚ) : 
  gary_grams = 30 → 
  gary_price = 15 → 
  anna_grams = 50 → 
  anna_price = 20 → 
  gary_grams * gary_price + anna_grams * anna_price = 1450 := by
  sorry


end NUMINAMATH_CALUDE_combined_gold_cost_l3095_309501


namespace NUMINAMATH_CALUDE_x_twelfth_power_l3095_309593

theorem x_twelfth_power (x : ℂ) (h : x + 1/x = -Real.sqrt 3) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l3095_309593


namespace NUMINAMATH_CALUDE_eccentricity_is_sqrt_three_l3095_309538

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A circle centered on a hyperbola and tangent to the x-axis at a focus -/
structure TangentCircle (h : Hyperbola) where
  center : ℝ × ℝ
  h_on_hyperbola : center.1^2 / h.a^2 - center.2^2 / h.b^2 = 1
  h_tangent_at_focus : center.1 = h.a * (h.a^2 + h.b^2).sqrt / h.a

/-- The property that the circle intersects the y-axis forming an equilateral triangle -/
def forms_equilateral_triangle (h : Hyperbola) (c : TangentCircle h) : Prop :=
  ∃ (y₁ y₂ : ℝ), 
    y₁ < c.center.2 ∧ c.center.2 < y₂ ∧
    (c.center.1^2 + (y₁ - c.center.2)^2 = (h.b^2 / h.a)^2) ∧
    (c.center.1^2 + (y₂ - c.center.2)^2 = (h.b^2 / h.a)^2) ∧
    (y₂ - y₁ = h.b^2 / h.a * Real.sqrt 3)

/-- The theorem stating that the eccentricity of the hyperbola is √3 -/
theorem eccentricity_is_sqrt_three (h : Hyperbola) (c : TangentCircle h)
  (h_equilateral : forms_equilateral_triangle h c) :
  (h.a^2 + h.b^2).sqrt / h.a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_is_sqrt_three_l3095_309538


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3095_309572

/-- Given a geometric sequence {a_n} where a₂ = 2 and a₆ = 8, 
    prove that a₃ * a₄ * a₅ = 64 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_a6 : a 6 = 8) : 
  a 3 * a 4 * a 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3095_309572


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l3095_309555

/-- Four real numbers are in arithmetic sequence -/
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r

/-- The sum of the first and last terms equals the sum of the middle terms -/
def sum_property (a b c d : ℝ) : Prop :=
  a + d = b + c

theorem arithmetic_sequence_condition (a b c d : ℝ) :
  (is_arithmetic_sequence a b c d → sum_property a b c d) ∧
  ¬(sum_property a b c d → is_arithmetic_sequence a b c d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l3095_309555


namespace NUMINAMATH_CALUDE_salt_problem_l3095_309579

theorem salt_problem (a x : ℝ) (h : a - x = 2 * (a - 2 * x)) : x = a / 3 := by
  sorry

end NUMINAMATH_CALUDE_salt_problem_l3095_309579


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3095_309541

/-- A line in 2D space with a given slope and x-intercept. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ := l.slope * (-l.x_intercept) + 0

/-- Theorem stating that a line with slope -3 and x-intercept (3, 0) has y-intercept (0, 9). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := 3 }
  y_intercept l = 9 := by
  sorry


end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3095_309541


namespace NUMINAMATH_CALUDE_expand_expression_l3095_309560

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3095_309560


namespace NUMINAMATH_CALUDE_power_function_through_point_l3095_309549

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 3 = 27 → ∀ x : ℝ, f x = x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3095_309549


namespace NUMINAMATH_CALUDE_johnson_calls_l3095_309578

def days_in_year : ℕ := 365

def call_frequencies : List ℕ := [2, 3, 6, 7]

/-- 
Calculates the number of days in a year where no calls are received, 
given a list of call frequencies (in days) for each grandchild.
-/
def days_without_calls (frequencies : List ℕ) (total_days : ℕ) : ℕ :=
  sorry

theorem johnson_calls : 
  days_without_calls call_frequencies days_in_year = 61 := by sorry

end NUMINAMATH_CALUDE_johnson_calls_l3095_309578


namespace NUMINAMATH_CALUDE_power_division_l3095_309515

theorem power_division (x : ℝ) (h : x ≠ 0) : x^8 / x^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l3095_309515


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l3095_309588

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8)
  (h2 : Real.sin (a - b) = 1/4) :
  Real.tan a / Real.tan b = 7/3 := by
sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l3095_309588


namespace NUMINAMATH_CALUDE_complex_equation_product_l3095_309520

theorem complex_equation_product (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (a + 2*i)/i = b + i) : a * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_product_l3095_309520


namespace NUMINAMATH_CALUDE_probability_spade_or_diamond_l3095_309587

theorem probability_spade_or_diamond (total_cards : ℕ) (ranks : ℕ) (suits : ℕ) 
  (h1 : total_cards = 52)
  (h2 : ranks = 13)
  (h3 : suits = 4)
  (h4 : total_cards = ranks * suits) :
  (2 : ℚ) * (ranks : ℚ) / (total_cards : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_spade_or_diamond_l3095_309587


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3095_309575

theorem diophantine_equation_solutions : 
  {(x, y) : ℕ × ℕ | 2 * x^2 + 2 * x * y - x + y = 2020} = {(0, 2020), (1, 673)} := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3095_309575


namespace NUMINAMATH_CALUDE_percentage_problem_l3095_309591

theorem percentage_problem : ∃ P : ℚ, P * 30 = 0.25 * 16 + 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3095_309591


namespace NUMINAMATH_CALUDE_one_white_ball_probability_l3095_309543

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (event total : ℕ) : ℚ := sorry

theorem one_white_ball_probability (bagA_white bagA_red bagB_white bagB_red : ℕ) 
  (h1 : bagA_white = 8)
  (h2 : bagA_red = 4)
  (h3 : bagB_white = 6)
  (h4 : bagB_red = 6) :
  probability 
    (choose bagA_white 1 * choose bagB_red 1 + choose bagA_red 1 * choose bagB_white 1)
    (choose (bagA_white + bagA_red) 1 * choose (bagB_white + bagB_red) 1) =
  probability 
    ((choose 8 1) * (choose 6 1) + (choose 4 1) * (choose 6 1))
    ((choose 12 1) * (choose 12 1)) :=
sorry

end NUMINAMATH_CALUDE_one_white_ball_probability_l3095_309543


namespace NUMINAMATH_CALUDE_num_paths_5x4_grid_l3095_309530

/-- The number of paths on a grid from point C to point D -/
def num_paths (grid_width grid_height path_length right_steps up_steps : ℕ) : ℕ :=
  Nat.choose path_length up_steps

/-- Theorem stating the number of paths on a 5x4 grid with specific constraints -/
theorem num_paths_5x4_grid : num_paths 5 4 8 5 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_5x4_grid_l3095_309530


namespace NUMINAMATH_CALUDE_slant_height_angle_is_30_degrees_l3095_309526

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- Side length of the base square -/
  base_side : ℝ
  /-- Angle between lateral face and base plane -/
  lateral_angle : ℝ

/-- Angle between slant height and adjacent face -/
def slant_height_angle (p : RegularQuadPyramid) : ℝ :=
  sorry

theorem slant_height_angle_is_30_degrees (p : RegularQuadPyramid) 
  (h : p.lateral_angle = Real.pi / 4) : 
  slant_height_angle p = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_slant_height_angle_is_30_degrees_l3095_309526


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3095_309512

theorem digit_sum_problem :
  ∀ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 →
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    22 * (a + b + c) = 462 →
    ((a = 4 ∧ b = 8 ∧ c = 9) ∨
     (a = 5 ∧ b = 7 ∧ c = 9) ∨
     (a = 6 ∧ b = 7 ∧ c = 8) ∨
     (a = 8 ∧ b = 4 ∧ c = 9) ∨
     (a = 7 ∧ b = 5 ∧ c = 9) ∨
     (a = 7 ∧ b = 6 ∧ c = 8) ∨
     (a = 9 ∧ b = 4 ∧ c = 8) ∨
     (a = 9 ∧ b = 5 ∧ c = 7) ∨
     (a = 8 ∧ b = 6 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3095_309512


namespace NUMINAMATH_CALUDE_nancy_added_pencils_l3095_309569

/-- The number of pencils Nancy placed in the drawer -/
def pencils_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Nancy added 45 pencils to the drawer -/
theorem nancy_added_pencils : pencils_added 27 72 = 45 := by
  sorry

end NUMINAMATH_CALUDE_nancy_added_pencils_l3095_309569


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3095_309522

theorem arithmetic_calculations :
  ((-53 + 21 + 79 - 37) = 10) ∧
  ((-9 - 1/3 - (abs (-4 - 5/6)) + (abs (0 - 5 - 1/6)) - 2/3) = -29/3) ∧
  ((-2^3 * (-4)^2 / (4/3) + abs (5 - 8)) = -93) ∧
  ((1/2 + 5/6 - 7/12) / (-1/36) = -27) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3095_309522


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3095_309527

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3095_309527


namespace NUMINAMATH_CALUDE_y_value_proof_l3095_309546

theorem y_value_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 8)
  (h2 : y + 1/x = 7/12)
  (h3 : x + y = 7) :
  y = 49/103 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l3095_309546


namespace NUMINAMATH_CALUDE_abs_c_equals_181_l3095_309580

def f (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_equals_181 (a b c : ℤ) :
  (Nat.gcd (Nat.gcd (a.natAbs) (b.natAbs)) (c.natAbs) = 1) →
  (f a b c (3 + 2*Complex.I) = 0) →
  (c.natAbs = 181) :=
sorry

end NUMINAMATH_CALUDE_abs_c_equals_181_l3095_309580


namespace NUMINAMATH_CALUDE_gcd_1213_1985_l3095_309504

theorem gcd_1213_1985 : 
  (¬ (1213 % 2 = 0)) → 
  (¬ (1213 % 3 = 0)) → 
  (¬ (1213 % 5 = 0)) → 
  (¬ (1985 % 2 = 0)) → 
  (¬ (1985 % 3 = 0)) → 
  (¬ (1985 % 5 = 0)) → 
  Nat.gcd 1213 1985 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1213_1985_l3095_309504


namespace NUMINAMATH_CALUDE_football_practice_hours_l3095_309514

/-- Calculates the total practice hours for a football team in a week with one missed day -/
theorem football_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 6 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 36 := by
sorry

end NUMINAMATH_CALUDE_football_practice_hours_l3095_309514


namespace NUMINAMATH_CALUDE_jerry_action_figures_l3095_309517

theorem jerry_action_figures (total_needed : ℕ) (cost_per_figure : ℕ) (amount_needed : ℕ) :
  total_needed = 16 →
  cost_per_figure = 8 →
  amount_needed = 72 →
  total_needed - (amount_needed / cost_per_figure) = 7 :=
by sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l3095_309517


namespace NUMINAMATH_CALUDE_point_coordinates_l3095_309529

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the fourth quadrant -/
def inFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the fourth quadrant, its distance to the x-axis is 5,
    and its distance to the y-axis is 3, then its coordinates are (3, -5) -/
theorem point_coordinates (p : Point) 
    (h1 : inFourthQuadrant p)
    (h2 : distanceToXAxis p = 5)
    (h3 : distanceToYAxis p = 3) :
    p = Point.mk 3 (-5) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3095_309529
