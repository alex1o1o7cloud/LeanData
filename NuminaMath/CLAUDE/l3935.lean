import Mathlib

namespace room_length_l3935_393563

/-- The length of a room satisfying given conditions -/
theorem room_length : ∃ (L : ℝ), 
  (L > 0) ∧ 
  (9 * (2 * 12 * (L + 15) - (6 * 3 + 3 * 4 * 3)) = 8154) → 
  L = 25 := by
  sorry

end room_length_l3935_393563


namespace travel_options_l3935_393557

/-- The number of train departures from City A to City B -/
def train_departures : ℕ := 10

/-- The number of flights from City A to City B -/
def flights : ℕ := 2

/-- The number of long-distance bus services from City A to City B -/
def bus_services : ℕ := 12

/-- The total number of ways Xiao Zhang can travel from City A to City B -/
def total_ways : ℕ := train_departures + flights + bus_services

theorem travel_options : total_ways = 24 := by sorry

end travel_options_l3935_393557


namespace trigonometric_identity_l3935_393540

theorem trigonometric_identity (θ c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h : Real.sin θ ^ 6 / c + Real.cos θ ^ 6 / d = 1 / (c + d)) :
  Real.sin θ ^ 18 / c^5 + Real.cos θ ^ 18 / d^5 = (c^4 + d^4) / (c + d)^9 := by
  sorry

end trigonometric_identity_l3935_393540


namespace problem_statement_l3935_393520

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  (a < 1/2 ∧ 1/2 < b) ∧ (a < a^2 + b^2 ∧ a^2 + b^2 < b) := by
  sorry

end problem_statement_l3935_393520


namespace jack_school_time_l3935_393514

/-- Given information about Dave and Jack's walking speeds and Dave's time to school,
    prove that Jack takes 18 minutes to reach the same school. -/
theorem jack_school_time (dave_steps_per_min : ℕ) (dave_step_length : ℕ) (dave_time : ℕ)
                         (jack_steps_per_min : ℕ) (jack_step_length : ℕ) :
  dave_steps_per_min = 90 →
  dave_step_length = 75 →
  dave_time = 16 →
  jack_steps_per_min = 100 →
  jack_step_length = 60 →
  (dave_steps_per_min * dave_step_length * dave_time) / (jack_steps_per_min * jack_step_length) = 18 :=
by sorry

end jack_school_time_l3935_393514


namespace triangle_altitude_segment_l3935_393593

theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  a^2 = x^2 + h^2 → b^2 = (c - x)^2 + h^2 → 
  c - x = 82.5 := by sorry

end triangle_altitude_segment_l3935_393593


namespace gcd_59_power_l3935_393536

theorem gcd_59_power : Nat.gcd (59^7 + 1) (59^7 + 59^3 + 1) = 1 := by
  sorry

end gcd_59_power_l3935_393536


namespace age_difference_is_four_l3935_393515

/-- The age difference between Angelina and Justin -/
def ageDifference (angelinaFutureAge : ℕ) (justinCurrentAge : ℕ) : ℕ :=
  (angelinaFutureAge - 5) - justinCurrentAge

/-- Theorem stating that the age difference between Angelina and Justin is 4 years -/
theorem age_difference_is_four :
  ageDifference 40 31 = 4 := by
  sorry

end age_difference_is_four_l3935_393515


namespace mindy_income_multiplier_l3935_393524

/-- Given tax rates and combined rate, prove Mindy's income multiplier --/
theorem mindy_income_multiplier 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (combined_rate : ℝ) 
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.25)
  (h3 : combined_rate = 0.29) :
  ∃ k : ℝ, k = 4 ∧ 
    (mork_rate + mindy_rate * k) / (1 + k) = combined_rate :=
by sorry

end mindy_income_multiplier_l3935_393524


namespace roots_of_polynomial_l3935_393559

theorem roots_of_polynomial (x : ℝ) : 
  x^2 * (x - 5)^2 * (x + 3) = 0 ↔ x = 0 ∨ x = 5 ∨ x = -3 := by
  sorry

end roots_of_polynomial_l3935_393559


namespace acute_angles_insufficient_for_congruence_l3935_393589

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- leg
  b : ℝ  -- leg
  c : ℝ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define congruence for right-angled triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define equality of acute angles
def equal_acute_angles (t1 t2 : RightTriangle) : Prop :=
  Real.arctan (t1.a / t1.b) = Real.arctan (t2.a / t2.b) ∧
  Real.arctan (t1.b / t1.a) = Real.arctan (t2.b / t2.a)

-- Theorem statement
theorem acute_angles_insufficient_for_congruence :
  ∃ (t1 t2 : RightTriangle), equal_acute_angles t1 t2 ∧ ¬congruent t1 t2 :=
sorry

end acute_angles_insufficient_for_congruence_l3935_393589


namespace arcsin_sin_2x_solutions_l3935_393567

theorem arcsin_sin_2x_solutions (x : Real) :
  x ∈ Set.Icc (-π/2) (π/2) ∧ Real.arcsin (Real.sin (2*x)) = x ↔ x = 0 ∨ x = -π/3 ∨ x = π/3 := by
  sorry

end arcsin_sin_2x_solutions_l3935_393567


namespace parallel_vectors_x_value_l3935_393556

def vector_a : ℝ × ℝ := (2, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -6)

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ vector_a = k • vector_b x) → x = -4 := by
  sorry

end parallel_vectors_x_value_l3935_393556


namespace cricket_team_age_difference_l3935_393588

/-- Prove that the difference between the average age of the whole cricket team
and the average age of the remaining players is 3 years. -/
theorem cricket_team_age_difference 
  (team_size : ℕ) 
  (team_avg_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_avg_age : ℝ) 
  (h1 : team_size = 11)
  (h2 : team_avg_age = 26)
  (h3 : wicket_keeper_age_diff = 3)
  (h4 : remaining_avg_age = 23) :
  team_avg_age - remaining_avg_age = 3 := by
sorry

end cricket_team_age_difference_l3935_393588


namespace other_side_formula_l3935_393500

/-- Represents a rectangle with perimeter 30 and one side x -/
structure Rectangle30 where
  x : ℝ
  other : ℝ
  perimeter_eq : x + other = 15

theorem other_side_formula (rect : Rectangle30) : rect.other = 15 - rect.x := by
  sorry

end other_side_formula_l3935_393500


namespace remainder_prime_divisible_by_210_l3935_393587

theorem remainder_prime_divisible_by_210 (p r : ℕ) : 
  Prime p → 
  r = p % 210 → 
  0 < r → 
  r < 210 → 
  ¬ Prime r → 
  (∃ (a b : ℕ), r = a^2 + b^2) → 
  r = 169 := by sorry

end remainder_prime_divisible_by_210_l3935_393587


namespace trucks_distance_l3935_393509

-- Define the speeds of trucks A and B in km/h
def speed_A : ℝ := 54
def speed_B : ℝ := 72

-- Define the time elapsed in seconds
def time_elapsed : ℝ := 30

-- Define the conversion factor from km to meters
def km_to_meters : ℝ := 1000

-- Define the conversion factor from hours to seconds
def hours_to_seconds : ℝ := 3600

-- Theorem statement
theorem trucks_distance :
  let speed_A_mps := speed_A * km_to_meters / hours_to_seconds
  let speed_B_mps := speed_B * km_to_meters / hours_to_seconds
  let distance_A := speed_A_mps * time_elapsed
  let distance_B := speed_B_mps * time_elapsed
  distance_A + distance_B = 1050 :=
by sorry

end trucks_distance_l3935_393509


namespace smallest_b_value_l3935_393522

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 8) :
  ∀ k : ℕ+, k.val < b.val → ¬(∃ a' : ℕ+, a'.val - k.val = 8 ∧ 
    Nat.gcd ((a'.val^3 + k.val^3) / (a'.val + k.val)) (a'.val * k.val) = 8) :=
sorry

end smallest_b_value_l3935_393522


namespace bruce_mangoes_purchase_l3935_393505

/-- The amount of grapes purchased in kg -/
def grapes_kg : ℕ := 8

/-- The price of grapes per kg -/
def grapes_price : ℕ := 70

/-- The price of mangoes per kg -/
def mangoes_price : ℕ := 55

/-- The total amount paid -/
def total_paid : ℕ := 1110

/-- The amount of mangoes purchased in kg -/
def mangoes_kg : ℕ := (total_paid - grapes_kg * grapes_price) / mangoes_price

theorem bruce_mangoes_purchase :
  mangoes_kg = 10 := by sorry

end bruce_mangoes_purchase_l3935_393505


namespace jisoo_drank_least_l3935_393594

-- Define the amount of juice each person drank
def jennie_juice : ℚ := 9/5

-- Define Jisoo's juice amount in terms of Jennie's
def jisoo_juice : ℚ := jennie_juice - 1/5

-- Define Rohee's juice amount in terms of Jisoo's
def rohee_juice : ℚ := jisoo_juice + 3/10

-- Theorem statement
theorem jisoo_drank_least : 
  jisoo_juice < jennie_juice ∧ jisoo_juice < rohee_juice := by
  sorry


end jisoo_drank_least_l3935_393594


namespace function_inequality_condition_l3935_393516

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^2 + x + 1) →
  a > 0 →
  b > 0 →
  (∀ x, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 3 :=
by sorry

end function_inequality_condition_l3935_393516


namespace initial_men_fraction_l3935_393529

theorem initial_men_fraction (initial_total : ℕ) (new_hires : ℕ) (final_women_percentage : ℚ) : 
  initial_total = 90 → 
  new_hires = 10 → 
  final_women_percentage = 2/5 → 
  (initial_total - (final_women_percentage * (initial_total + new_hires)).num) / initial_total = 2/3 := by
sorry

end initial_men_fraction_l3935_393529


namespace mean_variance_relationship_l3935_393591

-- Define the sample size
def sample_size : Nat := 50

-- Define the original mean and variance
def original_mean : Real := 70
def original_variance : Real := 75

-- Define the incorrect and correct data points
def incorrect_point1 : Real := 60
def incorrect_point2 : Real := 90
def correct_point1 : Real := 80
def correct_point2 : Real := 70

-- Define the new mean and variance after correction
def new_mean : Real := original_mean
noncomputable def new_variance : Real := original_variance - 8

-- Theorem statement
theorem mean_variance_relationship :
  new_mean = original_mean ∧ new_variance < original_variance :=
by sorry

end mean_variance_relationship_l3935_393591


namespace profit_percent_l3935_393554

theorem profit_percent (P : ℝ) (C : ℝ) (h : P > 0) (h2 : C > 0) :
  (2/3 * P = 0.84 * C) → (P - C) / C * 100 = 26 := by
  sorry

end profit_percent_l3935_393554


namespace sin_function_properties_l3935_393517

noncomputable def f (x φ A : ℝ) : ℝ := Real.sin (2 * x + φ) + A

theorem sin_function_properties (φ A : ℝ) :
  -- Amplitude is A
  (∃ (x : ℝ), f x φ A - A = 1) ∧
  (∀ (x : ℝ), f x φ A - A ≤ 1) ∧
  -- Period is π
  (∀ (x : ℝ), f (x + π) φ A = f x φ A) ∧
  -- Initial phase is φ
  (∀ (x : ℝ), f x φ A = Real.sin (2 * x + φ) + A) ∧
  -- Maximum value occurs when x = π/4 + kπ, k ∈ ℤ
  (∀ (x : ℝ), f x φ A = A + 1 ↔ ∃ (k : ℤ), x = π/4 + k * π) :=
by sorry

end sin_function_properties_l3935_393517


namespace fraction_product_equality_l3935_393586

theorem fraction_product_equality : (1 / 3 : ℚ)^4 * (1 / 8 : ℚ) = 1 / 648 := by
  sorry

end fraction_product_equality_l3935_393586


namespace last_boat_occupancy_l3935_393576

/-- The number of tourists in the travel group -/
def total_tourists (x : ℕ) : ℕ := 8 * x + 6

/-- The number of people that can be seated in (x-2) fully occupied 12-seat boats -/
def seated_tourists (x : ℕ) : ℕ := 12 * (x - 2)

theorem last_boat_occupancy (x : ℕ) (h : x > 2) :
  total_tourists x - seated_tourists x = 30 - 4 * x :=
by sorry

end last_boat_occupancy_l3935_393576


namespace prob_same_gender_specific_schools_l3935_393523

/-- Represents a school with a certain number of male and female teachers -/
structure School :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- The probability of selecting two teachers of the same gender from two schools -/
def prob_same_gender (school_a school_b : School) : ℚ :=
  let total_combinations := school_a.male_count * school_b.male_count + 
                            school_a.female_count * school_b.female_count
  let total_selections := (school_a.male_count + school_a.female_count) * 
                          (school_b.male_count + school_b.female_count)
  total_combinations / total_selections

theorem prob_same_gender_specific_schools :
  let school_a : School := ⟨2, 1⟩
  let school_b : School := ⟨1, 2⟩
  prob_same_gender school_a school_b = 4/9 := by
  sorry

end prob_same_gender_specific_schools_l3935_393523


namespace present_worth_calculation_l3935_393572

/-- Calculates the present worth of a sum given the banker's gain, time period, and interest rate -/
def present_worth (bankers_gain : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let simple_interest := (bankers_gain * rate * (100 + rate * time)).sqrt
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that the present worth is 7755 given the specified conditions -/
theorem present_worth_calculation :
  present_worth 24 2 (10/100) = 7755 := by
  sorry

end present_worth_calculation_l3935_393572


namespace car_distance_l3935_393568

/-- Proves that a car traveling 2/3 as fast as a train going 90 miles per hour will cover 40 miles in 40 minutes -/
theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (travel_time_minutes : ℝ) : 
  train_speed = 90 →
  car_speed_ratio = 2/3 →
  travel_time_minutes = 40 →
  (car_speed_ratio * train_speed) * (travel_time_minutes / 60) = 40 := by
  sorry

#check car_distance

end car_distance_l3935_393568


namespace smallest_sticker_collection_l3935_393575

theorem smallest_sticker_collection (S : ℕ) : 
  S > 1 ∧ 
  S % 5 = 2 ∧ 
  S % 9 = 2 ∧ 
  S % 11 = 2 → 
  S ≥ 497 :=
by sorry

end smallest_sticker_collection_l3935_393575


namespace root_polynomial_relation_l3935_393595

theorem root_polynomial_relation : ∃ (b c : ℤ), 
  (∀ r : ℝ, r^2 - 2*r - 1 = 0 → r^5 - b*r - c = 0) ∧ b*c = 348 := by
  sorry

end root_polynomial_relation_l3935_393595


namespace intersection_of_M_and_N_l3935_393531

def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_of_M_and_N : M ∩ N = {x | x > 1} := by sorry

end intersection_of_M_and_N_l3935_393531


namespace permutations_of_five_l3935_393551

theorem permutations_of_five (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end permutations_of_five_l3935_393551


namespace divisibility_of_20_pow_15_minus_1_l3935_393555

theorem divisibility_of_20_pow_15_minus_1 :
  (11 : ℕ) * 31 * 61 ∣ 20^15 - 1 := by
  sorry

end divisibility_of_20_pow_15_minus_1_l3935_393555


namespace math_exam_total_points_l3935_393597

/-- The total number of points in a math exam, given the scores of three students and the number of mistakes made by one of them. -/
theorem math_exam_total_points (bryan_score jen_score sammy_score : ℕ) (sammy_mistakes : ℕ) : 
  bryan_score = 20 →
  jen_score = bryan_score + 10 →
  sammy_score = jen_score - 2 →
  sammy_mistakes = 7 →
  bryan_score + (jen_score - bryan_score) + (sammy_score - jen_score + 2) + sammy_mistakes = 35 :=
by sorry

end math_exam_total_points_l3935_393597


namespace gina_coin_value_l3935_393581

/-- Calculates the total value of a pile of coins given the total number of coins and the number of dimes. -/
def total_coin_value (total_coins : ℕ) (num_dimes : ℕ) : ℚ :=
  let num_nickels : ℕ := total_coins - num_dimes
  let dime_value : ℚ := 10 / 100
  let nickel_value : ℚ := 5 / 100
  (num_dimes : ℚ) * dime_value + (num_nickels : ℚ) * nickel_value

/-- Proves that given 50 total coins with 14 dimes, the total value is $3.20. -/
theorem gina_coin_value : total_coin_value 50 14 = 32 / 10 := by
  sorry

end gina_coin_value_l3935_393581


namespace student_number_calculation_l3935_393598

theorem student_number_calculation (x : ℤ) (h : x = 63) : (x * 4) - 142 = 110 := by
  sorry

end student_number_calculation_l3935_393598


namespace product_ends_in_zero_theorem_l3935_393571

def is_valid_assignment (assignment : Char → ℕ) : Prop :=
  (∀ c₁ c₂, c₁ ≠ c₂ → assignment c₁ ≠ assignment c₂) ∧
  (∀ c, assignment c < 10)

def satisfies_equation (assignment : Char → ℕ) : Prop :=
  10 * (assignment 'Ж') + (assignment 'Ж') + (assignment 'Ж') =
  100 * (assignment 'М') + 10 * (assignment 'Ё') + (assignment 'Д')

def product_ends_in_zero (assignment : Char → ℕ) : Prop :=
  (assignment 'В' * assignment 'И' * assignment 'H' * assignment 'H' *
   assignment 'U' * assignment 'П' * assignment 'У' * assignment 'X') % 10 = 0

theorem product_ends_in_zero_theorem (assignment : Char → ℕ) :
  is_valid_assignment assignment → satisfies_equation assignment →
  product_ends_in_zero assignment :=
by
  sorry

#check product_ends_in_zero_theorem

end product_ends_in_zero_theorem_l3935_393571


namespace intersection_M_N_l3935_393558

def M : Set ℝ := {x | 2*x - x^2 ≥ 0}
def N : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - x^2)}

theorem intersection_M_N : M ∩ N = Set.Icc 0 1 \ {1} := by sorry

end intersection_M_N_l3935_393558


namespace quadratic_roots_difference_l3935_393573

theorem quadratic_roots_difference (R : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*α - R = 0 ∧ β^2 - 2*β - R = 0 ∧ α - β = 12) → R = 35 := by
  sorry

end quadratic_roots_difference_l3935_393573


namespace shoe_pairs_in_box_l3935_393513

theorem shoe_pairs_in_box (total_shoes : ℕ) (prob_matching : ℚ) : 
  total_shoes = 18 → prob_matching = 1 / 17 → ∃ n : ℕ, n * 2 = total_shoes ∧ n = 9 :=
by sorry

end shoe_pairs_in_box_l3935_393513


namespace reciprocal_of_complex_l3935_393537

/-- The reciprocal of the complex number -3 + 4i is -0.12 - 0.16i -/
theorem reciprocal_of_complex (G : ℂ) : 
  G = -3 + 4*I → 1 / G = -0.12 - 0.16*I := by
  sorry

end reciprocal_of_complex_l3935_393537


namespace expression_change_l3935_393527

theorem expression_change (x b : ℝ) (hb : b > 0) : 
  let f := fun t => t^3 - 2*t + 1
  (f (x + b) - f x = 3*b*x^2 + 3*b^2*x + b^3 - 2*b) ∧ 
  (f (x - b) - f x = -3*b*x^2 + 3*b^2*x - b^3 + 2*b) := by
  sorry

end expression_change_l3935_393527


namespace pasta_preference_ratio_l3935_393582

theorem pasta_preference_ratio (total_students : ℕ) 
  (spaghetti ravioli fettuccine penne : ℕ) : 
  total_students = 800 →
  spaghetti = 300 →
  ravioli = 200 →
  fettuccine = 150 →
  penne = 150 →
  (fettuccine : ℚ) / penne = 1 := by
  sorry

end pasta_preference_ratio_l3935_393582


namespace union_eq_univ_complement_inter_eq_open_interval_range_of_a_l3935_393549

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 6}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statements
theorem union_eq_univ : A ∪ B = Set.univ := by sorry

theorem complement_inter_eq_open_interval :
  (Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6} := by sorry

theorem range_of_a (h : ∀ a, C a ⊆ B) :
  {a | ∀ x, x ∈ C a → x ∈ B} = Set.Icc (-2) 8 := by sorry

end union_eq_univ_complement_inter_eq_open_interval_range_of_a_l3935_393549


namespace integer_difference_l3935_393532

theorem integer_difference (x y : ℤ) (h1 : x < y) (h2 : x + y = -9) (h3 : x = -5) (h4 : y = -4) : y - x = 1 := by
  sorry

end integer_difference_l3935_393532


namespace arithmetic_sequence_property_l3935_393592

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 := by
  sorry

end arithmetic_sequence_property_l3935_393592


namespace four_line_theorem_l3935_393539

/-- A line in a plane -/
structure Line where
  -- Add necessary fields here
  
/-- A point in a plane -/
structure Point where
  -- Add necessary fields here

/-- A circle in a plane -/
structure Circle where
  -- Add necessary fields here

/-- The set of four lines in the plane -/
def FourLines : Type := Fin 4 → Line

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- Predicate to check if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- Get the intersection point of two lines -/
def intersection (l1 l2 : Line) : Point := sorry

/-- Get the circumcircle of three points -/
def circumcircle (p1 p2 p3 : Point) : Circle := sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Main theorem -/
theorem four_line_theorem (lines : FourLines) :
  (∀ i j, i ≠ j → ¬are_parallel (lines i) (lines j)) →
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬are_concurrent (lines i) (lines j) (lines k)) →
  ∃ p : Point, ∀ i j k l,
    i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l →
    point_on_circle p (circumcircle 
      (intersection (lines i) (lines j))
      (intersection (lines j) (lines k))
      (intersection (lines k) (lines i))) :=
sorry

end four_line_theorem_l3935_393539


namespace abs_value_problem_l3935_393564

theorem abs_value_problem (x p : ℝ) : 
  |x - 3| = p ∧ x > 3 → x - p = 3 := by
sorry

end abs_value_problem_l3935_393564


namespace unique_root_is_half_l3935_393508

/-- Given real numbers a, b, c forming an arithmetic sequence with a ≥ b ≥ c ≥ 0,
    and the quadratic equation ax^2 - bx + c = 0 having exactly one root,
    prove that this root is 1/2. -/
theorem unique_root_is_half (a b c : ℝ) 
    (arith_seq : ∃ (d : ℝ), b = a - d ∧ c = a - 2*d)
    (ordered : a ≥ b ∧ b ≥ c ∧ c ≥ 0)
    (one_root : ∃! x, a*x^2 - b*x + c = 0) :
    ∃ x, a*x^2 - b*x + c = 0 ∧ x = 1/2 := by
  sorry

end unique_root_is_half_l3935_393508


namespace g_range_l3935_393535

def g (x : ℝ) := x^2 - 2*x

theorem g_range :
  ∀ x ∈ Set.Icc 0 3, -1 ≤ g x ∧ g x ≤ 3 ∧
  (∃ x₁ ∈ Set.Icc 0 3, g x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc 0 3, g x₂ = 3) :=
sorry

end g_range_l3935_393535


namespace fraction_value_l3935_393552

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 10) :
  (x + y) / (x - y) = -Real.sqrt (3 / 2) := by
  sorry

end fraction_value_l3935_393552


namespace function_properties_l3935_393519

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
variable (h : ∀ x y : ℝ, f (x * y) = x * f y + y * f x)

-- Theorem statement
theorem function_properties :
  (f 0 = 0) ∧ (f (-1) = 0) ∧ (∀ x : ℝ, f (-x) = -f x) :=
by sorry

end function_properties_l3935_393519


namespace unique_number_property_l3935_393546

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end unique_number_property_l3935_393546


namespace intersection_with_complement_is_empty_l3935_393550

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3}
def B : Set Nat := {1, 3, 4}

theorem intersection_with_complement_is_empty :
  A ∩ (U \ B) = ∅ := by
  sorry

end intersection_with_complement_is_empty_l3935_393550


namespace extreme_value_and_range_l3935_393583

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - a * x) / Real.exp x

theorem extreme_value_and_range :
  (∃ x : ℝ, ∀ y : ℝ, f 1 y ≥ f 1 x ∧ f 1 x = -1 / Real.exp 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x ≥ 1 - 2 * x) ↔ a ≤ 1) :=
sorry

end extreme_value_and_range_l3935_393583


namespace arithmetic_sequence_terms_l3935_393518

theorem arithmetic_sequence_terms (a d : ℝ) (n : ℕ) : 
  (n / 2 : ℝ) * (2 * a + (n - 1 : ℝ) * 2 * d) = 24 →
  (n / 2 : ℝ) * (2 * (a + d) + (n - 1 : ℝ) * 2 * d) = 30 →
  a + ((2 * n - 1 : ℝ) * d) - a = 10.5 →
  2 * n = 8 := by
  sorry

end arithmetic_sequence_terms_l3935_393518


namespace two_integers_make_fraction_integer_l3935_393525

theorem two_integers_make_fraction_integer : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, (1750 : ℕ) ∣ (m^2 - 4)) ∧ 
    (∀ m : ℕ, m > 0 → (1750 : ℕ) ∣ (m^2 - 4) → m ∈ S) ∧ 
    S.card = 2 := by
  sorry

end two_integers_make_fraction_integer_l3935_393525


namespace sum_of_three_numbers_l3935_393570

theorem sum_of_three_numbers (a b c : ℕ) : 
  a = 200 → 
  b = 2 * c → 
  c = 100 → 
  a + b + c = 500 := by
  sorry

end sum_of_three_numbers_l3935_393570


namespace max_sum_square_roots_l3935_393596

/-- Given a positive real number k, the function f(x) = x + √(k - x^2) 
    reaches its maximum value of √(2k) when x = √(k/2) -/
theorem max_sum_square_roots (k : ℝ) (h : k > 0) :
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 ≤ k ∧
    (∀ (y : ℝ), y ≥ 0 → y^2 ≤ k → x + Real.sqrt (k - x^2) ≥ y + Real.sqrt (k - y^2)) ∧
    x + Real.sqrt (k - x^2) = Real.sqrt (2 * k) ∧
    x = Real.sqrt (k / 2) := by
  sorry

end max_sum_square_roots_l3935_393596


namespace solution_satisfies_system_l3935_393577

theorem solution_satisfies_system :
  let x : ℝ := 1
  let y : ℝ := 1/2
  let w : ℝ := -1/2
  let z : ℝ := 1/3
  (Real.sqrt x - 1/y - 2*w + 3*z = 1) ∧
  (x + 1/(y^2) - 4*(w^2) - 9*(z^2) = 3) ∧
  (x * Real.sqrt x - 1/(y^3) - 8*(w^3) + 27*(z^3) = -5) ∧
  (x^2 + 1/(y^4) - 16*(w^4) - 81*(z^4) = 15) :=
by sorry

end solution_satisfies_system_l3935_393577


namespace correct_statements_reflect_relationship_l3935_393561

-- Define the statements
inductive Statement
| WaitingForRabbit
| GoodThingsThroughHardship
| PreventMinorIssues
| Insignificant

-- Define the philosophical principles
structure PhilosophicalPrinciple where
  name : String
  description : String

-- Define the relationship between quantitative and qualitative change
def reflectsQuantQualRelationship (s : Statement) (p : PhilosophicalPrinciple) : Prop :=
  match s with
  | Statement.GoodThingsThroughHardship => p.name = "Accumulation"
  | Statement.PreventMinorIssues => p.name = "Moderation"
  | _ => False

-- Theorem statement
theorem correct_statements_reflect_relationship :
  ∃ (p1 p2 : PhilosophicalPrinciple),
    reflectsQuantQualRelationship Statement.GoodThingsThroughHardship p1 ∧
    reflectsQuantQualRelationship Statement.PreventMinorIssues p2 :=
  sorry

end correct_statements_reflect_relationship_l3935_393561


namespace total_beads_used_l3935_393528

def necklaces_monday : ℕ := 10
def necklaces_tuesday : ℕ := 2
def bracelets : ℕ := 5
def earrings : ℕ := 7
def beads_per_necklace : ℕ := 20
def beads_per_bracelet : ℕ := 10
def beads_per_earring : ℕ := 5

theorem total_beads_used :
  (necklaces_monday + necklaces_tuesday) * beads_per_necklace +
  bracelets * beads_per_bracelet +
  earrings * beads_per_earring = 325 := by
sorry

end total_beads_used_l3935_393528


namespace polynomial_division_remainder_l3935_393510

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, 
    5 * X^6 + 3 * X^4 - 2 * X^3 + 7 * X^2 + 4 = 
    (X^2 + 2 * X + 1) * q + (-38 * X - 29) := by
  sorry

end polynomial_division_remainder_l3935_393510


namespace matthew_baking_time_l3935_393565

/-- The time it takes Matthew to make caramel-apple coffee cakes when his oven malfunctions -/
def baking_time (assembly_time bake_time_normal decorate_time : ℝ) : ℝ :=
  assembly_time + 2 * bake_time_normal + decorate_time

/-- Theorem stating that Matthew's total baking time is 5 hours when his oven malfunctions -/
theorem matthew_baking_time :
  baking_time 1 1.5 1 = 5 := by
  sorry

end matthew_baking_time_l3935_393565


namespace unique_positive_solution_l3935_393547

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 - 7 * x - 6 = 0 :=
by
  -- The unique positive solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the conditions
    constructor
    · -- Prove 3 > 0
      sorry
    · -- Prove 3 * 3^2 - 7 * 3 - 6 = 0
      sorry
  · -- Prove uniqueness
    sorry

end unique_positive_solution_l3935_393547


namespace intersection_implies_a_equals_one_l3935_393504

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem intersection_implies_a_equals_one :
  ∀ a : ℝ, (A ∩ B a = {3}) → a = 1 := by
  sorry

end intersection_implies_a_equals_one_l3935_393504


namespace min_odd_integers_l3935_393512

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 34)
  (sum2 : a + b + c + d = 51)
  (sum3 : a + b + c + d + e + f = 72) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ odds, Odd x) ∧ 
    odds.card = 2 ∧
    (∀ (odds' : Finset ℤ), odds' ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ odds', Odd x) → odds'.card ≥ 2) :=
by sorry

end min_odd_integers_l3935_393512


namespace cartesian_to_polar_conversion_l3935_393585

theorem cartesian_to_polar_conversion (x y ρ θ : ℝ) :
  x = -1 ∧ y = Real.sqrt 3 →
  ρ = 2 ∧ θ = 2 * Real.pi / 3 →
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ^2 = x^2 + y^2 := by
  sorry

end cartesian_to_polar_conversion_l3935_393585


namespace number_of_red_balls_l3935_393502

/-- Given a bag with white and red balls, prove the number of red balls. -/
theorem number_of_red_balls
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (red_frequency : ℚ) 
  (h1 : white_balls = 60)
  (h2 : red_frequency = 1/4)
  (h3 : total_balls = white_balls / (1 - red_frequency)) :
  total_balls - white_balls = 20 :=
by sorry

end number_of_red_balls_l3935_393502


namespace base4_division_l3935_393543

-- Define a function to convert from base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (4^i)) 0

-- Define the numbers in base 4
def num2013Base4 : List Nat := [3, 1, 0, 2]
def num13Base4 : List Nat := [3, 1]
def result13Base4 : List Nat := [3, 1]

-- State the theorem
theorem base4_division :
  (base4ToDecimal num2013Base4) / (base4ToDecimal num13Base4) = base4ToDecimal result13Base4 :=
sorry

end base4_division_l3935_393543


namespace hexagon_angle_measure_l3935_393579

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 138) (h2 : b = 85) (h3 : c = 130) (h4 : d = 120) (h5 : e = 95) :
  720 - (a + b + c + d + e) = 152 := by
  sorry

end hexagon_angle_measure_l3935_393579


namespace expression_value_l3935_393578

theorem expression_value : (4 * 4 + 4) / (2 * 2 - 2) = 10 := by
  sorry

end expression_value_l3935_393578


namespace game_ends_in_49_rounds_l3935_393544

/-- Represents a player in the token game -/
inductive Player : Type
  | A | B | C | D

/-- The state of the game at any point -/
structure GameState :=
  (tokens : Player → Nat)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := fun p => match p with
    | Player.A => 16
    | Player.B => 15
    | Player.C => 14
    | Player.D => 13 }

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (state : GameState) (count : Nat := 0) : Nat :=
  sorry

theorem game_ends_in_49_rounds :
  countRounds initialState = 49 :=
sorry

end game_ends_in_49_rounds_l3935_393544


namespace units_digit_63_plus_74_base9_l3935_393580

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (a b : ℕ) : ℕ := a * 9 + b

/-- Calculates the units digit of a number in base 9 -/
def unitsDigitBase9 (n : ℕ) : ℕ := n % 9

theorem units_digit_63_plus_74_base9 :
  unitsDigitBase9 (base9ToBase10 6 3 + base9ToBase10 7 4) = 7 := by
  sorry

end units_digit_63_plus_74_base9_l3935_393580


namespace cricket_team_size_l3935_393541

/-- The number of players on a cricket team satisfying certain conditions -/
theorem cricket_team_size :
  ∀ (total_players throwers right_handed : ℕ),
    throwers = 37 →
    right_handed = 57 →
    3 * (right_handed - throwers) = 2 * (total_players - throwers) →
    total_players = 67 := by
  sorry

end cricket_team_size_l3935_393541


namespace parking_lot_ratio_l3935_393584

/-- Proves that the ratio of full-sized car spaces to compact car spaces is 11:4 
    given the total number of spaces and the number of full-sized car spaces. -/
theorem parking_lot_ratio (total_spaces full_sized_spaces : ℕ) 
  (h1 : total_spaces = 450)
  (h2 : full_sized_spaces = 330) :
  (full_sized_spaces : ℚ) / (total_spaces - full_sized_spaces : ℚ) = 11 / 4 := by
  sorry

#check parking_lot_ratio

end parking_lot_ratio_l3935_393584


namespace half_percent_of_160_l3935_393553

theorem half_percent_of_160 : (1 / 2 * 1 / 100) * 160 = 0.8 := by
  sorry

end half_percent_of_160_l3935_393553


namespace negative_forty_divided_by_five_l3935_393533

theorem negative_forty_divided_by_five : (-40 : ℤ) / 5 = -8 := by
  sorry

end negative_forty_divided_by_five_l3935_393533


namespace max_cylinder_lateral_area_l3935_393530

/-- Given a rectangle with perimeter 36, prove that when rotated around one of its edges
    to form a cylinder, the maximum lateral surface area of the cylinder is 81. -/
theorem max_cylinder_lateral_area (l w : ℝ) : 
  (l + w = 18) →  -- Perimeter condition: 2(l + w) = 36, simplified to l + w = 18
  (∃ (h r : ℝ), h = w ∧ 2 * π * r = l ∧ 2 * π * r * h ≤ 81) ∧ 
  (∃ (h r : ℝ), h = w ∧ 2 * π * r = l ∧ 2 * π * r * h = 81) :=
sorry

end max_cylinder_lateral_area_l3935_393530


namespace expected_weekly_rain_l3935_393599

/-- Represents the possible weather outcomes for a day -/
inductive Weather
  | Sun
  | Rain3
  | Rain8

/-- The probability distribution of weather outcomes -/
def weather_prob : Weather → ℝ
  | Weather.Sun => 0.3
  | Weather.Rain3 => 0.4
  | Weather.Rain8 => 0.3

/-- The amount of rain for each weather outcome -/
def rain_amount : Weather → ℝ
  | Weather.Sun => 0
  | Weather.Rain3 => 3
  | Weather.Rain8 => 8

/-- The number of days in the week -/
def days_in_week : ℕ := 7

/-- Expected value of rain for a single day -/
def expected_daily_rain : ℝ :=
  (weather_prob Weather.Sun * rain_amount Weather.Sun) +
  (weather_prob Weather.Rain3 * rain_amount Weather.Rain3) +
  (weather_prob Weather.Rain8 * rain_amount Weather.Rain8)

/-- Theorem: The expected value of the total amount of rain for seven days is 25.2 inches -/
theorem expected_weekly_rain :
  (days_in_week : ℝ) * expected_daily_rain = 25.2 := by
  sorry


end expected_weekly_rain_l3935_393599


namespace total_cars_theorem_l3935_393534

/-- Calculates the total number of cars at the end of the play -/
def total_cars_at_end (front_cars : ℕ) (back_multiplier : ℕ) (additional_cars : ℕ) : ℕ :=
  front_cars + (back_multiplier * front_cars) + additional_cars

/-- Theorem: Given the initial conditions, the total number of cars at the end of the play is 600 -/
theorem total_cars_theorem : total_cars_at_end 100 2 300 = 600 := by
  sorry

end total_cars_theorem_l3935_393534


namespace remainder_proof_l3935_393560

theorem remainder_proof (x y : ℕ+) (r : ℕ) 
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 8 * (3 * y) + r)
  (h3 : 13 * y - x = 3) :
  r = 1 := by sorry

end remainder_proof_l3935_393560


namespace complex_fraction_simplification_l3935_393542

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 + 8 * i) / (5 - 4 * i) = (-2 : ℚ) / 41 + (64 : ℚ) / 41 * i := by sorry

end complex_fraction_simplification_l3935_393542


namespace least_four_digit_square_fourth_power_l3935_393538

theorem least_four_digit_square_fourth_power : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ a : ℕ, n = a ^ 2) ∧ 
  (∃ b : ℕ, n = b ^ 4) ∧
  (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) → (∃ c : ℕ, m = c ^ 2) → (∃ d : ℕ, m = d ^ 4) → n ≤ m) ∧
  n = 6561 :=
by sorry

end least_four_digit_square_fourth_power_l3935_393538


namespace favorite_numbers_exist_l3935_393521

theorem favorite_numbers_exist : ∃ x y : ℕ, x > y ∧ x ≠ y ∧ (x + y) + (x - y) + x * y + (x / y) = 98 := by
  sorry

end favorite_numbers_exist_l3935_393521


namespace partial_fraction_decomposition_sum_l3935_393562

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x^3 - 24*x^2 + 151*x - 650 = (x - p)*(x - q)*(x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 24*s^2 + 151*s - 650) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 251 := by
sorry

end partial_fraction_decomposition_sum_l3935_393562


namespace leastSquaresSolution_l3935_393574

-- Define the data points
def x : List ℝ := [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
def y : List ℝ := [6.01, 5.07, 4.30, 3.56, 3.07, 2.87, 2.18, 2.00, 2.14]

-- Define the quadratic model
def model (a₁ a₂ a₃ x : ℝ) : ℝ := a₁ * x^2 + a₂ * x + a₃

-- Define the sum of squared residuals
def sumSquaredResiduals (a₁ a₂ a₃ : ℝ) : ℝ :=
  List.sum (List.zipWith (λ xᵢ yᵢ => (yᵢ - model a₁ a₂ a₃ xᵢ)^2) x y)

-- State the theorem
theorem leastSquaresSolution :
  let a₁ : ℝ := 0.95586
  let a₂ : ℝ := -1.9733
  let a₃ : ℝ := 3.0684
  ∀ b₁ b₂ b₃ : ℝ, sumSquaredResiduals a₁ a₂ a₃ ≤ sumSquaredResiduals b₁ b₂ b₃ := by
  sorry

end leastSquaresSolution_l3935_393574


namespace investment_average_rate_l3935_393503

def total_investment : ℝ := 5000
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

theorem investment_average_rate :
  ∃ (x y : ℝ),
    x + y = total_investment ∧
    x * rate1 = y * rate2 / 2 ∧
    (x * rate1 + y * rate2) / total_investment = 0.041 :=
by sorry

end investment_average_rate_l3935_393503


namespace at_least_two_equations_have_solution_l3935_393526

theorem at_least_two_equations_have_solution (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let f₁ : ℝ → ℝ := λ x ↦ (x - b) * (x - c) - (x - a)
  let f₂ : ℝ → ℝ := λ x ↦ (x - c) * (x - a) - (x - b)
  let f₃ : ℝ → ℝ := λ x ↦ (x - a) * (x - b) - (x - c)
  ∃ (i j : Fin 3), i ≠ j ∧ (∃ x : ℝ, [f₁, f₂, f₃][i] x = 0) ∧ (∃ y : ℝ, [f₁, f₂, f₃][j] y = 0) :=
sorry

end at_least_two_equations_have_solution_l3935_393526


namespace q_polynomial_form_l3935_393506

def q (x : ℝ) : ℝ := sorry

theorem q_polynomial_form :
  ∀ x, q x + (2*x^6 + 5*x^4 + 10*x^2) = (9*x^4 + 30*x^3 + 40*x^2 + 5*x + 3) →
  q x = -2*x^6 + 4*x^4 + 30*x^3 + 30*x^2 + 5*x + 3 :=
by sorry

end q_polynomial_form_l3935_393506


namespace square_area_decrease_l3935_393511

theorem square_area_decrease (initial_area : ℝ) (side_decrease_percent : ℝ) 
  (h1 : initial_area = 50) 
  (h2 : side_decrease_percent = 20) : 
  let new_area := initial_area * (1 - side_decrease_percent / 100)^2
  (initial_area - new_area) / initial_area * 100 = 36 := by
sorry

end square_area_decrease_l3935_393511


namespace trigonometric_identity_l3935_393569

theorem trigonometric_identity (α : Real) :
  (Real.sin (6 * α) + Real.sin (7 * α) + Real.sin (8 * α) + Real.sin (9 * α)) /
  (Real.cos (6 * α) + Real.cos (7 * α) + Real.cos (8 * α) + Real.cos (9 * α)) =
  Real.tan ((15 * α) / 2) := by sorry

end trigonometric_identity_l3935_393569


namespace infinite_geometric_series_sum_l3935_393545

/-- The sum of the infinite geometric series 5/3 - 5/8 + 25/128 - 125/1024 + ... -/
def infiniteGeometricSeriesSum : ℚ := 8/3

/-- The first term of the geometric series -/
def firstTerm : ℚ := 5/3

/-- The common ratio of the geometric series -/
def commonRatio : ℚ := 3/8

/-- Theorem stating that the sum of the infinite geometric series is 8/3 -/
theorem infinite_geometric_series_sum :
  infiniteGeometricSeriesSum = firstTerm / (1 - commonRatio) :=
sorry

end infinite_geometric_series_sum_l3935_393545


namespace square_area_with_two_side_expressions_l3935_393548

theorem square_area_with_two_side_expressions (x : ℝ) :
  (5 * x + 10 = 35 - 2 * x) →
  ((5 * x + 10) ^ 2 : ℝ) = 38025 / 49 := by
  sorry

end square_area_with_two_side_expressions_l3935_393548


namespace permutation_sum_l3935_393507

theorem permutation_sum (n : ℕ+) (h1 : n + 3 ≤ 2*n) (h2 : n + 1 ≤ 4) : 
  (Nat.descFactorial (2*n) (n+3)) + (Nat.descFactorial 4 (n+1)) = 744 :=
by sorry

end permutation_sum_l3935_393507


namespace smallest_twin_egg_number_l3935_393501

/-- Definition of a "twin egg number" -/
def is_twin_egg (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧
  ((n / 100) % 10 = (n / 10) % 10)

/-- Function to swap digits as described -/
def swap_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  b * 1000 + a * 100 + d * 10 + c

/-- The F function as defined in the problem -/
def F (m : ℕ) : ℤ := (m - swap_digits m) / 11

/-- Main theorem statement -/
theorem smallest_twin_egg_number :
  ∀ m : ℕ,
  is_twin_egg m →
  (m / 1000 ≠ (m / 100) % 10) →
  ∃ k : ℕ, F m / 27 = k * k →
  4114 ≤ m :=
sorry

end smallest_twin_egg_number_l3935_393501


namespace hyperbola_m_range_l3935_393566

theorem hyperbola_m_range (m : ℝ) :
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m ≠ 0 ∧ m + 1 ≠ 0) ∧
   ((2 + m > 0 ∧ m + 1 < 0) ∨ (2 + m < 0 ∧ m + 1 > 0))) →
  m < -2 ∨ m > -1 :=
by sorry

end hyperbola_m_range_l3935_393566


namespace andrew_stamping_rate_l3935_393590

/-- Andrew's work schedule and permit stamping rate -/
def andrew_schedule (appointments : ℕ) (appointment_duration : ℕ) (workday_length : ℕ) (total_permits : ℕ) : ℕ :=
  let time_in_appointments := appointments * appointment_duration
  let time_stamping := workday_length - time_in_appointments
  total_permits / time_stamping

/-- Theorem stating Andrew's permit stamping rate given his schedule -/
theorem andrew_stamping_rate :
  andrew_schedule 2 3 8 100 = 50 := by
  sorry

end andrew_stamping_rate_l3935_393590
