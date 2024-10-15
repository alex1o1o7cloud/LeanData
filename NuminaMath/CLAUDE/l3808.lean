import Mathlib

namespace NUMINAMATH_CALUDE_prime_factors_of_n_smallest_prime_factors_difference_l3808_380869

def n : ℕ := 172561

-- Define a function to check if a number is prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

-- Define the prime factors of n
theorem prime_factors_of_n :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧
  p < q ∧ q < r ∧ n = p * q * r :=
sorry

-- Prove that the positive difference between the two smallest prime factors is 26
theorem smallest_prime_factors_difference :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧
  p < q ∧ q < r ∧ n = p * q * r ∧ q - p = 26 :=
sorry

end NUMINAMATH_CALUDE_prime_factors_of_n_smallest_prime_factors_difference_l3808_380869


namespace NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_16_l3808_380862

theorem three_digit_perfect_cube_divisible_by_16 : 
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 16 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_16_l3808_380862


namespace NUMINAMATH_CALUDE_logical_equivalences_l3808_380828

theorem logical_equivalences :
  (∀ A B C : Prop,
    ((A ∨ B) → C) ↔ ((A → C) ∧ (B → C))) ∧
  (∀ A B C : Prop,
    (A → (B ∧ C)) ↔ ((A → B) ∧ (A → C))) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalences_l3808_380828


namespace NUMINAMATH_CALUDE_backpack_price_l3808_380854

/-- The price of a backpack and three ring-binders, given price changes and total spent --/
theorem backpack_price (B : ℕ) : 
  (∃ (new_backpack_price new_binder_price : ℕ),
    -- Original price of each ring-binder
    20 = 20 ∧
    -- New backpack price is $5 more than original
    new_backpack_price = B + 5 ∧
    -- New ring-binder price is $2 less than original
    new_binder_price = 20 - 2 ∧
    -- Total spent is $109
    new_backpack_price + 3 * new_binder_price = 109) →
  -- Original backpack price was $50
  B = 50 := by
sorry

end NUMINAMATH_CALUDE_backpack_price_l3808_380854


namespace NUMINAMATH_CALUDE_box_width_proof_l3808_380867

/-- Given a rectangular box with length 12 cm, height 6 cm, and volume 1152 cubic cm,
    prove that the width of the box is 16 cm. -/
theorem box_width_proof (length : ℝ) (height : ℝ) (volume : ℝ) (width : ℝ) 
    (h1 : length = 12)
    (h2 : height = 6)
    (h3 : volume = 1152)
    (h4 : volume = length * width * height) :
  width = 16 := by
  sorry

end NUMINAMATH_CALUDE_box_width_proof_l3808_380867


namespace NUMINAMATH_CALUDE_part_one_part_two_l3808_380851

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : Set.Icc (-2 : ℝ) 2 = {x | f (x + 1/2) ≤ 2*m + 1}) : 
  m = 3/2 := by sorry

-- Part 2
theorem part_two : 
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧ 
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3808_380851


namespace NUMINAMATH_CALUDE_sum_remainder_l3808_380866

theorem sum_remainder (f y : ℤ) (hf : f % 5 = 3) (hy : y % 5 = 4) : (f + y) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l3808_380866


namespace NUMINAMATH_CALUDE_function_continuity_l3808_380820

-- Define a function f on the real line
variable (f : ℝ → ℝ)

-- Define the condition that f(x) + f(ax) is continuous for any a > 1
def condition (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 1 → Continuous (fun x ↦ f x + f (a * x))

-- Theorem statement
theorem function_continuity (h : condition f) : Continuous f := by
  sorry

end NUMINAMATH_CALUDE_function_continuity_l3808_380820


namespace NUMINAMATH_CALUDE_curves_intersection_equality_l3808_380825

-- Define the four curves
def C₁ (x y : ℝ) : Prop := x^2 - y^2 = x / (x^2 + y^2)
def C₂ (x y : ℝ) : Prop := 2*x*y + y / (x^2 + y^2) = 3
def C₃ (x y : ℝ) : Prop := x^3 - 3*x*y^2 + 3*y = 1
def C₄ (x y : ℝ) : Prop := 3*y*x^2 - 3*x - y^3 = 0

-- State the theorem
theorem curves_intersection_equality :
  ∀ (x y : ℝ), (C₁ x y ∧ C₂ x y) ↔ (C₃ x y ∧ C₄ x y) := by sorry

end NUMINAMATH_CALUDE_curves_intersection_equality_l3808_380825


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_squares_sum_square_infinitely_many_solutions_for_non_square_l3808_380889

-- Part a
def ConsecutiveSquaresSum (n : ℕ+) (x : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun k => (x + k) ^ 2)

def IsConsecutiveSquaresSumSquare (n : ℕ+) : Prop :=
  ∃ x k : ℕ, ConsecutiveSquaresSum n x = k ^ 2

theorem infinitely_many_consecutive_squares_sum_square :
  Set.Infinite {n : ℕ+ | IsConsecutiveSquaresSumSquare n} :=
sorry

-- Part b
theorem infinitely_many_solutions_for_non_square (n : ℕ+) (h : ¬ ∃ m : ℕ, n = m ^ 2) :
  (∃ x k : ℕ, ConsecutiveSquaresSum n x = k ^ 2) →
  Set.Infinite {y : ℕ | ∃ k : ℕ, ConsecutiveSquaresSum n y = k ^ 2} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_consecutive_squares_sum_square_infinitely_many_solutions_for_non_square_l3808_380889


namespace NUMINAMATH_CALUDE_bobbys_blocks_l3808_380840

/-- The number of blocks Bobby's father gave him -/
def blocks_from_father (initial_blocks final_blocks : ℕ) : ℕ :=
  final_blocks - initial_blocks

/-- Proof that Bobby's father gave him 6 blocks -/
theorem bobbys_blocks :
  blocks_from_father 2 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_blocks_l3808_380840


namespace NUMINAMATH_CALUDE_wall_building_time_l3808_380863

/-- Given that 8 persons can build a 140m wall in 8 days, this theorem calculates
    the number of days it takes 30 persons to build a similar 100m wall. -/
theorem wall_building_time (persons1 persons2 : ℕ) (length1 length2 : ℝ) (days1 : ℝ) : 
  persons1 = 8 →
  persons2 = 30 →
  length1 = 140 →
  length2 = 100 →
  days1 = 8 →
  ∃ days2 : ℝ, days2 = (persons1 * days1 * length2) / (persons2 * length1) :=
by sorry

end NUMINAMATH_CALUDE_wall_building_time_l3808_380863


namespace NUMINAMATH_CALUDE_average_length_of_writing_instruments_l3808_380876

theorem average_length_of_writing_instruments :
  let pen_length : ℝ := 20
  let pencil_length : ℝ := 16
  let number_of_instruments : ℕ := 2
  (pen_length + pencil_length) / number_of_instruments = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_average_length_of_writing_instruments_l3808_380876


namespace NUMINAMATH_CALUDE_smallest_m_inequality_l3808_380856

theorem smallest_m_inequality (a b c : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  ∃ (m : ℝ), (∀ (x y z : ℤ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    m * (x^3 + y^3 + z^3 : ℝ) ≥ 6 * (x^2 + y^2 + z^2 : ℝ) + 1) ∧ 
  m = 27 ∧
  ∀ (n : ℝ), (∀ (x y z : ℤ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    n * (x^3 + y^3 + z^3 : ℝ) ≥ 6 * (x^2 + y^2 + z^2 : ℝ) + 1) → n ≥ 27 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_inequality_l3808_380856


namespace NUMINAMATH_CALUDE_ammonium_nitrate_reaction_l3808_380835

-- Define the chemical species as types
structure ChemicalSpecies where
  name : String
  formula : String

-- Define the reaction
def reaction : List (ℕ × ChemicalSpecies) → List (ℕ × ChemicalSpecies) → Prop :=
  sorry

-- Define the chemical species involved
def nh4no3 : ChemicalSpecies := ⟨"Ammonium nitrate", "NH4NO3"⟩
def naoh : ChemicalSpecies := ⟨"Sodium hydroxide", "NaOH"⟩
def nano3 : ChemicalSpecies := ⟨"Sodium nitrate", "NaNO3"⟩
def nh3 : ChemicalSpecies := ⟨"Ammonia", "NH3"⟩
def h2o : ChemicalSpecies := ⟨"Water", "H2O"⟩

-- State the theorem
theorem ammonium_nitrate_reaction 
  (balanced_equation : reaction [(1, nh4no3), (1, naoh)] [(1, nano3), (1, nh3), (1, h2o)])
  (naoh_reacted : ℕ) (nano3_formed : ℕ) (nh3_formed : ℕ)
  (h1 : naoh_reacted = 3)
  (h2 : nano3_formed = 3)
  (h3 : nh3_formed = 3) :
  ∃ (nh4no3_required : ℕ) (h2o_formed : ℕ),
    nh4no3_required = 3 ∧ h2o_formed = 3 :=
  sorry

end NUMINAMATH_CALUDE_ammonium_nitrate_reaction_l3808_380835


namespace NUMINAMATH_CALUDE_pages_copied_for_15_dollars_l3808_380833

/-- The number of pages that can be copied given a certain amount of money and cost per page. -/
def pages_copied (total_money : ℚ) (cost_per_page : ℚ) : ℚ :=
  (total_money * 100) / cost_per_page

/-- Theorem stating that with $15 and a cost of 3 cents per page, 500 pages can be copied. -/
theorem pages_copied_for_15_dollars : 
  pages_copied 15 3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_15_dollars_l3808_380833


namespace NUMINAMATH_CALUDE_movie_ticket_cost_proof_l3808_380852

def movie_ticket_cost (total_money : ℚ) (change : ℚ) (num_sisters : ℕ) : ℚ :=
  (total_money - change) / num_sisters

theorem movie_ticket_cost_proof (total_money : ℚ) (change : ℚ) (num_sisters : ℕ) 
  (h1 : total_money = 25)
  (h2 : change = 9)
  (h3 : num_sisters = 2) :
  movie_ticket_cost total_money change num_sisters = 8 := by
  sorry

#eval movie_ticket_cost 25 9 2

end NUMINAMATH_CALUDE_movie_ticket_cost_proof_l3808_380852


namespace NUMINAMATH_CALUDE_roller_coaster_probability_l3808_380853

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 5

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 5

/-- The probability of riding in a specific car on a single ride -/
def prob_single_car : ℚ := 1 / num_cars

/-- The probability of riding in each of the cars over the given number of rides -/
def prob_all_cars : ℚ := (num_cars.factorial : ℚ) / num_cars ^ num_rides

theorem roller_coaster_probability :
  prob_all_cars = 24 / 625 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_probability_l3808_380853


namespace NUMINAMATH_CALUDE_cubic_quadratic_inequality_l3808_380803

theorem cubic_quadratic_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2*y + x^2*z + y^2*z + y^2*x + z^2*x + z^2*y :=
by sorry

end NUMINAMATH_CALUDE_cubic_quadratic_inequality_l3808_380803


namespace NUMINAMATH_CALUDE_lcm_gcd_product_9_10_l3808_380807

theorem lcm_gcd_product_9_10 : Nat.lcm 9 10 * Nat.gcd 9 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_9_10_l3808_380807


namespace NUMINAMATH_CALUDE_common_ratio_is_two_l3808_380899

/-- An increasing geometric sequence with specific conditions -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  is_increasing : ∀ n, a n < a (n + 1)
  is_geometric : ∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = a n * q
  a2_eq_2 : a 2 = 2
  a4_minus_a3_eq_4 : a 4 - a 3 = 4

/-- The common ratio of the increasing geometric sequence is 2 -/
theorem common_ratio_is_two (seq : IncreasingGeometricSequence) : 
  ∃ q : ℝ, (∀ n, seq.a (n + 1) = seq.a n * q) ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_common_ratio_is_two_l3808_380899


namespace NUMINAMATH_CALUDE_min_a_for_increasing_f_l3808_380814

/-- The function f(x) defined as x² + (a-2)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 2) * x - 1

/-- The property that f is increasing on the interval [2, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → x < y → f a x < f a y

/-- The theorem stating the minimum value of a for which f is increasing on [2, +∞) -/
theorem min_a_for_increasing_f :
  (∃ a_min : ℝ, (∀ a : ℝ, is_increasing_on_interval a ↔ a_min ≤ a) ∧ a_min = -2) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_increasing_f_l3808_380814


namespace NUMINAMATH_CALUDE_inequality_proof_l3808_380896

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  let p := x + y + z
  let q := x*y + y*z + z*x
  let r := x*y*z
  (p^2 ≥ 3*q) ∧
  (p^3 ≥ 27*r) ∧
  (p*q ≥ 9*r) ∧
  (q^2 ≥ 3*p*r) ∧
  (p^2*q + 3*p*r ≥ 4*q^2) ∧
  (p^3 + 9*r ≥ 4*p*q) ∧
  (p*q^2 ≥ 2*p^2*r + 3*q*r) ∧
  (p*q^2 + 3*q*r ≥ 4*p^2*r) ∧
  (2*q^3 + 9*r^2 ≥ 7*p*q*r) ∧
  (p^4 + 4*q^2 + 6*p*r ≥ 5*p^2*q) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3808_380896


namespace NUMINAMATH_CALUDE_water_bucket_addition_l3808_380805

theorem water_bucket_addition (initial_water : ℝ) (added_water : ℝ) :
  initial_water = 3 → added_water = 6.8 → initial_water + added_water = 9.8 :=
by
  sorry

end NUMINAMATH_CALUDE_water_bucket_addition_l3808_380805


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3808_380824

theorem no_real_roots_quadratic (m : ℝ) : 
  (∀ x : ℝ, -2 * x^2 + 6 * x + m ≠ 0) → m < -4.5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3808_380824


namespace NUMINAMATH_CALUDE_chemistry_class_size_l3808_380826

theorem chemistry_class_size :
  -- Total number of students
  let total : ℕ := 120
  -- Students in both chemistry and biology
  let chem_bio : ℕ := 35
  -- Students in both biology and physics
  let bio_phys : ℕ := 15
  -- Students in both chemistry and physics
  let chem_phys : ℕ := 10
  -- Function to calculate total students in a class
  let class_size (only : ℕ) (with_other1 : ℕ) (with_other2 : ℕ) := only + with_other1 + with_other2
  -- Constraint: Chemistry class is four times as large as biology class
  ∀ (bio_only : ℕ) (chem_only : ℕ) (phys_only : ℕ),
    class_size chem_only chem_bio chem_phys = 4 * class_size bio_only chem_bio bio_phys →
    -- Constraint: No student takes all three classes
    class_size bio_only chem_bio bio_phys + class_size chem_only chem_bio chem_phys + class_size phys_only bio_phys chem_phys = total →
    -- Conclusion: Chemistry class size is 198
    class_size chem_only chem_bio chem_phys = 198 :=
by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_size_l3808_380826


namespace NUMINAMATH_CALUDE_proportion_with_one_half_one_third_l3808_380897

def forms_proportion (a b c d : ℚ) : Prop := a / b = c / d

theorem proportion_with_one_half_one_third :
  forms_proportion (1/2) (1/3) 3 2 ∧
  ¬forms_proportion (1/2) (1/3) 5 4 ∧
  ¬forms_proportion (1/2) (1/3) (1/3) (1/4) ∧
  ¬forms_proportion (1/2) (1/3) (1/3) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_proportion_with_one_half_one_third_l3808_380897


namespace NUMINAMATH_CALUDE_triangle_property_l3808_380875

theorem triangle_property (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_condition : a^3 - b^3 = a^2*b - a*b^2 + a*c^2 - b*c^2) : 
  a = b ∨ a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_l3808_380875


namespace NUMINAMATH_CALUDE_function_derivative_value_l3808_380883

/-- Given a function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem function_derivative_value (a : ℝ) : 
  let f := λ x : ℝ => a * x^3 + 3 * x^2 + 2
  let f' := λ x : ℝ => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by sorry

end NUMINAMATH_CALUDE_function_derivative_value_l3808_380883


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3808_380846

/-- A geometric sequence with a_3 = 2 and a_6 = 16 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1))
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 16) : 
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3808_380846


namespace NUMINAMATH_CALUDE_raja_monthly_savings_l3808_380850

/-- Raja's monthly savings calculation --/
theorem raja_monthly_savings :
  let monthly_income : ℝ := 24999.999999999993
  let household_percentage : ℝ := 0.60
  let clothes_percentage : ℝ := 0.10
  let medicines_percentage : ℝ := 0.10
  let total_spent_percentage : ℝ := household_percentage + clothes_percentage + medicines_percentage
  let savings_percentage : ℝ := 1 - total_spent_percentage
  let savings : ℝ := savings_percentage * monthly_income
  ⌊savings⌋ = 5000 := by sorry

end NUMINAMATH_CALUDE_raja_monthly_savings_l3808_380850


namespace NUMINAMATH_CALUDE_division_problem_l3808_380882

theorem division_problem : (120 : ℚ) / ((6 / 2) + 4) = 17 + 1/7 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3808_380882


namespace NUMINAMATH_CALUDE_diana_biking_time_l3808_380857

def total_distance : ℝ := 10
def initial_speed : ℝ := 3
def initial_duration : ℝ := 2
def tired_speed : ℝ := 1

theorem diana_biking_time : 
  let initial_distance := initial_speed * initial_duration
  let remaining_distance := total_distance - initial_distance
  let tired_duration := remaining_distance / tired_speed
  initial_duration + tired_duration = 6 := by sorry

end NUMINAMATH_CALUDE_diana_biking_time_l3808_380857


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3808_380844

theorem arithmetic_calculations :
  ((294.4 - 19.2 * 6) / (6 + 8) = 12.8) ∧
  (12.5 * 0.4 * 8 * 2.5 = 100) ∧
  (333 * 334 + 999 * 222 = 333000) ∧
  (999 + 99.9 + 9.99 + 0.999 = 1109.889) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3808_380844


namespace NUMINAMATH_CALUDE_smallest_number_of_oranges_l3808_380849

/-- Represents the number of oranges in a container --/
def container_capacity : ℕ := 15

/-- Represents the number of containers that are not full --/
def short_containers : ℕ := 3

/-- Represents the number of oranges missing from each short container --/
def missing_oranges : ℕ := 2

/-- Represents the minimum number of oranges --/
def min_oranges : ℕ := 201

theorem smallest_number_of_oranges (n : ℕ) : 
  n * container_capacity - short_containers * missing_oranges > min_oranges →
  ∃ (m : ℕ), m * container_capacity - short_containers * missing_oranges > min_oranges ∧
             m * container_capacity - short_containers * missing_oranges ≤ 
             n * container_capacity - short_containers * missing_oranges →
  n * container_capacity - short_containers * missing_oranges ≥ 204 :=
by sorry

#check smallest_number_of_oranges

end NUMINAMATH_CALUDE_smallest_number_of_oranges_l3808_380849


namespace NUMINAMATH_CALUDE_oxen_grazing_problem_l3808_380813

theorem oxen_grazing_problem (total_rent : ℕ) (a_months b_oxen b_months c_oxen c_months : ℕ) (c_share : ℕ) :
  total_rent = 175 →
  a_months = 7 →
  b_oxen = 12 →
  b_months = 5 →
  c_oxen = 15 →
  c_months = 3 →
  c_share = 45 →
  ∃ a_oxen : ℕ, a_oxen * a_months + b_oxen * b_months + c_oxen * c_months = total_rent ∧ a_oxen = 10 := by
  sorry


end NUMINAMATH_CALUDE_oxen_grazing_problem_l3808_380813


namespace NUMINAMATH_CALUDE_max_area_semicircle_l3808_380895

/-- A semicircle with diameter AB and radius R -/
structure Semicircle where
  R : ℝ
  A : Point
  B : Point

/-- Points C and D on the semicircle -/
structure PointsOnSemicircle (S : Semicircle) where
  C : Point
  D : Point

/-- The area of quadrilateral ACDB -/
def area (S : Semicircle) (P : PointsOnSemicircle S) : ℝ :=
  sorry

/-- C and D divide the semicircle into three equal parts -/
def equalParts (S : Semicircle) (P : PointsOnSemicircle S) : Prop :=
  sorry

theorem max_area_semicircle (S : Semicircle) :
  ∃ (P : PointsOnSemicircle S),
    equalParts S P ∧
    ∀ (Q : PointsOnSemicircle S), area S Q ≤ area S P ∧
    area S P = (3 * Real.sqrt 3 / 4) * S.R^2 :=
  sorry

end NUMINAMATH_CALUDE_max_area_semicircle_l3808_380895


namespace NUMINAMATH_CALUDE_village_population_l3808_380822

theorem village_population (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.8 = 4554 → P = 6325 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3808_380822


namespace NUMINAMATH_CALUDE_unique_valid_number_l3808_380865

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a > b ∧ b > c ∧
    (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = n

theorem unique_valid_number :
  ∃! n, is_valid_number n ∧ n = 495 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3808_380865


namespace NUMINAMATH_CALUDE_gcf_36_45_l3808_380886

theorem gcf_36_45 : Nat.gcd 36 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_45_l3808_380886


namespace NUMINAMATH_CALUDE_sweet_potato_harvest_l3808_380861

theorem sweet_potato_harvest (sold_to_adams : ℕ) (sold_to_lenon : ℕ) (not_sold : ℕ) :
  sold_to_adams = 20 →
  sold_to_lenon = 15 →
  not_sold = 45 →
  sold_to_adams + sold_to_lenon + not_sold = 80 :=
by sorry

end NUMINAMATH_CALUDE_sweet_potato_harvest_l3808_380861


namespace NUMINAMATH_CALUDE_sum_mod_nine_l3808_380873

theorem sum_mod_nine : (88135 + 88136 + 88137 + 88138 + 88139 + 88140) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l3808_380873


namespace NUMINAMATH_CALUDE_unique_c_value_l3808_380880

theorem unique_c_value (c : ℝ) : c ≠ 0 ∧
  (∃! (b₁ b₂ b₃ : ℝ), b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0 ∧ b₁ ≠ b₂ ∧ b₂ ≠ b₃ ∧ b₁ ≠ b₃ ∧
    (∀ x : ℝ, x^2 + 2*(b₁ + 1/b₁)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₁ + 1/b₁)*y + c = 0) ∧
    (∀ x : ℝ, x^2 + 2*(b₂ + 1/b₂)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₂ + 1/b₂)*y + c = 0) ∧
    (∀ x : ℝ, x^2 + 2*(b₃ + 1/b₃)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₃ + 1/b₃)*y + c = 0)) →
  c = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_c_value_l3808_380880


namespace NUMINAMATH_CALUDE_system_solution_existence_l3808_380839

theorem system_solution_existence (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ |x| + |y| = |b|) ↔ |a| ≤ |b| ∧ |b| ≤ Real.sqrt 2 * |a| :=
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3808_380839


namespace NUMINAMATH_CALUDE_complex_number_simplification_l3808_380811

theorem complex_number_simplification :
  (7 - 3*Complex.I) - 3*(2 - 5*Complex.I) + 4*Complex.I = 1 + 16*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l3808_380811


namespace NUMINAMATH_CALUDE_hemisphere_container_volume_l3808_380874

/-- Given a total volume of water and the number of hemisphere containers needed,
    calculate the volume of each hemisphere container. -/
theorem hemisphere_container_volume
  (total_volume : ℝ)
  (num_containers : ℕ)
  (h_total_volume : total_volume = 10976)
  (h_num_containers : num_containers = 2744) :
  total_volume / num_containers = 4 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_container_volume_l3808_380874


namespace NUMINAMATH_CALUDE_floor_of_e_equals_two_l3808_380848

noncomputable def e : ℝ := Real.exp 1

theorem floor_of_e_equals_two : ⌊e⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_equals_two_l3808_380848


namespace NUMINAMATH_CALUDE_inequality_range_l3808_380864

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3808_380864


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3808_380855

theorem gcd_of_three_numbers (A B C : ℕ+) 
  (h_lcm : Nat.lcm A.val (Nat.lcm B.val C.val) = 1540)
  (h_prod : A.val * B.val * C.val = 1230000) :
  Nat.gcd A.val (Nat.gcd B.val C.val) = 20 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3808_380855


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l3808_380878

-- Define an odd function that is monotonically decreasing on [-1, 0]
def is_odd_and_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x > f y)

-- Define acute angles of a triangle
def is_acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

-- Theorem statement
theorem sin_cos_inequality 
  (f : ℝ → ℝ) 
  (A B : ℝ) 
  (h_f : is_odd_and_decreasing f) 
  (h_A : is_acute_angle A) 
  (h_B : is_acute_angle B) : 
  f (Real.sin A) < f (Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l3808_380878


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l3808_380832

def ellipse (b : ℝ) (x y : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

theorem max_value_on_ellipse (b : ℝ) (h : b > 0) :
  (∃ (x y : ℝ), ellipse b x y ∧ 
    ∀ (x' y' : ℝ), ellipse b x' y' → x^2 + 2*y ≥ x'^2 + 2*y') ∧
  (∃ (max : ℝ), 
    (0 < b ∧ b ≤ 4 → max = b^2/4 + 4) ∧
    (b > 4 → max = 2*b) ∧
    ∀ (x y : ℝ), ellipse b x y → x^2 + 2*y ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l3808_380832


namespace NUMINAMATH_CALUDE_no_real_roots_composite_l3808_380823

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem no_real_roots_composite (b c : ℝ) :
  (∀ x : ℝ, f b c x ≠ x) →
  (∀ x : ℝ, f b c (f b c x) ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_composite_l3808_380823


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_one_l3808_380898

theorem no_solution_implies_m_equals_one (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (x - 3) / (x - 2) ≠ m / (2 - x)) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_one_l3808_380898


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3808_380860

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 5}
def B : Set Nat := {2, 3, 5}

theorem complement_intersection_theorem :
  (U \ B) ∩ A = {1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3808_380860


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l3808_380871

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

-- Define the point
def p : ℝ × ℝ := (0, 1)

-- Define the slope of the tangent line
def m : ℝ := 3

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 3*x - y + 1 = 0

-- Theorem statement
theorem tangent_line_at_point :
  tangent_line p.1 p.2 ∧
  ∀ x y, tangent_line x y ↔ y - f p.1 = m * (x - p.1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l3808_380871


namespace NUMINAMATH_CALUDE_typing_time_together_l3808_380812

/-- Given Meso's and Tyler's typing speeds, calculate the time it takes them to type 40 pages together -/
theorem typing_time_together 
  (meso_pages : ℕ) (meso_time : ℕ) (tyler_pages : ℕ) (tyler_time : ℕ) (total_pages : ℕ) :
  meso_pages = 15 →
  meso_time = 5 →
  tyler_pages = 15 →
  tyler_time = 3 →
  total_pages = 40 →
  (total_pages : ℚ) / ((meso_pages : ℚ) / (meso_time : ℚ) + (tyler_pages : ℚ) / (tyler_time : ℚ)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_together_l3808_380812


namespace NUMINAMATH_CALUDE_bobik_distance_l3808_380836

/-- The problem of Seryozha, Valera, and Bobik's movement --/
theorem bobik_distance (distance : ℝ) (speed_seryozha speed_valera speed_bobik : ℝ) :
  distance = 21 →
  speed_seryozha = 4 →
  speed_valera = 3 →
  speed_bobik = 11 →
  speed_bobik * (distance / (speed_seryozha + speed_valera)) = 33 :=
by sorry

end NUMINAMATH_CALUDE_bobik_distance_l3808_380836


namespace NUMINAMATH_CALUDE_five_students_two_groups_l3808_380806

/-- The number of ways to assign n students to k groups, where each student
    must be assigned to exactly one group. -/
def assignmentWays (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to assign 5 students to 2 groups is 32. -/
theorem five_students_two_groups : assignmentWays 5 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_five_students_two_groups_l3808_380806


namespace NUMINAMATH_CALUDE_number_value_l3808_380815

theorem number_value (x number : ℝ) 
  (h1 : (x + 5) * (number - 5) = 0)
  (h2 : ∀ y z : ℝ, (y + 5) * (z - 5) = 0 → x^2 + number^2 ≤ y^2 + z^2) :
  number = 5 := by
sorry

end NUMINAMATH_CALUDE_number_value_l3808_380815


namespace NUMINAMATH_CALUDE_inequality_proof_l3808_380830

theorem inequality_proof (n : ℕ) (h : n > 1) : 1 + n * 2^((n-1)/2) < 2^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3808_380830


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_AFCH_l3808_380816

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the cross-shaped figure formed by two intersecting rectangles -/
structure CrossFigure where
  rect1 : Rectangle
  rect2 : Rectangle

/-- Theorem: Area of quadrilateral AFCH in the cross-shaped figure -/
theorem area_of_quadrilateral_AFCH (cf : CrossFigure)
  (h1 : cf.rect1.width = 9)
  (h2 : cf.rect1.height = 5)
  (h3 : cf.rect2.width = 3)
  (h4 : cf.rect2.height = 10) :
  area (Rectangle.mk 9 10) - (area cf.rect1 + area cf.rect2 - area (Rectangle.mk 3 5)) / 2 = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_AFCH_l3808_380816


namespace NUMINAMATH_CALUDE_special_number_theorem_l3808_380819

/-- The type of positive integers with at least seven divisors -/
def HasAtLeastSevenDivisors (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ d₄ < d₅ ∧ d₅ < d₆ ∧ d₆ < d₇ ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧ d₅ ∣ n ∧ d₆ ∣ n ∧ d₇ ∣ n

/-- The property that n + 1 is equal to the sum of squares of its 6th and 7th divisors -/
def SumOfSquaresProperty (n : ℕ) : Prop :=
  ∃ (d₆ d₇ : ℕ), d₆ < d₇ ∧ d₆ ∣ n ∧ d₇ ∣ n ∧
    (∀ d : ℕ, d ∣ n → d < d₆ ∨ d = d₆ ∨ d = d₇ ∨ d₇ < d) ∧
    n + 1 = d₆^2 + d₇^2

theorem special_number_theorem (n : ℕ) 
  (h1 : HasAtLeastSevenDivisors n)
  (h2 : SumOfSquaresProperty n) :
  n = 144 ∨ n = 1984 :=
sorry

end NUMINAMATH_CALUDE_special_number_theorem_l3808_380819


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l3808_380894

theorem complex_roots_on_circle :
  ∀ (z : ℂ), (z + 2)^6 = 64 * z^6 →
  Complex.abs (z + 2/3) = 2/3 := by sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l3808_380894


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l3808_380842

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f) (h_periodic : is_periodic_2 f) (h_value : f (1 + a) = 1) :
  f (1 - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l3808_380842


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_ratio_l3808_380877

/-- An equilateral triangle with an inscribed circle -/
structure EquilateralTriangleWithInscribedCircle where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- The radius of the inscribed circle -/
  circle_radius : ℝ
  /-- The points of tangency are on the sides of the triangle -/
  tangency_points_on_sides : True

/-- The ratio of the inscribed circle's radius to the triangle's side length is 1/16 -/
theorem inscribed_circle_radius_ratio 
  (triangle : EquilateralTriangleWithInscribedCircle) : 
  triangle.circle_radius / triangle.side_length = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_ratio_l3808_380877


namespace NUMINAMATH_CALUDE_relationship_abc_l3808_380890

theorem relationship_abc (a b c : ℝ) : 
  a = (1/2)^(2/3) → b = (1/5)^(2/3) → c = (1/2)^(1/3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3808_380890


namespace NUMINAMATH_CALUDE_f_g_zero_range_l3808_380804

def f (x : ℝ) : ℝ := sorry

def g (x : ℝ) : ℝ := f x

theorem f_g_zero_range (π : ℝ) (h_π : π > 0) :
  (∀ x ∈ Set.Icc (1 / π) π, f x = f (1 / x)) →
  (∀ x ∈ Set.Icc (1 / π) 1, f x = Real.log x) →
  (∃ x ∈ Set.Icc (1 / π) π, g x = 0) →
  Set.Icc (-π * Real.log π) 0 = {a | g a = 0} := by sorry

end NUMINAMATH_CALUDE_f_g_zero_range_l3808_380804


namespace NUMINAMATH_CALUDE_fruit_display_total_l3808_380834

/-- The number of bananas on the display -/
def num_bananas : ℕ := 5

/-- The number of oranges on the display -/
def num_oranges : ℕ := 2 * num_bananas

/-- The number of apples on the display -/
def num_apples : ℕ := 2 * num_oranges

/-- The total number of fruits on the display -/
def total_fruits : ℕ := num_bananas + num_oranges + num_apples

theorem fruit_display_total :
  total_fruits = 35 :=
by sorry

end NUMINAMATH_CALUDE_fruit_display_total_l3808_380834


namespace NUMINAMATH_CALUDE_square_division_perimeter_l3808_380800

theorem square_division_perimeter (s : ℝ) (h1 : s > 0) : 
  let square_perimeter := 4 * s
  let rectangle_length := s
  let rectangle_width := s / 2
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  square_perimeter = 200 → rectangle_perimeter = 150 := by
  sorry

end NUMINAMATH_CALUDE_square_division_perimeter_l3808_380800


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l3808_380888

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_divisibility (m n : ℕ) (h1 : m ≥ 1) (h2 : n > 1) :
  ∃ k : ℕ, fib (m * n - 1) - (fib (n - 1))^m = k * (fib n)^2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l3808_380888


namespace NUMINAMATH_CALUDE_number_difference_l3808_380838

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 125000)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100)
  (a_div_5 : 5 ∣ a)
  (b_div_5 : 5 ∣ b) :
  b - a = 122265 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3808_380838


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3808_380808

/-- Proves that for an arithmetic sequence with given properties, m = 2 when S_m is the arithmetic mean of a_m and a_{m+1} -/
theorem arithmetic_sequence_problem (m : ℕ) : 
  let a : ℕ → ℤ := λ n => 2*n - 1
  let S : ℕ → ℤ := λ n => n^2
  (S m = (a m + a (m+1)) / 2) → m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3808_380808


namespace NUMINAMATH_CALUDE_shortest_side_in_triangle_l3808_380817

theorem shortest_side_in_triangle (A B C : Real) (a b c : Real) :
  B = 45 * π / 180 →  -- Convert 45° to radians
  C = 60 * π / 180 →  -- Convert 60° to radians
  c = 1 →
  A + B + C = π →     -- Sum of angles in a triangle
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  b / Real.sin B = c / Real.sin C →  -- Law of Sines
  b < a ∧ b < c →     -- b is the shortest side
  b = Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_shortest_side_in_triangle_l3808_380817


namespace NUMINAMATH_CALUDE_arithmetic_progression_contains_10_start_l3808_380872

/-- An infinite increasing arithmetic progression of natural numbers contains a number starting with 10 -/
theorem arithmetic_progression_contains_10_start (a d : ℕ) (h : 0 < d) :
  ∃ k : ℕ, ∃ m : ℕ, (a + k * d) = 10 * 10^m + (a + k * d - 10 * 10^m) ∧ 
    10 * 10^m ≤ (a + k * d) ∧ (a + k * d) < 11 * 10^m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_contains_10_start_l3808_380872


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l3808_380870

/-- Given that Bryan has a total number of books and each bookshelf contains a fixed number of books,
    calculate the number of bookshelves he has. -/
def calculate_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

/-- Prove that Bryan has 19 bookshelves given the conditions. -/
theorem bryan_bookshelves :
  calculate_bookshelves 38 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l3808_380870


namespace NUMINAMATH_CALUDE_marathon_positions_l3808_380841

/-- Represents a marathon with participants -/
structure Marathon where
  total_participants : ℕ
  john_from_right : ℕ
  john_from_left : ℕ
  mike_ahead : ℕ

/-- Theorem about the marathon positions -/
theorem marathon_positions (m : Marathon) 
  (h1 : m.john_from_right = 28)
  (h2 : m.john_from_left = 42)
  (h3 : m.mike_ahead = 10) :
  m.total_participants = 69 ∧ 
  m.john_from_left - m.mike_ahead = 32 ∧ 
  m.john_from_right - m.mike_ahead = 18 := by
  sorry


end NUMINAMATH_CALUDE_marathon_positions_l3808_380841


namespace NUMINAMATH_CALUDE_jamie_quiz_score_l3808_380858

def school_quiz (total_questions correct_answers incorrect_answers unanswered_questions : ℕ)
  (points_correct points_incorrect points_unanswered : ℚ) : Prop :=
  total_questions = correct_answers + incorrect_answers + unanswered_questions ∧
  (correct_answers : ℚ) * points_correct +
  (incorrect_answers : ℚ) * points_incorrect +
  (unanswered_questions : ℚ) * points_unanswered = 28

theorem jamie_quiz_score :
  school_quiz 30 16 10 4 2 (-1/2) (1/4) :=
by sorry

end NUMINAMATH_CALUDE_jamie_quiz_score_l3808_380858


namespace NUMINAMATH_CALUDE_prize_orders_count_l3808_380868

/-- Represents the number of bowlers in the tournament -/
def num_bowlers : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := 5

/-- Calculates the number of possible outcomes for a single game -/
def outcomes_per_game : ℕ := 2

/-- Calculates the total number of possible prize orders -/
def total_prize_orders : ℕ := outcomes_per_game ^ num_games

/-- Theorem stating that the total number of possible prize orders is 32 -/
theorem prize_orders_count : total_prize_orders = 32 := by
  sorry


end NUMINAMATH_CALUDE_prize_orders_count_l3808_380868


namespace NUMINAMATH_CALUDE_triangle_ratio_greater_than_two_l3808_380818

/-- In a right triangle ABC with ∠BAC = 90°, AB = 5, BC = 6, and point K dividing AC in ratio 3:1 from A,
    the ratio BK/AH is greater than 2, where AH is the altitude from A to BC. -/
theorem triangle_ratio_greater_than_two (A B C K H : ℝ × ℝ) : 
  -- Triangle ABC is right-angled at A
  (A.1 = 0 ∧ A.2 = 0) → 
  (B.1 = 5 ∧ B.2 = 0) → 
  (C.1 = 0 ∧ C.2 = 6) → 
  -- K divides AC in ratio 3:1 from A
  (K.1 = (3/4) * C.1 ∧ K.2 = (3/4) * C.2) →
  -- H is the foot of the altitude from A to BC
  (H.1 = 0 ∧ H.2 = 30 / Real.sqrt 61) →
  -- The ratio BK/AH is greater than 2
  Real.sqrt ((K.1 - B.1)^2 + (K.2 - B.2)^2) / Real.sqrt (H.1^2 + H.2^2) > 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_greater_than_two_l3808_380818


namespace NUMINAMATH_CALUDE_company_size_l3808_380891

/-- Represents the number of employees in a company -/
structure Company where
  total : ℕ
  senior : ℕ
  sample_size : ℕ
  sample_senior : ℕ

/-- Given a company with 15 senior-titled employees and a stratified sample of 30 employees
    containing 3 senior-titled employees, the total number of employees is 150 -/
theorem company_size (c : Company)
  (h1 : c.senior = 15)
  (h2 : c.sample_size = 30)
  (h3 : c.sample_senior = 3)
  : c.total = 150 := by
  sorry

end NUMINAMATH_CALUDE_company_size_l3808_380891


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_consecutive_list_l3808_380881

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_consecutive_list :
  let D := consecutive_integers (-4) 12
  let positives := positive_integers D
  range positives = 6 := by sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_consecutive_list_l3808_380881


namespace NUMINAMATH_CALUDE_clock_angle_theorem_l3808_380810

/-- The angle in radians through which the minute hand of a clock turns from 1:00 to 3:20 -/
def clock_angle_radians : ℝ := sorry

/-- The angle in degrees that the minute hand turns per minute -/
def minute_hand_degrees_per_minute : ℝ := 6

/-- The time difference in minutes from 1:00 to 3:20 -/
def time_difference_minutes : ℕ := 2 * 60 + 20

theorem clock_angle_theorem : 
  clock_angle_radians = -(minute_hand_degrees_per_minute * time_difference_minutes * (π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_theorem_l3808_380810


namespace NUMINAMATH_CALUDE_g_max_min_l3808_380827

noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 8 + 8 * Real.cos x ^ 8

theorem g_max_min :
  (∀ x, g x ≤ 8) ∧ (∃ x, g x = 8) ∧ (∀ x, 8/27 ≤ g x) ∧ (∃ x, g x = 8/27) :=
sorry

end NUMINAMATH_CALUDE_g_max_min_l3808_380827


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3808_380847

theorem system_of_equations_solution :
  ∀ s t : ℝ,
  (11 * s + 7 * t = 240) →
  (s = (1/2) * t + 3) →
  (t = 414/25 ∧ s = 11.28) :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3808_380847


namespace NUMINAMATH_CALUDE_stating_min_weighings_to_find_lighter_ball_l3808_380893

/-- Represents the number of balls -/
def num_balls : ℕ := 9

/-- Represents the number of heavier balls -/
def num_heavy : ℕ := 8

/-- Represents the weight of the heavier balls in grams -/
def heavy_weight : ℕ := 10

/-- Represents the weight of the lighter ball in grams -/
def light_weight : ℕ := 9

/-- Represents the availability of a balance scale -/
def has_balance_scale : Prop := True

/-- 
Theorem stating that the minimum number of weighings required to find the lighter ball is 2
given the conditions of the problem.
-/
theorem min_weighings_to_find_lighter_ball :
  ∀ (balls : Fin num_balls → ℕ),
  (∃ (i : Fin num_balls), balls i = light_weight) ∧
  (∀ (i : Fin num_balls), balls i = light_weight ∨ balls i = heavy_weight) ∧
  has_balance_scale →
  (∃ (n : ℕ), n = 2 ∧ 
    ∀ (m : ℕ), (∃ (strategy : ℕ → ℕ → Bool), 
      (∀ (i : Fin num_balls), balls i = light_weight → 
        ∃ (k : Fin m), strategy k (balls i) = true) ∧
      (∀ (i j : Fin num_balls), i ≠ j → balls i ≠ balls j → 
        ∃ (k : Fin m), strategy k (balls i) ≠ strategy k (balls j))) → 
    m ≥ n) :=
sorry

end NUMINAMATH_CALUDE_stating_min_weighings_to_find_lighter_ball_l3808_380893


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3808_380845

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(9 ∣ (51234 + m))) ∧ (9 ∣ (51234 + n)) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3808_380845


namespace NUMINAMATH_CALUDE_segment_length_l3808_380801

theorem segment_length : 
  let endpoints := {x : ℝ | |x - (27 : ℝ)^(1/3)| = 5}
  ∃ a b : ℝ, a ∈ endpoints ∧ b ∈ endpoints ∧ |a - b| = 10 :=
by sorry

end NUMINAMATH_CALUDE_segment_length_l3808_380801


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3808_380892

def smallest_three_digit_multiple_of_5 : ℕ := 100

def smallest_four_digit_multiple_of_7 : ℕ := 1001

theorem sum_of_multiples : 
  smallest_three_digit_multiple_of_5 + smallest_four_digit_multiple_of_7 = 1101 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3808_380892


namespace NUMINAMATH_CALUDE_symmetry_implies_m_and_n_l3808_380809

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and their y-coordinates are opposites -/
def symmetric_about_x_axis (a b : ℝ × ℝ) : Prop :=
  a.1 = b.1 ∧ a.2 = -b.2

/-- The theorem stating that if A(-4, m-3) and B(2n, 1) are symmetric about the x-axis, then m = 2 and n = -2 -/
theorem symmetry_implies_m_and_n (m n : ℝ) :
  symmetric_about_x_axis (-4, m - 3) (2*n, 1) → m = 2 ∧ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_m_and_n_l3808_380809


namespace NUMINAMATH_CALUDE_seq1_infinitely_many_composites_seq2_infinitely_many_composites_l3808_380829

-- Define the first sequence
def seq1 (n : ℕ) : ℕ :=
  3^n * 10^n + 7

-- Define the second sequence
def seq2 (n : ℕ) : ℕ :=
  3^n * 10^n + 31

-- Statement for the first sequence
theorem seq1_infinitely_many_composites :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ¬ Nat.Prime (seq1 n) :=
sorry

-- Statement for the second sequence
theorem seq2_infinitely_many_composites :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ¬ Nat.Prime (seq2 n) :=
sorry

end NUMINAMATH_CALUDE_seq1_infinitely_many_composites_seq2_infinitely_many_composites_l3808_380829


namespace NUMINAMATH_CALUDE_intersection_M_N_l3808_380837

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3808_380837


namespace NUMINAMATH_CALUDE_unique_m_solution_l3808_380885

def S (n : ℕ) : ℕ := n^2

def a (n : ℕ) : ℕ := 2*n - 1

theorem unique_m_solution :
  ∃! m : ℕ+, 
    (∀ n : ℕ, S n = n^2) ∧ 
    (S m = (a m.val + a (m.val + 1)) / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_m_solution_l3808_380885


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l3808_380879

/-- The function f(x) = x^2/2 - a*ln(x) is monotonic on [1,2] if and only if a ∈ (0, 1] ∪ [4, +∞) -/
theorem monotonic_function_a_range (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => x^2 / 2 - a * Real.log x)) ↔ 
  a ∈ Set.Ioo 0 1 ∪ Set.Iic 1 ∪ Set.Ici 4 := by
  sorry


end NUMINAMATH_CALUDE_monotonic_function_a_range_l3808_380879


namespace NUMINAMATH_CALUDE_max_product_sum_200_l3808_380859

theorem max_product_sum_200 : 
  ∀ x y : ℤ, x + y = 200 → x * y ≤ 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_200_l3808_380859


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3808_380821

theorem regular_polygon_interior_angle_sum (exterior_angle : ℝ) : 
  exterior_angle = 72 → (360 / exterior_angle - 2) * 180 = 540 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3808_380821


namespace NUMINAMATH_CALUDE_sequence_sum_l3808_380831

theorem sequence_sum (a b c d : ℕ) (h1 : 0 < a ∧ a < b ∧ b < c ∧ c < d)
  (h2 : b - a = c - b) (h3 : c * c = b * d) (h4 : d - a = 30) :
  a + b + c + d = 129 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3808_380831


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3808_380887

theorem arithmetic_equality : 239 - 27 + 45 + 33 - 11 = 279 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3808_380887


namespace NUMINAMATH_CALUDE_perimeter_division_ratio_l3808_380802

/-- A square with a point on its diagonal and a line passing through that point. -/
structure SquareWithLine where
  /-- Side length of the square -/
  a : ℝ
  /-- Point M divides diagonal AC in ratio 2:1 -/
  m : ℝ × ℝ
  /-- The line divides the square's area in ratio 9:31 -/
  areaRatio : ℝ × ℝ
  /-- Conditions -/
  h1 : a > 0
  h2 : m = (2*a/3, 2*a/3)
  h3 : areaRatio = (9, 31)

/-- The theorem to be proved -/
theorem perimeter_division_ratio (s : SquareWithLine) :
  let p1 := (9 : ℝ) / 10 * (4 * s.a)
  let p2 := (31 : ℝ) / 10 * (4 * s.a)
  (p1, p2) = (9, 31) := by sorry

end NUMINAMATH_CALUDE_perimeter_division_ratio_l3808_380802


namespace NUMINAMATH_CALUDE_balance_difference_approx_l3808_380843

def angela_deposit : ℝ := 9000
def bob_deposit : ℝ := 11000
def angela_rate : ℝ := 0.08
def bob_rate : ℝ := 0.09
def years : ℕ := 25

def angela_balance : ℝ := angela_deposit * (1 + angela_rate) ^ years
def bob_balance : ℝ := bob_deposit * (1 + bob_rate * years)

theorem balance_difference_approx :
  ‖angela_balance - bob_balance - 25890‖ < 1 := by sorry

end NUMINAMATH_CALUDE_balance_difference_approx_l3808_380843


namespace NUMINAMATH_CALUDE_dinner_tasks_is_four_l3808_380884

/-- Represents Trey's chore list for Sunday -/
structure ChoreList where
  clean_house_tasks : Nat
  shower_tasks : Nat
  dinner_tasks : Nat
  time_per_task : Nat
  total_time : Nat

/-- Calculates the number of dinner tasks given the chore list -/
def calculate_dinner_tasks (chores : ChoreList) : Nat :=
  (chores.total_time - (chores.clean_house_tasks + chores.shower_tasks) * chores.time_per_task) / chores.time_per_task

/-- Theorem stating that the number of dinner tasks is 4 -/
theorem dinner_tasks_is_four (chores : ChoreList) 
  (h1 : chores.clean_house_tasks = 7)
  (h2 : chores.shower_tasks = 1)
  (h3 : chores.time_per_task = 10)
  (h4 : chores.total_time = 120) :
  calculate_dinner_tasks chores = 4 := by
  sorry

#eval calculate_dinner_tasks { clean_house_tasks := 7, shower_tasks := 1, dinner_tasks := 0, time_per_task := 10, total_time := 120 }

end NUMINAMATH_CALUDE_dinner_tasks_is_four_l3808_380884
