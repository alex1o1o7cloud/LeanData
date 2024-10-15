import Mathlib

namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2884_288466

theorem fraction_product_simplification :
  (20 : ℚ) / 21 * 35 / 48 * 84 / 55 * 11 / 40 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2884_288466


namespace NUMINAMATH_CALUDE_alternating_student_arrangements_l2884_288495

def num_male_students : ℕ := 4
def num_female_students : ℕ := 5

theorem alternating_student_arrangements :
  (num_male_students.factorial * num_female_students.factorial : ℕ) = 2880 :=
by sorry

end NUMINAMATH_CALUDE_alternating_student_arrangements_l2884_288495


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2884_288400

/-- The quadratic polynomial that satisfies the given conditions -/
def q (x : ℚ) : ℚ := (6/5) * x^2 - (4/5) * x + 8/5

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-2) = 8 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2884_288400


namespace NUMINAMATH_CALUDE_bedroom_size_calculation_l2884_288461

theorem bedroom_size_calculation (total_area : ℝ) (difference : ℝ) :
  total_area = 300 →
  difference = 60 →
  ∃ (smaller_room : ℝ),
    smaller_room + (smaller_room + difference) = total_area ∧
    smaller_room = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_bedroom_size_calculation_l2884_288461


namespace NUMINAMATH_CALUDE_max_value_ratio_l2884_288475

/-- An arithmetic sequence with properties S_4 = 10 and S_8 = 36 -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_def : ∀ n : ℕ+, S n = (n : ℚ) * (a 1 + a n) / 2
  S_4 : S 4 = 10
  S_8 : S 8 = 36

/-- The maximum value of a_n / S_(n+3) for the given arithmetic sequence is 1/7 -/
theorem max_value_ratio (seq : ArithmeticSequence) :
  (∃ n : ℕ+, seq.a n / seq.S (n + 3) = 1 / 7) ∧
  (∀ n : ℕ+, seq.a n / seq.S (n + 3) ≤ 1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_max_value_ratio_l2884_288475


namespace NUMINAMATH_CALUDE_quadratic_equation_real_root_l2884_288458

theorem quadratic_equation_real_root (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k + Complex.I) * x - 2 - k * Complex.I = 0) → 
  (k = 1 ∨ k = -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_root_l2884_288458


namespace NUMINAMATH_CALUDE_average_of_three_l2884_288408

theorem average_of_three (y : ℝ) : (15 + 25 + y) / 3 = 20 → y = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_l2884_288408


namespace NUMINAMATH_CALUDE_bc_is_one_twelfth_of_ad_l2884_288415

/-- Given a line segment AD with points B and C on it, prove that BC is 1/12 of AD -/
theorem bc_is_one_twelfth_of_ad (A B C D : ℝ) : 
  (B ≤ C) →  -- B is before or at C on the line
  (C ≤ D) →  -- C is before or at D on the line
  (A ≤ B) →  -- A is before or at B on the line
  (B - A = 3 * (D - B)) →  -- AB is 3 times BD
  (C - A = 5 * (D - C)) →  -- AC is 5 times CD
  (C - B = (D - A) / 12) := by  -- BC is 1/12 of AD
sorry

end NUMINAMATH_CALUDE_bc_is_one_twelfth_of_ad_l2884_288415


namespace NUMINAMATH_CALUDE_zeros_sum_greater_than_2a_l2884_288439

/-- The function f(x) = ln x + a/x - 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x - 2

/-- Theorem: If x₁ and x₂ are the two zeros of f(x) with x₁ < x₂, then x₁ + x₂ > 2a -/
theorem zeros_sum_greater_than_2a (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ < x₂)
  (h₄ : f a x₁ = 0) (h₅ : f a x₂ = 0) :
  x₁ + x₂ > 2 * a := by
  sorry

end NUMINAMATH_CALUDE_zeros_sum_greater_than_2a_l2884_288439


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2884_288499

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 391) :
  |x - y| = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2884_288499


namespace NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l2884_288429

theorem divisibility_implies_multiple_of_three (a b : ℤ) :
  (9 : ℤ) ∣ (a^2 + a*b + b^2) → (3 : ℤ) ∣ a ∧ (3 : ℤ) ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l2884_288429


namespace NUMINAMATH_CALUDE_inequality_proof_l2884_288487

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) (h4 : n ≥ 2) :
  (3 / 2 : ℝ) < 1 / (a^n + 1) + 1 / (b^n + 1) ∧ 
  1 / (a^n + 1) + 1 / (b^n + 1) ≤ (2^(n+1) : ℝ) / (2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2884_288487


namespace NUMINAMATH_CALUDE_remaining_box_mass_l2884_288485

/-- Given a list of box masses, prove that the 20 kg box remains in the store when two companies buy five boxes, with one company taking twice the mass of the other. -/
theorem remaining_box_mass (boxes : List ℕ) : boxes = [15, 16, 18, 19, 20, 31] →
  ∃ (company1 company2 : List ℕ),
    (company1.sum + company2.sum = boxes.sum - 20) ∧
    (company2.sum = 2 * company1.sum) ∧
    (company1.length + company2.length = 5) ∧
    (∀ x ∈ company1, x ∈ boxes) ∧
    (∀ x ∈ company2, x ∈ boxes) :=
by sorry

end NUMINAMATH_CALUDE_remaining_box_mass_l2884_288485


namespace NUMINAMATH_CALUDE_product_of_sums_geq_product_l2884_288419

theorem product_of_sums_geq_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_product_of_sums_geq_product_l2884_288419


namespace NUMINAMATH_CALUDE_total_pages_bought_l2884_288477

def total_spent : ℚ := 10
def cost_per_notepad : ℚ := 5/4  -- $1.25 expressed as a rational number
def pages_per_notepad : ℕ := 60

theorem total_pages_bought : ℕ := by
  -- Proof goes here
  sorry

#check total_pages_bought = 480

end NUMINAMATH_CALUDE_total_pages_bought_l2884_288477


namespace NUMINAMATH_CALUDE_tomato_bean_percentage_is_50_l2884_288457

/-- Represents the number of cans of each ingredient in a normal batch of chili -/
structure ChiliBatch where
  chilis : ℕ
  beans : ℕ
  tomatoes : ℕ

/-- Defines a normal batch of chili -/
def normal_batch : ChiliBatch :=
  { chilis := 1, beans := 2, tomatoes := 3 }

/-- Calculates the total number of cans in a batch -/
def total_cans (batch : ChiliBatch) : ℕ :=
  batch.chilis + batch.beans + batch.tomatoes

/-- States that a quadruple batch requires 24 cans -/
axiom quadruple_batch_cans : 4 * (total_cans normal_batch) = 24

/-- Calculates the percentage of more tomatoes than beans -/
def tomato_bean_percentage (batch : ChiliBatch) : ℚ :=
  (batch.tomatoes - batch.beans : ℚ) / batch.beans * 100

/-- Theorem stating that the percentage of more tomatoes than beans is 50% -/
theorem tomato_bean_percentage_is_50 : 
  tomato_bean_percentage normal_batch = 50 := by sorry

end NUMINAMATH_CALUDE_tomato_bean_percentage_is_50_l2884_288457


namespace NUMINAMATH_CALUDE_inserted_eights_composite_l2884_288494

theorem inserted_eights_composite (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (1880 * 10^n - 611) / 9 = a * b :=
sorry

end NUMINAMATH_CALUDE_inserted_eights_composite_l2884_288494


namespace NUMINAMATH_CALUDE_function_monotonicity_l2884_288414

/-- f is an odd function -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- g is an even function -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- The main theorem -/
theorem function_monotonicity (f g : ℝ → ℝ) 
    (h_odd : IsOdd f) (h_even : IsEven g) 
    (h_sum : ∀ x, f x + g x = 3^x) :
    ∀ a b, a > b → f a > f b := by
  sorry

end NUMINAMATH_CALUDE_function_monotonicity_l2884_288414


namespace NUMINAMATH_CALUDE_complex_equality_implies_sum_l2884_288480

theorem complex_equality_implies_sum (a b : ℝ) :
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (2 + Complex.I) * (1 - b * Complex.I) = a + Complex.I →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_sum_l2884_288480


namespace NUMINAMATH_CALUDE_total_legs_in_collection_l2884_288423

/-- The number of legs for a spider -/
def spider_legs : ℕ := 8

/-- The number of legs for an ant -/
def ant_legs : ℕ := 6

/-- The number of spiders in the collection -/
def num_spiders : ℕ := 8

/-- The number of ants in the collection -/
def num_ants : ℕ := 12

/-- Theorem stating that the total number of legs in the collection is 136 -/
theorem total_legs_in_collection : 
  num_spiders * spider_legs + num_ants * ant_legs = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_collection_l2884_288423


namespace NUMINAMATH_CALUDE_range_of_expression_l2884_288426

theorem range_of_expression (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_one : x + y + z = 1) :
  -1/8 ≤ (z - x) * (z - y) ∧ (z - x) * (z - y) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l2884_288426


namespace NUMINAMATH_CALUDE_cube_root_equal_self_l2884_288471

theorem cube_root_equal_self : {x : ℝ | x = x^(1/3)} = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_cube_root_equal_self_l2884_288471


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2884_288425

/-- Given a geometric sequence {a_n} with positive terms where a_4 * a_10 = 16, prove a_7 = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ r : ℝ, ∀ n, a (n + 1) = r * a n)
  (h_product : a 4 * a 10 = 16) : 
  a 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2884_288425


namespace NUMINAMATH_CALUDE_tea_mixture_price_is_153_l2884_288445

/-- Calculates the price of a tea mixture given the prices of three tea varieties and their mixing ratio. -/
def tea_mixture_price (p1 p2 p3 : ℚ) (r1 r2 r3 : ℚ) : ℚ :=
  (p1 * r1 + p2 * r2 + p3 * r3) / (r1 + r2 + r3)

/-- Theorem stating that the price of a specific tea mixture is 153. -/
theorem tea_mixture_price_is_153 :
  tea_mixture_price 126 135 175.5 1 1 2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_is_153_l2884_288445


namespace NUMINAMATH_CALUDE_units_digit_of_M_M8_l2884_288456

-- Define the Lucas-like sequence M_n
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => 2 * M (n + 1) + M n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M8 : unitsDigit (M (M 8)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M8_l2884_288456


namespace NUMINAMATH_CALUDE_tylers_age_l2884_288455

theorem tylers_age (tyler_age brother_age : ℕ) : 
  tyler_age + 3 = brother_age →
  tyler_age + brother_age = 11 →
  tyler_age = 4 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l2884_288455


namespace NUMINAMATH_CALUDE_quality_difference_confidence_l2884_288431

/-- Data for machine production --/
structure MachineData where
  first_class : ℕ
  second_class : ℕ

/-- Calculate K² statistic --/
def calculate_k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the confidence level in quality difference --/
theorem quality_difference_confidence
  (machine_a machine_b : MachineData)
  (h_total : machine_a.first_class + machine_a.second_class = 200)
  (h_total_b : machine_b.first_class + machine_b.second_class = 200)
  (h_a_first : machine_a.first_class = 150)
  (h_b_first : machine_b.first_class = 120) :
  calculate_k_squared machine_a.first_class machine_a.second_class
                      machine_b.first_class machine_b.second_class > 6635 / 1000 :=
sorry

end NUMINAMATH_CALUDE_quality_difference_confidence_l2884_288431


namespace NUMINAMATH_CALUDE_max_trig_product_l2884_288416

theorem max_trig_product (x y z : ℝ) : 
  (Real.sin x + Real.sin (2*y) + Real.sin (3*z)) * 
  (Real.cos x + Real.cos (2*y) + Real.cos (3*z)) ≤ 4.5 := by
sorry

end NUMINAMATH_CALUDE_max_trig_product_l2884_288416


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_1419_l2884_288412

def consecutiveEvenProduct (n : ℕ) : ℕ :=
  (List.range ((n / 2) - 1)).foldl (λ acc i => acc * (2 * (i + 2))) 2

theorem smallest_n_divisible_by_1419 : 
  (∀ m : ℕ, m < 106 → m % 2 = 0 → ¬(consecutiveEvenProduct m % 1419 = 0)) ∧ 
  (consecutiveEvenProduct 106 % 1419 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_1419_l2884_288412


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l2884_288436

def geometric_series : ℕ → ℚ
  | 0 => 7/8
  | 1 => -21/32
  | 2 => 63/128
  | (n+3) => (-3/4) * geometric_series n

theorem common_ratio_of_geometric_series :
  ∀ n : ℕ, n ≥ 1 → geometric_series (n+1) / geometric_series n = -3/4 :=
by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l2884_288436


namespace NUMINAMATH_CALUDE_cost_of_treats_treats_cost_is_twelve_l2884_288465

/-- Calculates the cost of a bag of treats given the total spent and other expenses --/
theorem cost_of_treats (puppy_cost : ℝ) (dog_food : ℝ) (toys : ℝ) (crate : ℝ) (bed : ℝ) (collar_leash : ℝ) 
  (discount_rate : ℝ) (total_spent : ℝ) : ℝ :=
  let other_items := dog_food + toys + crate + bed + collar_leash
  let discounted_other_items := other_items * (1 - discount_rate)
  let treats_total := total_spent - puppy_cost - discounted_other_items
  treats_total / 2

/-- Proves that the cost of a bag of treats is $12.00 --/
theorem treats_cost_is_twelve : 
  cost_of_treats 20 20 15 20 20 15 0.2 96 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_treats_treats_cost_is_twelve_l2884_288465


namespace NUMINAMATH_CALUDE_no_real_solutions_l2884_288478

/-- The quadratic equation x^2 + 2x + 3 = 0 has no real solutions -/
theorem no_real_solutions : ¬∃ (x : ℝ), x^2 + 2*x + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2884_288478


namespace NUMINAMATH_CALUDE_list_number_fraction_l2884_288491

theorem list_number_fraction (n : ℕ) (S : ℝ) (h1 : n > 0) (h2 : S ≥ 0) : 
  n = 3 * (S / (n - 1)) → n / (S + n) = 3 / (n + 2) :=
by sorry

end NUMINAMATH_CALUDE_list_number_fraction_l2884_288491


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l2884_288404

/-- The area of the cross-section of a cylinder intersected by a plane -/
theorem cylinder_cross_section_area 
  (r : ℝ) -- radius of the cylinder base
  (α : ℝ) -- angle between the intersecting plane and the base plane
  (h₁ : r > 0) -- radius is positive
  (h₂ : 0 < α ∧ α < π / 2) -- angle is between 0 and π/2 (exclusive)
  : ∃ (A : ℝ), A = π * r^2 / Real.cos α :=
sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l2884_288404


namespace NUMINAMATH_CALUDE_gathering_handshakes_l2884_288432

/-- Represents the number of handshakes in a gathering with specific group dynamics -/
def number_of_handshakes (total_people : ℕ) (group1_size : ℕ) (group2_size : ℕ) (group3_size : ℕ) (known_by_group3 : ℕ) : ℕ :=
  let group2_handshakes := group2_size * (total_people - group2_size)
  let group3_handshakes := group3_size * (group1_size - known_by_group3 + group2_size)
  (group2_handshakes + group3_handshakes) / 2

/-- The theorem states that for the given group sizes and dynamics, the number of handshakes is 210 -/
theorem gathering_handshakes :
  number_of_handshakes 35 25 5 5 18 = 210 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l2884_288432


namespace NUMINAMATH_CALUDE_minimum_value_of_sum_l2884_288449

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ a 1 > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem minimum_value_of_sum (a : ℕ → ℝ) :
  PositiveGeometricSequence a →
  a 4 + a 3 = a 2 + a 1 + 8 →
  ∀ x, a 6 + a 5 ≥ x →
  x ≤ 32 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_sum_l2884_288449


namespace NUMINAMATH_CALUDE_complex_number_problem_l2884_288410

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  ∃ (r : ℝ), z₁ * z₂ = r →
  z₂ = 4 + 2 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2884_288410


namespace NUMINAMATH_CALUDE_number_of_sevens_in_Q_l2884_288473

/-- Definition of R_k as an integer consisting of k repetitions of the digit 7 -/
def R (k : ℕ) : ℕ := 7 * ((10^k - 1) / 9)

/-- The quotient of R_16 divided by R_2 -/
def Q : ℕ := R 16 / R 2

/-- Count the number of sevens in a natural number -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of sevens in Q is equal to 2 -/
theorem number_of_sevens_in_Q : count_sevens Q = 2 := by sorry

end NUMINAMATH_CALUDE_number_of_sevens_in_Q_l2884_288473


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2884_288470

def tshirt_price : ℝ := 10
def sweater_price : ℝ := 25
def jacket_price : ℝ := 100
def jeans_price : ℝ := 40
def shoes_price : ℝ := 70

def tshirt_discount : ℝ := 0.20
def sweater_discount : ℝ := 0.10
def jacket_discount : ℝ := 0.15
def jeans_discount : ℝ := 0.05
def shoes_discount : ℝ := 0.25

def clothes_tax : ℝ := 0.06
def shoes_tax : ℝ := 0.09

def tshirt_quantity : ℕ := 8
def sweater_quantity : ℕ := 5
def jacket_quantity : ℕ := 3
def jeans_quantity : ℕ := 6
def shoes_quantity : ℕ := 4

def total_cost : ℝ :=
  (tshirt_price * tshirt_quantity * (1 - tshirt_discount) * (1 + clothes_tax)) +
  (sweater_price * sweater_quantity * (1 - sweater_discount) * (1 + clothes_tax)) +
  (jacket_price * jacket_quantity * (1 - jacket_discount) * (1 + clothes_tax)) +
  (jeans_price * jeans_quantity * (1 - jeans_discount) * (1 + clothes_tax)) +
  (shoes_price * shoes_quantity * (1 - shoes_discount) * (1 + shoes_tax))

theorem total_cost_is_correct : total_cost = 927.97 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2884_288470


namespace NUMINAMATH_CALUDE_triangle_inequality_l2884_288464

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 + b^3 + c^3 ≤ (a + b + c) * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2884_288464


namespace NUMINAMATH_CALUDE_range_of_a_in_linear_program_l2884_288479

/-- The range of values for a given the specified constraints and maximum point -/
theorem range_of_a_in_linear_program (x y a : ℝ) : 
  (1 ≤ x + y) → (x + y ≤ 4) → 
  (-2 ≤ x - y) → (x - y ≤ 2) → 
  (a > 0) →
  (∀ x' y', (1 ≤ x' + y') → (x' + y' ≤ 4) → (-2 ≤ x' - y') → (x' - y' ≤ 2) → 
    (a * x' + y' ≤ a * x + y)) →
  (x = 3 ∧ y = 1) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_linear_program_l2884_288479


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2884_288405

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^23 + i^75 = -2*i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2884_288405


namespace NUMINAMATH_CALUDE_dave_initial_tickets_l2884_288434

/-- The number of tickets Dave spent on a stuffed tiger -/
def spent_tickets : ℕ := 43

/-- The number of tickets Dave had left after the purchase -/
def remaining_tickets : ℕ := 55

/-- The initial number of tickets Dave had -/
def initial_tickets : ℕ := spent_tickets + remaining_tickets

theorem dave_initial_tickets : initial_tickets = 98 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_tickets_l2884_288434


namespace NUMINAMATH_CALUDE_connie_marbles_l2884_288459

/-- Calculates the remaining marbles after giving some away. -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proves that Connie has 3 marbles left after giving away 70 from her initial 73 marbles. -/
theorem connie_marbles : remaining_marbles 73 70 = 3 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l2884_288459


namespace NUMINAMATH_CALUDE_modulus_of_z_l2884_288440

-- Define the complex number z
def z : ℂ := Complex.I * (3 + 2 * Complex.I)

-- State the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2884_288440


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2884_288467

/-- Given that the solution set of ax² + 2x + c < 0 is (-∞, -1/3) ∪ (1/2, +∞),
    prove that the solution set of cx² + 2x + a ≤ 0 is [-3, 2]. -/
theorem solution_set_equivalence 
  (h : ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1/3 ∨ x > 1/2))
  (a c : ℝ) :
  ∀ x : ℝ, (c*x^2 + 2*x + a ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2884_288467


namespace NUMINAMATH_CALUDE_forty_ab_over_c_value_l2884_288497

theorem forty_ab_over_c_value (a b c : ℝ) 
  (eq1 : 4 * a = 5 * b)
  (eq2 : 5 * b = 30)
  (eq3 : a + b + c = 15) :
  40 * a * b / c = 1200 := by
  sorry

end NUMINAMATH_CALUDE_forty_ab_over_c_value_l2884_288497


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2884_288469

/-- Represents a three-digit number ABC --/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents a two-digit number AB --/
def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

/-- Predicate to check if a number is a single digit --/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- Predicate to check if four numbers are distinct --/
def are_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem unique_solution_for_equation :
  ∃! (a b c d : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    is_single_digit c ∧
    is_single_digit d ∧
    are_distinct a b c d ∧
    three_digit_number a b c * two_digit_number a b + c * d = 2017 ∧
    two_digit_number a b = 14 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2884_288469


namespace NUMINAMATH_CALUDE_square_root_equality_l2884_288417

theorem square_root_equality (a b : ℝ) : 
  (a^2 + b^2)^2 = (4*a - 6*b + 13)^2 → (a^2 + b^2)^2 = 169 := by
sorry

end NUMINAMATH_CALUDE_square_root_equality_l2884_288417


namespace NUMINAMATH_CALUDE_factorial_five_equals_120_l2884_288468

theorem factorial_five_equals_120 : 5 * 4 * 3 * 2 * 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_five_equals_120_l2884_288468


namespace NUMINAMATH_CALUDE_smallest_sticker_count_l2884_288448

theorem smallest_sticker_count (S : ℕ) (h1 : S > 3) 
  (h2 : S % 5 = 3) (h3 : S % 11 = 3) (h4 : S % 13 = 3) : 
  S ≥ 718 ∧ ∃ (T : ℕ), T = 718 ∧ T % 5 = 3 ∧ T % 11 = 3 ∧ T % 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sticker_count_l2884_288448


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2884_288460

def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x^2 < 4}

theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2884_288460


namespace NUMINAMATH_CALUDE_sum_of_squares_equivalence_l2884_288489

theorem sum_of_squares_equivalence (n : ℕ) :
  (∃ (a b : ℤ), (n : ℤ) = a^2 + b^2) ↔ (∃ (c d : ℤ), (2 * n : ℤ) = c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equivalence_l2884_288489


namespace NUMINAMATH_CALUDE_max_value_of_z_l2884_288403

theorem max_value_of_z (x y : ℝ) (h1 : y ≥ x) (h2 : x + y ≤ 1) (h3 : y ≥ -1) :
  ∃ (z_max : ℝ), z_max = 1/2 ∧ ∀ z, z = 2*x - y → z ≤ z_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2884_288403


namespace NUMINAMATH_CALUDE_cable_cost_calculation_l2884_288498

/-- Calculates the total cost of cable for a neighborhood given the following parameters:
* Number of east-west and north-south streets
* Length of east-west and north-south streets
* Cable required per mile of street
* Cost of regular cable for east-west and north-south streets
* Number of intersections and cost per intersection
* Number of streets requiring higher grade cable and its cost
-/
def total_cable_cost (
  num_ew_streets : ℕ
  ) (num_ns_streets : ℕ
  ) (len_ew_street : ℝ
  ) (len_ns_street : ℝ
  ) (cable_per_mile : ℝ
  ) (cost_ew_cable : ℝ
  ) (cost_ns_cable : ℝ
  ) (num_intersections : ℕ
  ) (cost_per_intersection : ℝ
  ) (num_hg_ew_streets : ℕ
  ) (num_hg_ns_streets : ℕ
  ) (cost_hg_cable : ℝ
  ) : ℝ :=
  let regular_ew_cost := (num_ew_streets : ℝ) * len_ew_street * cable_per_mile * cost_ew_cable
  let regular_ns_cost := (num_ns_streets : ℝ) * len_ns_street * cable_per_mile * cost_ns_cable
  let hg_ew_cost := (num_hg_ew_streets : ℝ) * len_ew_street * cable_per_mile * cost_hg_cable
  let hg_ns_cost := (num_hg_ns_streets : ℝ) * len_ns_street * cable_per_mile * cost_hg_cable
  let intersection_cost := (num_intersections : ℝ) * cost_per_intersection
  regular_ew_cost + regular_ns_cost + hg_ew_cost + hg_ns_cost + intersection_cost

theorem cable_cost_calculation :
  total_cable_cost 18 10 2 4 5 2500 3500 20 5000 3 2 4000 = 1530000 := by
  sorry

end NUMINAMATH_CALUDE_cable_cost_calculation_l2884_288498


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2884_288411

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (1 - x / (x + 1)) / ((x^2 - 2*x + 1) / (x^2 - 1)) = 1 / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2884_288411


namespace NUMINAMATH_CALUDE_central_angle_of_sector_l2884_288442

-- Define the sector
structure Sector where
  radius : ℝ
  area : ℝ

-- Define the theorem
theorem central_angle_of_sector (s : Sector) (h1 : s.radius = 2) (h2 : s.area = 8) :
  (2 * s.area) / (s.radius ^ 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_of_sector_l2884_288442


namespace NUMINAMATH_CALUDE_tangent_problems_l2884_288496

theorem tangent_problems (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + Real.pi/4) = -3) ∧
  (Real.sin (2*α) / (Real.sin α ^ 2 + Real.sin α * Real.cos α) = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_problems_l2884_288496


namespace NUMINAMATH_CALUDE_total_fish_count_l2884_288421

theorem total_fish_count (pikes sturgeon herring : ℕ) 
  (h1 : pikes = 30)
  (h2 : sturgeon = 40)
  (h3 : herring = 75) :
  pikes + sturgeon + herring = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2884_288421


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l2884_288409

/-- The quadratic function f(x) = -(x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

/-- Point P1 on the graph of f -/
def P1 : ℝ × ℝ := (-1, f (-1))

/-- Point P2 on the graph of f -/
def P2 : ℝ × ℝ := (3, f 3)

/-- Point P3 on the graph of f -/
def P3 : ℝ × ℝ := (5, f 5)

theorem quadratic_points_relationship : P1.2 = P2.2 ∧ P1.2 > P3.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l2884_288409


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2884_288406

theorem complex_fraction_equality : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2884_288406


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_seven_l2884_288447

/-- The repeating decimal 0.777... as a rational number -/
def repeating_seven : ℚ := 7 / 9

/-- The reciprocal of the repeating decimal 0.777... -/
def reciprocal_repeating_seven : ℚ := 9 / 7

/-- Theorem stating that the reciprocal of 0.777... is 9/7 -/
theorem reciprocal_of_repeating_seven :
  (repeating_seven)⁻¹ = reciprocal_repeating_seven :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_seven_l2884_288447


namespace NUMINAMATH_CALUDE_balanced_quadruple_theorem_l2884_288483

def is_balanced (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem balanced_quadruple_theorem :
  ∀ x : ℝ, x > 0 →
  (∀ a b c d : ℝ, is_balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔
  x ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_balanced_quadruple_theorem_l2884_288483


namespace NUMINAMATH_CALUDE_function_properties_l2884_288493

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.cos (ω * x / 2))^2 + Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) - 1/2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : is_periodic (f ω) Real.pi) (h3 : ∀ T, 0 < T → T < Real.pi → ¬ is_periodic (f ω) T) :
  (ω = 2) ∧ 
  (∀ x, f ω x ≤ 1) ∧
  (∀ x, f ω x ≥ -1) ∧
  (∃ x, f ω x = 1) ∧
  (∃ x, f ω x = -1) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6), 
    ∀ y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6),
    x ≤ y → f ω x ≤ f ω y) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2884_288493


namespace NUMINAMATH_CALUDE_min_value_of_function_l2884_288450

theorem min_value_of_function (x : ℝ) (h : x > 10) : x^2 / (x - 10) ≥ 40 ∧ ∃ y > 10, y^2 / (y - 10) = 40 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2884_288450


namespace NUMINAMATH_CALUDE_count_valid_sequences_l2884_288418

/-- Represents a die throw result -/
inductive DieThrow
  | even (n : Nat)
  | odd (n : Nat)

/-- Represents a point in 2D space -/
structure Point where
  x : Nat
  y : Nat

/-- Defines how a point moves based on a die throw -/
def move (p : Point) (t : DieThrow) : Point :=
  match t with
  | DieThrow.even n => Point.mk (p.x + n) p.y
  | DieThrow.odd n => Point.mk p.x (p.y + n)

/-- Defines a valid sequence of die throws -/
def validSequence (seq : List DieThrow) : Prop :=
  let finalPoint := seq.foldl move (Point.mk 0 0)
  finalPoint.x = 4 ∧ finalPoint.y = 4

/-- The main theorem to prove -/
theorem count_valid_sequences : 
  (∃ (seqs : List (List DieThrow)), 
    (∀ seq ∈ seqs, validSequence seq) ∧ 
    (∀ seq, validSequence seq → seq ∈ seqs) ∧
    seqs.length = 38) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_sequences_l2884_288418


namespace NUMINAMATH_CALUDE_sophie_germain_prime_units_digit_l2884_288451

/-- A positive prime number p is a Sophie Germain prime if 2p + 1 is also prime. -/
def SophieGermainPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p + 1)

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem sophie_germain_prime_units_digit (p : ℕ) (h : SophieGermainPrime p) (h_gt : p > 6) :
  unitsDigit p = 1 ∨ unitsDigit p = 3 :=
sorry

end NUMINAMATH_CALUDE_sophie_germain_prime_units_digit_l2884_288451


namespace NUMINAMATH_CALUDE_mod_product_equivalence_l2884_288486

theorem mod_product_equivalence (m : ℕ) : 
  (241 * 398 ≡ m [ZMOD 50]) → 
  (0 ≤ m ∧ m < 50) → 
  m = 18 := by
sorry

end NUMINAMATH_CALUDE_mod_product_equivalence_l2884_288486


namespace NUMINAMATH_CALUDE_winner_votes_not_unique_l2884_288402

/-- Represents an election result --/
structure ElectionResult where
  totalVotes : ℕ
  winnerVotes : ℕ
  secondPlaceVotes : ℕ

/-- Conditions of the election --/
def electionConditions (result : ElectionResult) : Prop :=
  (result.winnerVotes : ℚ) / result.totalVotes = 58 / 100 ∧
  result.winnerVotes - result.secondPlaceVotes = 1200

/-- Theorem stating that the number of votes for the winning candidate cannot be uniquely determined --/
theorem winner_votes_not_unique :
  ∃ (result1 result2 : ElectionResult),
    result1 ≠ result2 ∧
    electionConditions result1 ∧
    electionConditions result2 :=
sorry

end NUMINAMATH_CALUDE_winner_votes_not_unique_l2884_288402


namespace NUMINAMATH_CALUDE_complement_of_union_l2884_288444

def U : Set Int := {x | x^2 - 5*x - 6 ≤ 0}

def A : Set Int := {x | x*(2-x) ≥ 0}

def B : Set Int := {1, 2, 3}

theorem complement_of_union : (U \ (A ∪ B)) = {-1, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2884_288444


namespace NUMINAMATH_CALUDE_internet_cost_comparison_l2884_288438

/-- Cost calculation for dial-up internet access -/
def dialup_cost (hours : ℝ) : ℝ := 4.2 * hours

/-- Cost calculation for monthly subscription -/
def subscription_cost : ℝ := 130 - 25

/-- The number of hours where both methods cost the same -/
def equal_cost_hours : ℝ := 25

theorem internet_cost_comparison :
  /- Part 1: Prove that costs are equal at 25 hours -/
  dialup_cost equal_cost_hours = subscription_cost ∧
  /- Part 2: Prove that subscription is cheaper for 30 hours -/
  dialup_cost 30 > subscription_cost := by
  sorry

#check internet_cost_comparison

end NUMINAMATH_CALUDE_internet_cost_comparison_l2884_288438


namespace NUMINAMATH_CALUDE_paint_area_is_123_l2884_288443

/-- Calculates the area to be painted on a wall with given dimensions and window areas -/
def area_to_paint (wall_height wall_length window1_height window1_width window2_height window2_width : ℝ) : ℝ :=
  let wall_area := wall_height * wall_length
  let window1_area := window1_height * window1_width
  let window2_area := window2_height * window2_width
  wall_area - (window1_area + window2_area)

/-- Theorem: The area to be painted is 123 square feet -/
theorem paint_area_is_123 :
  area_to_paint 10 15 3 5 2 6 = 123 := by
  sorry

#eval area_to_paint 10 15 3 5 2 6

end NUMINAMATH_CALUDE_paint_area_is_123_l2884_288443


namespace NUMINAMATH_CALUDE_min_value_of_E_l2884_288401

theorem min_value_of_E (x : ℝ) :
  let f (E : ℝ) := |x - 4| + |E| + |x - 5|
  (∃ (E : ℝ), f E = 10 ∧ ∀ (E' : ℝ), f E' ≥ 10) →
  (∃ (E_min : ℝ), |E_min| = 9 ∧ ∀ (E : ℝ), |E| ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_E_l2884_288401


namespace NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l2884_288482

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2}

-- Define set A
def A : Set Int := {x ∈ U | x^2 + x - 2 = 0}

-- Define set B
def B : Set Int := {0, -2}

-- Theorem statement
theorem intersection_of_B_and_complement_of_A :
  B ∩ (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l2884_288482


namespace NUMINAMATH_CALUDE_track_length_is_400_l2884_288420

/-- Represents a circular running track -/
structure Track :=
  (length : ℝ)

/-- Represents a runner on the track -/
structure Runner :=
  (speed : ℝ)
  (initialPosition : ℝ)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (position : ℝ)
  (time : ℝ)

/-- The scenario of two runners on a circular track -/
def runningScenario (t : Track) (r1 r2 : Runner) (m1 m2 : Meeting) : Prop :=
  r1.initialPosition = 0 ∧
  r2.initialPosition = t.length / 2 ∧
  r1.speed > 0 ∧
  r2.speed < 0 ∧
  m1.position = 100 ∧
  m2.position - m1.position = 150 ∧
  m1.time * r1.speed = 100 ∧
  m1.time * r2.speed = t.length / 2 - 100 ∧
  m2.time * r1.speed = t.length / 2 - 50 ∧
  m2.time * r2.speed = t.length / 2 + 50

theorem track_length_is_400 (t : Track) (r1 r2 : Runner) (m1 m2 : Meeting) :
  runningScenario t r1 r2 m1 m2 → t.length = 400 :=
by sorry

end NUMINAMATH_CALUDE_track_length_is_400_l2884_288420


namespace NUMINAMATH_CALUDE_good_number_count_and_gcd_l2884_288428

def is_good_number (n : ℕ) : Prop :=
  n ≤ 2012 ∧ n % 9 = 6

theorem good_number_count_and_gcd :
  (∃ (S : Finset ℕ), (∀ n, n ∈ S ↔ is_good_number n) ∧ S.card = 223) ∧
  (∃ d : ℕ, d > 0 ∧ (∀ n, is_good_number n → d ∣ n) ∧
    ∀ m, m > 0 → (∀ n, is_good_number n → m ∣ n) → m ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_good_number_count_and_gcd_l2884_288428


namespace NUMINAMATH_CALUDE_intersection_eq_union_implies_a_eq_3_intersection_eq_nonempty_implies_a_eq_neg_5_div_2_l2884_288474

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 + (4 - a^2) * x + a + 3 = 0}
def B := {x : ℝ | x^2 - 5 * x + 6 = 0}
def C := {x : ℝ | 2 * x^2 - 5 * x + 2 = 0}

-- Theorem 1
theorem intersection_eq_union_implies_a_eq_3 :
  ∃ a : ℝ, (A a) ∩ B = (A a) ∪ B → a = 3 := by sorry

-- Theorem 2
theorem intersection_eq_nonempty_implies_a_eq_neg_5_div_2 :
  ∃ a : ℝ, (A a) ∩ B = (A a) ∩ C ∧ (A a) ∩ B ≠ ∅ → a = -5/2 := by sorry

end NUMINAMATH_CALUDE_intersection_eq_union_implies_a_eq_3_intersection_eq_nonempty_implies_a_eq_neg_5_div_2_l2884_288474


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2884_288441

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2884_288441


namespace NUMINAMATH_CALUDE_commutator_power_zero_l2884_288490

open Matrix

theorem commutator_power_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (h_n : n ≥ 2) 
  (h_x : ∃ x : ℝ, x ≠ 0 ∧ x ≠ 1/2 ∧ x ≠ 1 ∧ x • (A * B) + (1 - x) • (B * A) = 1) :
  (A * B - B * A) ^ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_commutator_power_zero_l2884_288490


namespace NUMINAMATH_CALUDE_train_tunnel_time_l2884_288435

/-- Calculates the time taken for a train to pass through a tunnel -/
theorem train_tunnel_time (train_length : ℝ) (pole_passing_time : ℝ) (tunnel_length : ℝ) :
  train_length = 500 →
  pole_passing_time = 20 →
  tunnel_length = 500 →
  (train_length + tunnel_length) / (train_length / pole_passing_time) = 40 := by
  sorry


end NUMINAMATH_CALUDE_train_tunnel_time_l2884_288435


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2884_288430

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2884_288430


namespace NUMINAMATH_CALUDE_exponential_inequality_l2884_288452

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2^a + 2*a = 2^b + 3*b → a > b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2884_288452


namespace NUMINAMATH_CALUDE_remainder_of_large_number_div_16_l2884_288492

theorem remainder_of_large_number_div_16 :
  65985241545898754582556898522454889 % 16 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_div_16_l2884_288492


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2884_288407

theorem arithmetic_calculation : 1^2 + (2 * 3)^3 - 4^2 + Real.sqrt 9 = 204 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2884_288407


namespace NUMINAMATH_CALUDE_model_c_net_change_l2884_288427

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_increase (price : ℝ) (increase : ℝ) : ℝ :=
  price * (1 + increase)

def model_c_price : ℝ := 2000

def model_c_discount1 : ℝ := 0.20
def model_c_increase : ℝ := 0.20
def model_c_discount2 : ℝ := 0.05

theorem model_c_net_change :
  let price1 := apply_discount model_c_price model_c_discount1
  let price2 := apply_increase price1 model_c_increase
  let price3 := apply_discount price2 model_c_discount2
  price3 - model_c_price = -176 := by
  sorry

end NUMINAMATH_CALUDE_model_c_net_change_l2884_288427


namespace NUMINAMATH_CALUDE_intersection_complement_sets_l2884_288484

open Set

theorem intersection_complement_sets (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let M : Set ℝ := {x | b < x ∧ x < (a + b) / 2}
  let N : Set ℝ := {x | Real.sqrt (a * b) < x ∧ x < a}
  M ∩ (Nᶜ) = {x | b < x ∧ x ≤ Real.sqrt (a * b)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_sets_l2884_288484


namespace NUMINAMATH_CALUDE_m_range_l2884_288437

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∀ x : ℝ, x^2 + (m-3)*x + 1 ≠ 0

def Q (m : ℝ) : Prop := ∃ a b : ℝ, a > b ∧ a^2 + b^2 = m-1 ∧
  ∀ x y : ℝ, x^2 + y^2/(m-1) = 1 ↔ (x/a)^2 + (y/b)^2 = 1

-- Define the theorem
theorem m_range :
  (∀ m : ℝ, (¬(P m) → False) ∧ ((P m ∧ Q m) → False)) →
  {m : ℝ | 1 < m ∧ m ≤ 2} = {m : ℝ | ∃ x : ℝ, m = x ∧ 1 < x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2884_288437


namespace NUMINAMATH_CALUDE_function_maximum_condition_l2884_288446

open Real

theorem function_maximum_condition (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (1/2) * Real.exp (2*x) + (a - Real.exp 1) * Real.exp x - a * Real.exp 1 + b
  (∀ x, f x ≤ f 1) → a < -Real.exp 1 :=
by
  sorry

end NUMINAMATH_CALUDE_function_maximum_condition_l2884_288446


namespace NUMINAMATH_CALUDE_power_of_two_difference_divisible_by_1987_l2884_288481

theorem power_of_two_difference_divisible_by_1987 :
  ∃ (a b : ℕ), 0 ≤ b ∧ b < a ∧ a ≤ 1987 ∧ (2^a - 2^b) % 1987 = 0 :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_difference_divisible_by_1987_l2884_288481


namespace NUMINAMATH_CALUDE_sheila_work_hours_l2884_288413

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ  -- Hours worked on Monday, Wednesday, and Friday combined
  tue_thu_hours : ℕ      -- Hours worked on Tuesday and Thursday combined
  hourly_rate : ℕ        -- Hourly rate in dollars
  weekly_earnings : ℕ    -- Total weekly earnings in dollars

/-- Theorem: Given Sheila's work schedule and earnings, prove she works 24 hours on Mon, Wed, Fri -/
theorem sheila_work_hours (s : WorkSchedule) 
  (h1 : s.tue_thu_hours = 12)     -- 6 hours each on Tuesday and Thursday
  (h2 : s.hourly_rate = 12)       -- $12 per hour
  (h3 : s.weekly_earnings = 432)  -- $432 per week
  : s.mon_wed_fri_hours = 24 := by
  sorry


end NUMINAMATH_CALUDE_sheila_work_hours_l2884_288413


namespace NUMINAMATH_CALUDE_divisibility_property_l2884_288422

theorem divisibility_property (q : ℕ) (h1 : q > 1) (h2 : Odd q) :
  ∃ k : ℕ, (q + 1) ^ ((q + 1) / 2) = (q + 1) * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2884_288422


namespace NUMINAMATH_CALUDE_at_least_one_equals_one_iff_sum_gt_product_l2884_288433

theorem at_least_one_equals_one_iff_sum_gt_product (m n : ℕ+) :
  (m = 1 ∨ n = 1) ↔ (m + n : ℝ) > m * n := by sorry

end NUMINAMATH_CALUDE_at_least_one_equals_one_iff_sum_gt_product_l2884_288433


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_4a_l2884_288472

theorem factorization_a_squared_minus_4a (a : ℝ) : a^2 - 4*a = a*(a - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_4a_l2884_288472


namespace NUMINAMATH_CALUDE_line_slope_l2884_288463

theorem line_slope (x y : ℝ) : 3 * y + 2 = -4 * x - 9 → (y - (-11/3)) / (x - 0) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2884_288463


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2884_288488

/-- A line passing through (2,3) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through (2,3)
  passes_through : a * 2 + b * 3 + c = 0
  -- The line has equal intercepts on both axes
  equal_intercepts : a ≠ 0 ∧ b ≠ 0 ∧ (c / a = c / b ∨ c = 0)

/-- The equation of an equal intercept line is either x+y-5=0 or 3x-2y=0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) ∨ (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2884_288488


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2884_288453

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 9*x + 18 = 0

-- Define the roots of the equation
def root1 : ℝ := 3
def root2 : ℝ := 6

-- Define the isosceles triangle formed by the roots
def isosceles_triangle (a b : ℝ) : Prop :=
  (quadratic_equation a ∧ quadratic_equation b) ∧
  ((a = root1 ∧ b = root2) ∨ (a = root2 ∧ b = root1))

-- State the theorem
theorem isosceles_triangle_perimeter :
  ∀ a b : ℝ, isosceles_triangle a b → a + 2*b = 15 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2884_288453


namespace NUMINAMATH_CALUDE_eight_fourth_equals_sixteen_n_l2884_288462

theorem eight_fourth_equals_sixteen_n (n : ℕ) : 8^4 = 16^n → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_eight_fourth_equals_sixteen_n_l2884_288462


namespace NUMINAMATH_CALUDE_bottle_caps_distribution_l2884_288424

theorem bottle_caps_distribution (num_children : ℕ) (total_caps : ℕ) (caps_per_child : ℕ) :
  num_children = 9 →
  total_caps = 45 →
  total_caps = num_children * caps_per_child →
  caps_per_child = 5 := by
sorry

end NUMINAMATH_CALUDE_bottle_caps_distribution_l2884_288424


namespace NUMINAMATH_CALUDE_loan_payback_calculation_l2884_288476

/-- Calculates the total amount to be paid back for a loan with interest -/
def total_payback (principal : ℝ) (interest_rate : ℝ) : ℝ :=
  principal * (1 + interest_rate)

/-- Theorem: Given a loan of $1200 with a 10% interest rate, the total amount to be paid back is $1320 -/
theorem loan_payback_calculation :
  total_payback 1200 0.1 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_loan_payback_calculation_l2884_288476


namespace NUMINAMATH_CALUDE_max_value_theorem_min_value_theorem_l2884_288454

-- Statement 1
theorem max_value_theorem (x : ℝ) (h : x < 1/2) :
  ∃ (max_val : ℝ), max_val = -1 ∧ 
  ∀ y : ℝ, y < 1/2 → 2*y + 1/(2*y - 1) ≤ max_val :=
sorry

-- Statement 2
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 2/b = 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2*Real.sqrt 2 ∧
  a*(b - 1) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_min_value_theorem_l2884_288454
