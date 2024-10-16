import Mathlib

namespace NUMINAMATH_CALUDE_charlie_has_largest_answer_l133_13396

def starting_number : ℕ := 15

def alice_operation (n : ℕ) : ℕ := ((n - 2)^2 + 3)

def bob_operation (n : ℕ) : ℕ := (n^2 - 2 + 3)

def charlie_operation (n : ℕ) : ℕ := ((n - 2 + 3)^2)

theorem charlie_has_largest_answer :
  charlie_operation starting_number > alice_operation starting_number ∧
  charlie_operation starting_number > bob_operation starting_number := by
  sorry

end NUMINAMATH_CALUDE_charlie_has_largest_answer_l133_13396


namespace NUMINAMATH_CALUDE_short_trees_after_planting_l133_13314

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem stating that the total number of short trees after planting is 98 -/
theorem short_trees_after_planting :
  total_short_trees 41 57 = 98 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_l133_13314


namespace NUMINAMATH_CALUDE_tank_plastering_cost_per_sqm_l133_13397

/-- Given a tank with specified dimensions and total plastering cost, 
    calculate the cost per square meter for plastering. -/
theorem tank_plastering_cost_per_sqm 
  (length width depth : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 25) 
  (h2 : width = 12) 
  (h3 : depth = 6) 
  (h4 : total_cost = 186) : 
  total_cost / (length * width + 2 * length * depth + 2 * width * depth) = 0.25 := by
  sorry

#check tank_plastering_cost_per_sqm

end NUMINAMATH_CALUDE_tank_plastering_cost_per_sqm_l133_13397


namespace NUMINAMATH_CALUDE_candidate_percentage_l133_13333

theorem candidate_percentage (passing_mark total_mark : ℕ) 
  (h1 : passing_mark = 160)
  (h2 : total_mark = 300)
  (h3 : (60 : ℕ) * total_mark / 100 = passing_mark + 20)
  (h4 : passing_mark - 40 > 0) : 
  (passing_mark - 40) * 100 / total_mark = 40 := by
  sorry

end NUMINAMATH_CALUDE_candidate_percentage_l133_13333


namespace NUMINAMATH_CALUDE_worlds_largest_dough_ball_profit_l133_13350

/-- Calculate the profit from making the world's largest dough ball -/
theorem worlds_largest_dough_ball_profit :
  let flour_needed : ℕ := 500
  let salt_needed : ℕ := 10
  let sugar_needed : ℕ := 20
  let butter_needed : ℕ := 50
  let flour_bag_size : ℕ := 50
  let flour_bag_price : ℚ := 20
  let salt_price_per_pound : ℚ := 0.2
  let sugar_price_per_pound : ℚ := 0.5
  let butter_price_per_pound : ℚ := 2
  let butter_discount : ℚ := 0.1
  let chef_a_payment : ℚ := 200
  let chef_b_payment : ℚ := 250
  let chef_c_payment : ℚ := 300
  let chef_tax_rate : ℚ := 0.05
  let promotion_cost : ℚ := 1000
  let ticket_price : ℚ := 20
  let tickets_sold : ℕ := 1200

  let flour_cost := (flour_needed / flour_bag_size : ℚ) * flour_bag_price
  let salt_cost := salt_needed * salt_price_per_pound
  let sugar_cost := sugar_needed * sugar_price_per_pound
  let butter_cost := butter_needed * butter_price_per_pound * (1 - butter_discount)
  let ingredient_cost := flour_cost + salt_cost + sugar_cost + butter_cost

  let chefs_payment := chef_a_payment + chef_b_payment + chef_c_payment
  let chefs_tax := chefs_payment * chef_tax_rate
  let total_chef_cost := chefs_payment + chefs_tax

  let total_cost := ingredient_cost + total_chef_cost + promotion_cost
  let revenue := tickets_sold * ticket_price
  let profit := revenue - total_cost

  profit = 21910.50 := by sorry

end NUMINAMATH_CALUDE_worlds_largest_dough_ball_profit_l133_13350


namespace NUMINAMATH_CALUDE_expression_evaluation_l133_13383

/-- Given real numbers x, y, and z, prove that the expression
    ((P+Q)/(P-Q) - (P-Q)/(P+Q)) equals (x^2 - y^2 - 2yz - z^2) / (xy + xz),
    where P = x + y + z and Q = x - y - z. -/
theorem expression_evaluation (x y z : ℝ) :
  let P := x + y + z
  let Q := x - y - z
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (x^2 - y^2 - 2*y*z - z^2) / (x*y + x*z) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l133_13383


namespace NUMINAMATH_CALUDE_bryans_deposit_l133_13352

theorem bryans_deposit (mark_deposit : ℕ) (bryan_deposit : ℕ) 
  (h1 : mark_deposit = 88)
  (h2 : bryan_deposit < 5 * mark_deposit)
  (h3 : mark_deposit + bryan_deposit = 400) :
  bryan_deposit = 312 := by
sorry

end NUMINAMATH_CALUDE_bryans_deposit_l133_13352


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l133_13320

/-- Represents the composition of a student body -/
structure StudentBody where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  sum_eq_total : freshmen + sophomores + juniors = total

/-- Represents a stratified sample from a student body -/
structure StratifiedSample where
  body : StudentBody
  sample_size : ℕ
  sampled_freshmen : ℕ
  sampled_sophomores : ℕ
  sampled_juniors : ℕ
  sum_eq_sample_size : sampled_freshmen + sampled_sophomores + sampled_juniors = sample_size

/-- Checks if a stratified sample is proportionally correct -/
def is_proportional_sample (sample : StratifiedSample) : Prop :=
  sample.sampled_freshmen * sample.body.total = sample.body.freshmen * sample.sample_size ∧
  sample.sampled_sophomores * sample.body.total = sample.body.sophomores * sample.sample_size ∧
  sample.sampled_juniors * sample.body.total = sample.body.juniors * sample.sample_size

theorem correct_stratified_sample :
  let school : StudentBody := {
    total := 1000,
    freshmen := 400,
    sophomores := 340,
    juniors := 260,
    sum_eq_total := by sorry
  }
  let sample : StratifiedSample := {
    body := school,
    sample_size := 50,
    sampled_freshmen := 20,
    sampled_sophomores := 17,
    sampled_juniors := 13,
    sum_eq_sample_size := by sorry
  }
  is_proportional_sample sample := by sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l133_13320


namespace NUMINAMATH_CALUDE_product_divisibility_l133_13359

theorem product_divisibility (a b c : ℤ) 
  (h1 : (a + b + c)^2 = -(a*b + a*c + b*c))
  (h2 : a + b ≠ 0)
  (h3 : b + c ≠ 0)
  (h4 : a + c ≠ 0) :
  (∃ k : ℤ, (a + b) * (a + c) = k * (b + c)) ∧
  (∃ k : ℤ, (a + b) * (b + c) = k * (a + c)) ∧
  (∃ k : ℤ, (a + c) * (b + c) = k * (a + b)) :=
sorry

end NUMINAMATH_CALUDE_product_divisibility_l133_13359


namespace NUMINAMATH_CALUDE_nine_digit_square_impossibility_l133_13325

theorem nine_digit_square_impossibility (n : ℕ) : 
  (100000000 ≤ n ∧ n < 1000000000) →  -- n is a nine-digit number
  (∃ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
    n = 100000000 * d1 + 10000000 * d2 + 1000000 * d3 + 100000 * d4 + 
        10000 * d5 + 1000 * d6 + 100 * d7 + 10 * d8 + 5 ∧
    ({d1, d2, d3, d4, d5, d6, d7, d8, 5} : Finset ℕ) = Finset.range 9) →  -- n uses all digits from 1 to 9 and ends in 5
  ¬∃ (m : ℕ), n = m^2 :=  -- n is not a perfect square
by
  sorry

end NUMINAMATH_CALUDE_nine_digit_square_impossibility_l133_13325


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l133_13331

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {1, 3, 5}

theorem intersection_complement_equality : N ∩ (U \ M) = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l133_13331


namespace NUMINAMATH_CALUDE_opposite_to_turquoise_is_pink_l133_13336

/-- Represents the colors of the squares --/
inductive Color
  | Pink
  | Violet
  | Turquoise
  | Orange

/-- Represents a face of the cube --/
structure Face where
  color : Color

/-- Represents the cube formed by folding the squares --/
structure Cube where
  faces : List Face
  opposite : Face → Face

/-- The configuration of the cube --/
def cube_config : Cube :=
  { faces := [
      Face.mk Color.Pink,
      Face.mk Color.Pink,
      Face.mk Color.Pink,
      Face.mk Color.Violet,
      Face.mk Color.Violet,
      Face.mk Color.Turquoise,
      Face.mk Color.Orange
    ],
    opposite := sorry  -- The actual implementation of the opposite function
  }

/-- Theorem stating that the face opposite to Turquoise is Pink --/
theorem opposite_to_turquoise_is_pink :
  ∃ (f : Face), f ∈ cube_config.faces ∧ 
    f.color = Color.Turquoise ∧ 
    (cube_config.opposite f).color = Color.Pink :=
  sorry


end NUMINAMATH_CALUDE_opposite_to_turquoise_is_pink_l133_13336


namespace NUMINAMATH_CALUDE_homework_problems_per_page_l133_13309

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 101)
  (h2 : finished_problems = 47)
  (h3 : remaining_pages = 6)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end NUMINAMATH_CALUDE_homework_problems_per_page_l133_13309


namespace NUMINAMATH_CALUDE_intersection_A_B_l133_13344

def A : Set ℝ := {-3, -1, 0, 1}

def B : Set ℝ := {x | (x + 2) * (x - 1) < 0}

theorem intersection_A_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l133_13344


namespace NUMINAMATH_CALUDE_range_of_m_for_single_valued_function_l133_13348

/-- A function is single-valued on an interval if there exists a unique x in the interval
    that satisfies (b-a) * f'(x) = f(b) - f(a) --/
def SingleValued (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ (b - a) * (deriv f x) = f b - f a

/-- The function f(x) = x^3 - x^2 + m --/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ x^3 - x^2 + m

theorem range_of_m_for_single_valued_function (a : ℝ) (h_a : a ≥ 1) :
  ∀ m : ℝ, SingleValued (f m) 0 a ∧ 
  (∃ x y, 0 ≤ x ∧ x < y ∧ y ≤ a ∧ f m x = 0 ∧ f m y = 0) ∧ 
  (∀ z, 0 ≤ z ∧ z ≤ a ∧ f m z = 0 → z = x ∨ z = y) →
  -1 ≤ m ∧ m < 4/27 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_single_valued_function_l133_13348


namespace NUMINAMATH_CALUDE_sequence_integer_count_l133_13334

def sequence_term (n : ℕ) : ℚ :=
  9720 / 2^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n))) ∧
  (∀ (k : ℕ), k > 0 →
    ((∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
     (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n)))
    → k = 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l133_13334


namespace NUMINAMATH_CALUDE_investment_sum_l133_13323

theorem investment_sum (raghu_investment : ℕ) : 
  raghu_investment = 2100 →
  let trishul_investment := raghu_investment - raghu_investment / 10
  let vishal_investment := trishul_investment + trishul_investment / 10
  raghu_investment + trishul_investment + vishal_investment = 6069 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l133_13323


namespace NUMINAMATH_CALUDE_marker_difference_l133_13389

theorem marker_difference (price : ℚ) (hector_count alicia_count : ℕ) : 
  price > 1/100 →  -- More than a penny each
  price * hector_count = 276/100 →  -- Hector paid $2.76
  price * alicia_count = 407/100 →  -- Alicia paid $4.07
  alicia_count - hector_count = 13 := by
  sorry

end NUMINAMATH_CALUDE_marker_difference_l133_13389


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l133_13354

/-- A rectangular solid with prime edge lengths and volume 231 has surface area 262 -/
theorem rectangular_solid_surface_area : 
  ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 231 →
  2 * (a * b + b * c + a * c) = 262 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l133_13354


namespace NUMINAMATH_CALUDE_difference_between_shares_l133_13316

/-- Represents the distribution of money among three people -/
structure MoneyDistribution where
  ratio : Fin 3 → ℕ
  vasimInitialShare : ℕ
  farukTaxRate : ℚ
  vasimTaxRate : ℚ
  ranjithTaxRate : ℚ

/-- Calculates the final share after tax for a given initial share and tax rate -/
def finalShareAfterTax (initialShare : ℕ) (taxRate : ℚ) : ℚ :=
  (1 - taxRate) * initialShare

/-- Theorem stating the difference between Ranjith's and Faruk's final shares -/
theorem difference_between_shares (d : MoneyDistribution) 
  (h1 : d.ratio 0 = 3) 
  (h2 : d.ratio 1 = 5) 
  (h3 : d.ratio 2 = 8) 
  (h4 : d.vasimInitialShare = 1500) 
  (h5 : d.farukTaxRate = 1/10) 
  (h6 : d.vasimTaxRate = 3/20) 
  (h7 : d.ranjithTaxRate = 3/25) : 
  finalShareAfterTax (d.ratio 2 * d.vasimInitialShare / d.ratio 1) d.ranjithTaxRate -
  finalShareAfterTax (d.ratio 0 * d.vasimInitialShare / d.ratio 1) d.farukTaxRate = 1302 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_shares_l133_13316


namespace NUMINAMATH_CALUDE_probability_of_shaded_triangle_l133_13304

/-- Given a diagram with 6 triangles, where 3 are shaded and all have equal selection probability, 
    the probability of selecting a shaded triangle is 1/2 -/
theorem probability_of_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) :
  total_triangles = 6 →
  shaded_triangles = 3 →
  (shaded_triangles : ℚ) / (total_triangles : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_shaded_triangle_l133_13304


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_product_difference_l133_13319

theorem consecutive_integers_square_sum_product_difference : 
  let a : ℕ := 9
  let b : ℕ := 10
  (a^2 + b^2) - (a * b) = 91 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_product_difference_l133_13319


namespace NUMINAMATH_CALUDE_equality_condition_l133_13322

theorem equality_condition (x y : ℝ) : 
  (x - 9)^2 + (y - 10)^2 + (x - y)^2 = 1/3 → x = 28/3 ∧ y = 29/3 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l133_13322


namespace NUMINAMATH_CALUDE_function_inequalities_l133_13362

theorem function_inequalities (p q r s : ℝ) (h : p * s - q * r < 0) :
  let f := fun x => (p * x + q) / (r * x + s)
  ∀ x₁ x₂ ε : ℝ,
    ε > 0 →
    (x₁ < x₂ ∧ x₂ < -s/r → f x₁ > f x₂) ∧
    (-s/r < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
    (x₁ < x₂ ∧ x₂ < -s/r → f (x₁ - ε) - f x₁ < f (x₂ - ε) - f x₂) ∧
    (-s/r < x₁ ∧ x₁ < x₂ → f x₁ - f (x₁ + ε) > f x₂ - f (x₂ + ε)) :=
by sorry

end NUMINAMATH_CALUDE_function_inequalities_l133_13362


namespace NUMINAMATH_CALUDE_initial_chips_count_l133_13384

/-- The number of tortilla chips Nancy initially had in her bag -/
def initial_chips : ℕ := sorry

/-- The number of tortilla chips Nancy gave to her brother -/
def chips_to_brother : ℕ := 7

/-- The number of tortilla chips Nancy gave to her sister -/
def chips_to_sister : ℕ := 5

/-- The number of tortilla chips Nancy kept for herself -/
def chips_kept : ℕ := 10

/-- Theorem stating that the initial number of chips is 22 -/
theorem initial_chips_count : initial_chips = 22 := by sorry

end NUMINAMATH_CALUDE_initial_chips_count_l133_13384


namespace NUMINAMATH_CALUDE_inequality_proof_l133_13364

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 
  3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l133_13364


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l133_13303

theorem least_positive_linear_combination : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (x y : ℤ), 24 * x + 18 * y = m) → n ≤ m) ∧ 
  (∃ (x y : ℤ), 24 * x + 18 * y = n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l133_13303


namespace NUMINAMATH_CALUDE_factorial_15_value_l133_13307

theorem factorial_15_value : Nat.factorial 15 = 1307674368000 := by
  sorry

#eval Nat.factorial 15

end NUMINAMATH_CALUDE_factorial_15_value_l133_13307


namespace NUMINAMATH_CALUDE_planting_schemes_count_l133_13372

/-- The number of seed types -/
def num_seed_types : ℕ := 5

/-- The number of plots -/
def num_plots : ℕ := 4

/-- The number of seed types to be selected -/
def num_selected : ℕ := 4

/-- The number of options for the first plot (pumpkins or pomegranates) -/
def first_plot_options : ℕ := 2

/-- Calculate the number of planting schemes -/
def num_planting_schemes : ℕ :=
  first_plot_options * (Nat.choose (num_seed_types - 1) (num_selected - 1)) * (Nat.factorial (num_plots - 1))

theorem planting_schemes_count : num_planting_schemes = 48 := by sorry

end NUMINAMATH_CALUDE_planting_schemes_count_l133_13372


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l133_13330

/-- The first term of the geometric series -/
def a₁ : ℚ := 4/3

/-- The second term of the geometric series -/
def a₂ : ℚ := 16/9

/-- The third term of the geometric series -/
def a₃ : ℚ := 64/27

/-- The common ratio of the geometric series -/
def r : ℚ := 4/3

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l133_13330


namespace NUMINAMATH_CALUDE_min_length_MN_l133_13398

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Theorem: Minimum length of MN in a unit cube -/
theorem min_length_MN (cube : Cube) (M N L : Point3D) : 
  (cube.A.x = 0 ∧ cube.A.y = 0 ∧ cube.A.z = 0) →  -- A is at origin
  (cube.B.x = 1 ∧ cube.B.y = 0 ∧ cube.B.z = 0) →  -- B is at (1,0,0)
  (cube.C.x = 1 ∧ cube.C.y = 1 ∧ cube.C.z = 0) →  -- C is at (1,1,0)
  (cube.D.x = 0 ∧ cube.D.y = 1 ∧ cube.D.z = 0) →  -- D is at (0,1,0)
  (cube.A1.x = 0 ∧ cube.A1.y = 0 ∧ cube.A1.z = 1) →  -- A1 is at (0,0,1)
  (cube.C1.x = 1 ∧ cube.C1.y = 1 ∧ cube.C1.z = 1) →  -- C1 is at (1,1,1)
  (cube.D1.x = 0 ∧ cube.D1.y = 1 ∧ cube.D1.z = 1) →  -- D1 is at (0,1,1)
  (∃ t : ℝ, M.x = t * cube.A1.x ∧ M.y = t * cube.A1.y ∧ M.z = t * cube.A1.z) →  -- M is on ray AA1
  (∃ s : ℝ, N.x = cube.B.x + s * (cube.C.x - cube.B.x) ∧ 
            N.y = cube.B.y + s * (cube.C.y - cube.B.y) ∧ 
            N.z = cube.B.z + s * (cube.C.z - cube.B.z)) →  -- N is on ray BC
  (∃ u : ℝ, L.x = cube.C1.x + u * (cube.D1.x - cube.C1.x) ∧ 
            L.y = cube.C1.y + u * (cube.D1.y - cube.C1.y) ∧ 
            L.z = cube.C1.z + u * (cube.D1.z - cube.C1.z)) →  -- L is on edge C1D1
  (∃ v : ℝ, M.x + v * (N.x - M.x) = L.x ∧ 
            M.y + v * (N.y - M.y) = L.y ∧ 
            M.z + v * (N.z - M.z) = L.z) →  -- MN intersects C1D1 at L
  (∀ M' N' : Point3D, 
    (∃ t' : ℝ, M'.x = t' * cube.A1.x ∧ M'.y = t' * cube.A1.y ∧ M'.z = t' * cube.A1.z) →
    (∃ s' : ℝ, N'.x = cube.B.x + s' * (cube.C.x - cube.B.x) ∧ 
              N'.y = cube.B.y + s' * (cube.C.y - cube.B.y) ∧ 
              N'.z = cube.B.z + s' * (cube.C.z - cube.B.z)) →
    Real.sqrt ((M'.x - N'.x)^2 + (M'.y - N'.y)^2 + (M'.z - N'.z)^2) ≥ 3) →
  Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2 + (M.z - N.z)^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_length_MN_l133_13398


namespace NUMINAMATH_CALUDE_complex_square_plus_self_l133_13332

theorem complex_square_plus_self (z : ℂ) :
  z = -1/2 + (Complex.I * Real.sqrt 3) / 2 →
  z^2 + z = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_square_plus_self_l133_13332


namespace NUMINAMATH_CALUDE_a_positive_iff_sum_geq_two_l133_13346

theorem a_positive_iff_sum_geq_two (a : ℝ) : a > 0 ↔ a + 1/a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_a_positive_iff_sum_geq_two_l133_13346


namespace NUMINAMATH_CALUDE_max_value_is_70_l133_13317

/-- Represents the types of rocks available --/
inductive RockType
  | Seven
  | Three
  | Two

/-- The weight of a rock in pounds --/
def weight : RockType → ℕ
  | RockType.Seven => 7
  | RockType.Three => 3
  | RockType.Two => 2

/-- The value of a rock in dollars --/
def value : RockType → ℕ
  | RockType.Seven => 20
  | RockType.Three => 10
  | RockType.Two => 4

/-- The maximum weight Carl can carry in pounds --/
def maxWeight : ℕ := 21

/-- The minimum number of each type of rock available --/
def minAvailable : ℕ := 15

/-- A function to calculate the total value of a combination of rocks --/
def totalValue (combination : RockType → ℕ) : ℕ :=
  (combination RockType.Seven * value RockType.Seven) +
  (combination RockType.Three * value RockType.Three) +
  (combination RockType.Two * value RockType.Two)

/-- A function to calculate the total weight of a combination of rocks --/
def totalWeight (combination : RockType → ℕ) : ℕ :=
  (combination RockType.Seven * weight RockType.Seven) +
  (combination RockType.Three * weight RockType.Three) +
  (combination RockType.Two * weight RockType.Two)

/-- The main theorem stating that the maximum value of rocks Carl can carry is $70 --/
theorem max_value_is_70 :
  ∃ (combination : RockType → ℕ),
    (∀ rock, combination rock ≤ minAvailable) ∧
    totalWeight combination ≤ maxWeight ∧
    totalValue combination = 70 ∧
    (∀ other : RockType → ℕ,
      (∀ rock, other rock ≤ minAvailable) →
      totalWeight other ≤ maxWeight →
      totalValue other ≤ 70) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_is_70_l133_13317


namespace NUMINAMATH_CALUDE_simple_interest_increase_l133_13395

/-- Given that the simple interest on $2000 increases by $40 when the time increases by x years,
    and the rate percent per annum is 0.5, prove that x = 4. -/
theorem simple_interest_increase (x : ℝ) : 
  (2000 * 0.5 * x) / 100 = 40 → x = 4 := by sorry

end NUMINAMATH_CALUDE_simple_interest_increase_l133_13395


namespace NUMINAMATH_CALUDE_nonadjacent_arrangements_correct_nonadjacent_arrangements_simplified_l133_13390

/-- The number of circular arrangements of n people where two specific people are not adjacent -/
def nonadjacent_arrangements (n : ℕ) : ℕ :=
  (n - 3) * (n - 2).factorial

/-- Theorem stating the number of circular arrangements of n people (n ≥ 3) 
    where two specific people are not adjacent -/
theorem nonadjacent_arrangements_correct (n : ℕ) (h : n ≥ 3) :
  nonadjacent_arrangements n = (n - 1).factorial - 2 * (n - 2).factorial :=
by
  sorry

/-- Corollary: The number of arrangements where two specific people are not adjacent
    is equal to (n-3)(n-2)! -/
theorem nonadjacent_arrangements_simplified (n : ℕ) (h : n ≥ 3) :
  nonadjacent_arrangements n = (n - 3) * (n - 2).factorial :=
by
  sorry

end NUMINAMATH_CALUDE_nonadjacent_arrangements_correct_nonadjacent_arrangements_simplified_l133_13390


namespace NUMINAMATH_CALUDE_pies_sold_theorem_l133_13387

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of customers who ordered apple pie slices -/
def apple_customers : ℕ := 56

/-- Represents the number of customers who ordered peach pie slices -/
def peach_customers : ℕ := 48

/-- Calculates the total number of pies sold given the number of slices per pie and the number of customers -/
def total_pies (apple_slices peach_slices apple_customers peach_customers : ℕ) : ℕ :=
  (apple_customers / apple_slices) + (peach_customers / peach_slices)

/-- Theorem stating that the total number of pies sold is 15 -/
theorem pies_sold_theorem : total_pies apple_slices peach_slices apple_customers peach_customers = 15 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_theorem_l133_13387


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l133_13327

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 = 2 → a 5 = 7 → a 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l133_13327


namespace NUMINAMATH_CALUDE_locus_characterization_locus_is_ray_l133_13370

/-- The locus of points P satisfying |PM| - |PN| = 4, where M(-2,0) and N(2,0) -/
def locus : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0 ∧ p.1 ≥ 2}

theorem locus_characterization (P : ℝ × ℝ) :
  P ∈ locus ↔ Real.sqrt ((P.1 + 2)^2 + P.2^2) - Real.sqrt ((P.1 - 2)^2 + P.2^2) = 4 :=
sorry

/-- The points M and N -/
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

theorem locus_is_ray :
  locus = {p : ℝ × ℝ | p.2 = 0 ∧ p.1 ≥ 2} :=
sorry

end NUMINAMATH_CALUDE_locus_characterization_locus_is_ray_l133_13370


namespace NUMINAMATH_CALUDE_arrangement_count_l133_13365

-- Define the number of children
def n : ℕ := 6

-- Define the number of odd positions available for the specific child
def odd_positions : ℕ := 3

-- Define the function to calculate the number of arrangements
def arrangements (n : ℕ) (odd_positions : ℕ) : ℕ :=
  odd_positions * Nat.factorial (n - 1)

-- Theorem statement
theorem arrangement_count :
  arrangements n odd_positions = 360 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l133_13365


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_area_l133_13366

theorem right_triangle_inscribed_circle_area
  (r : ℝ) (c : ℝ) (h_r : r = 5) (h_c : c = 34) :
  let s := (c + 2 * r + (c - 2 * r)) / 2
  r * s = 195 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_area_l133_13366


namespace NUMINAMATH_CALUDE_equation_solution_l133_13341

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l133_13341


namespace NUMINAMATH_CALUDE_other_diagonal_length_l133_13388

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 15 cm and an area of 90 cm²,
    the length of the other diagonal is 12 cm -/
theorem other_diagonal_length :
  ∀ r : Rhombus, r.d1 = 15 ∧ r.area = 90 → r.d2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l133_13388


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l133_13380

/-- Represents days of the week --/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties --/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numberOfFridays : Nat
  numberOfDays : Nat
  firstDayNotFriday : firstDay ≠ DayOfWeek.Friday
  lastDayNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : numberOfFridays = 5

/-- Function to determine the day of the week for a given day number --/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

theorem twelfth_day_is_monday (m : Month) : 
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l133_13380


namespace NUMINAMATH_CALUDE_cat_cafe_ratio_l133_13393

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 10

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := 40

/-- The theorem stating the ratio of cats between Cat Cafe Paw and Cat Cafe Cool -/
theorem cat_cafe_ratio : paw_cats / cool_cats = 2 := by
  sorry

end NUMINAMATH_CALUDE_cat_cafe_ratio_l133_13393


namespace NUMINAMATH_CALUDE_age_difference_proof_l133_13324

theorem age_difference_proof (ann_age susan_age : ℕ) : 
  ann_age > susan_age →
  ann_age + susan_age = 27 →
  susan_age = 11 →
  ann_age - susan_age = 5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l133_13324


namespace NUMINAMATH_CALUDE_hexagon_rearrangement_l133_13385

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Represents the problem setup -/
structure HexagonProblem where
  original_rectangle : Rectangle
  resulting_square : Square
  is_valid : Prop

/-- The theorem stating the relationship between the original rectangle and the resulting square -/
theorem hexagon_rearrangement (p : HexagonProblem) 
  (h1 : p.original_rectangle.length = 9)
  (h2 : p.original_rectangle.width = 16)
  (h3 : p.is_valid)
  (h4 : p.original_rectangle.length * p.original_rectangle.width = p.resulting_square.side ^ 2) :
  p.resulting_square.side / 2 = 6 := by sorry

end NUMINAMATH_CALUDE_hexagon_rearrangement_l133_13385


namespace NUMINAMATH_CALUDE_min_coach_handshakes_zero_l133_13391

/-- Represents the total number of handshakes in the gymnastics meet -/
def total_handshakes : ℕ := 325

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the minimum number of coach handshakes is 0 -/
theorem min_coach_handshakes_zero :
  ∃ (n : ℕ), gymnast_handshakes n = total_handshakes ∧ n > 1 :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_zero_l133_13391


namespace NUMINAMATH_CALUDE_specific_number_probability_l133_13379

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The total number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of favorable outcomes for a specific type of number -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a specific type of number when tossing two dice -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem specific_number_probability :
  probability = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_specific_number_probability_l133_13379


namespace NUMINAMATH_CALUDE_three_X_five_l133_13351

/-- The operation X defined for real numbers -/
def X (a b : ℝ) : ℝ := b + 15 * a - 2 * a^2

/-- Theorem stating that 3X5 equals 32 -/
theorem three_X_five : X 3 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_three_X_five_l133_13351


namespace NUMINAMATH_CALUDE_red_peppers_weight_l133_13340

/-- The weight of red peppers bought by Dale's Vegetarian Restaurant -/
def weight_red_peppers : ℝ :=
  5.666666667 - 2.8333333333333335

/-- Theorem stating that the weight of red peppers is the difference between
    the total weight of peppers and the weight of green peppers -/
theorem red_peppers_weight :
  weight_red_peppers = 5.666666667 - 2.8333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_red_peppers_weight_l133_13340


namespace NUMINAMATH_CALUDE_divisibility_condition_l133_13399

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def form_number (B : ℕ) : ℕ := 5000 + 200 + 10 * B + 6

theorem divisibility_condition (B : ℕ) (h : B ≤ 9) :
  is_divisible_by_3 (form_number B) ↔ (B = 2 ∨ B = 5 ∨ B = 8) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l133_13399


namespace NUMINAMATH_CALUDE_tetrahedron_face_area_relation_l133_13315

/-- Theorem about the relationship between face areas, edges, and angles in a tetrahedron -/
theorem tetrahedron_face_area_relation 
  (S₁ S₂ a b : ℝ) (α φ : ℝ) 
  (h_S₁ : S₁ > 0) (h_S₂ : S₂ > 0) 
  (h_a : a > 0) (h_b : b > 0)
  (h_α : 0 < α ∧ α < π) (h_φ : 0 < φ ∧ φ < π) :
  S₁^2 + S₂^2 - 2*S₁*S₂*(Real.cos α) = (a*b*(Real.sin φ) / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_face_area_relation_l133_13315


namespace NUMINAMATH_CALUDE_rational_square_plus_one_positive_l133_13367

theorem rational_square_plus_one_positive (a : ℚ) : a^2 + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_plus_one_positive_l133_13367


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l133_13312

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l133_13312


namespace NUMINAMATH_CALUDE_energy_change_in_triangle_l133_13311

/-- The energy stored between two point charges -/
def energy_between_charges (distance : ℝ) : ℝ := sorry

/-- The total energy stored in a system of three point charges -/
def total_energy (d1 d2 d3 : ℝ) : ℝ := 
  energy_between_charges d1 + energy_between_charges d2 + energy_between_charges d3

theorem energy_change_in_triangle (initial_energy : ℝ) :
  initial_energy = 18 →
  ∃ (energy_func : ℝ → ℝ),
    (energy_func 1 + energy_func 1 + energy_func (Real.sqrt 2) = initial_energy) ∧
    (energy_func 1 + energy_func (Real.sqrt 2 / 2) + energy_func (Real.sqrt 2 / 2) = 6 + 12 * Real.sqrt 2) := by
  sorry

#check energy_change_in_triangle

end NUMINAMATH_CALUDE_energy_change_in_triangle_l133_13311


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l133_13386

/-- Calculate the total amount owed after one year with simple interest. -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 75) 
  (h2 : rate = 0.07) 
  (h3 : time = 1) : 
  principal * (1 + rate * time) = 80.25 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l133_13386


namespace NUMINAMATH_CALUDE_inverse_proportion_points_order_l133_13377

/-- Given points A(x₁, -6), B(x₂, -2), C(x₃, 3) on the graph of y = -12/x,
    prove that x₃ < x₁ < x₂ -/
theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) : 
  (-6 : ℝ) = -12 / x₁ → 
  (-2 : ℝ) = -12 / x₂ → 
  (3 : ℝ) = -12 / x₃ → 
  x₃ < x₁ ∧ x₁ < x₂ := by
  sorry

#check inverse_proportion_points_order

end NUMINAMATH_CALUDE_inverse_proportion_points_order_l133_13377


namespace NUMINAMATH_CALUDE_sum_divisible_by_17_l133_13349

theorem sum_divisible_by_17 : 
  ∃ k : ℤ, 82 + 83 + 84 + 85 + 86 + 87 + 88 + 89 = 17 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_17_l133_13349


namespace NUMINAMATH_CALUDE_original_average_theorem_l133_13353

theorem original_average_theorem (S : Finset ℝ) (f : ℝ → ℝ) :
  S.card = 7 →
  (∀ x ∈ S, f x = 5 * x) →
  (S.sum f) / S.card = 75 →
  (S.sum id) / S.card = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_original_average_theorem_l133_13353


namespace NUMINAMATH_CALUDE_length_width_ratio_l133_13382

/-- Represents a rectangular roof --/
structure RectangularRoof where
  length : ℝ
  width : ℝ

/-- Properties of the specific roof in the problem --/
def problem_roof : RectangularRoof → Prop
  | roof => roof.length * roof.width = 900 ∧ 
            roof.length - roof.width = 45

/-- The theorem stating the ratio of length to width --/
theorem length_width_ratio (roof : RectangularRoof) 
  (h : problem_roof roof) : 
  roof.length / roof.width = 4 := by
  sorry

#check length_width_ratio

end NUMINAMATH_CALUDE_length_width_ratio_l133_13382


namespace NUMINAMATH_CALUDE_mary_savings_problem_l133_13306

theorem mary_savings_problem (S : ℝ) (x : ℝ) (h1 : S > 0) (h2 : 0 ≤ x ∧ x ≤ 1) : 
  12 * x * S = 7 * (1 - x) * S → (1 - x) = 12 / 19 := by
  sorry

end NUMINAMATH_CALUDE_mary_savings_problem_l133_13306


namespace NUMINAMATH_CALUDE_trig_identity_l133_13355

theorem trig_identity (α : ℝ) : 
  (Real.sin α + Real.sin (3 * α) - Real.sin (5 * α)) / 
  (Real.cos α - Real.cos (3 * α) - Real.cos (5 * α)) = Real.tan α :=
sorry

end NUMINAMATH_CALUDE_trig_identity_l133_13355


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l133_13335

/-- Given a line segment from (2, 5) to (x, 15) with length 13 and x > 0, prove x = 2 + √69 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 2)^2 + 10^2)^(1/2 : ℝ) = 13 → 
  x = 2 + (69 : ℝ)^(1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l133_13335


namespace NUMINAMATH_CALUDE_sixth_root_of_24414062515625_l133_13347

theorem sixth_root_of_24414062515625 : (24414062515625 : ℝ) ^ (1/6 : ℝ) = 51 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_24414062515625_l133_13347


namespace NUMINAMATH_CALUDE_tangent_circle_height_difference_l133_13338

/-- A circle tangent to the parabola y = x^2 + 1 at two points and lying inside the parabola -/
structure TangentCircle where
  /-- x-coordinate of the point of tangency -/
  a : ℝ
  /-- y-coordinate of the center of the circle -/
  b : ℝ
  /-- radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_point : b = a^2 + 1/2
  /-- The circle equation satisfies the tangency condition -/
  circle_eq : b^2 - r^2 = a^4 + 1

/-- The difference in height between the center of the circle and the points of tangency is -1/2 -/
theorem tangent_circle_height_difference (c : TangentCircle) :
  c.b - (c.a^2 + 1) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_height_difference_l133_13338


namespace NUMINAMATH_CALUDE_no_integer_solution_l133_13300

theorem no_integer_solution : ¬ ∃ (m n : ℤ), m^2 + 1954 = n^2 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l133_13300


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l133_13374

theorem concentric_circles_radii_difference
  (r R : ℝ) -- r is radius of smaller circle, R is radius of larger circle
  (h_positive : r > 0) -- r is positive
  (h_area_ratio : R^2 / r^2 = 16 / 3) -- area ratio condition
  : R - r = r * (4 * Real.sqrt 3 - 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l133_13374


namespace NUMINAMATH_CALUDE_deducted_salary_proof_l133_13302

-- Define the weekly salary
def weekly_salary : ℚ := 1043

-- Define the number of workdays in a week
def workdays_per_week : ℕ := 5

-- Define the number of absent days
def absent_days : ℕ := 2

-- Define the daily wage
def daily_wage : ℚ := weekly_salary / workdays_per_week

-- Define the deduction
def deduction : ℚ := daily_wage * absent_days

-- Define the deducted salary
def deducted_salary : ℚ := weekly_salary - deduction

-- Theorem to prove
theorem deducted_salary_proof : deducted_salary = 625.80 := by
  sorry

end NUMINAMATH_CALUDE_deducted_salary_proof_l133_13302


namespace NUMINAMATH_CALUDE_equation_solution_l133_13373

theorem equation_solution : ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l133_13373


namespace NUMINAMATH_CALUDE_hospital_babies_l133_13310

theorem hospital_babies (total_babies : ℕ) (triplets quadruplets quintuplets : ℕ) : 
  total_babies = 2500 →
  triplets = 2 * quadruplets →
  quintuplets = quadruplets / 2 →
  3 * triplets + 4 * quadruplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 500 := by
  sorry

end NUMINAMATH_CALUDE_hospital_babies_l133_13310


namespace NUMINAMATH_CALUDE_complex_division_equality_l133_13345

theorem complex_division_equality : (3 - Complex.I) / Complex.I = -1 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l133_13345


namespace NUMINAMATH_CALUDE_max_page_number_proof_l133_13392

/-- Counts the number of '5' digits used in numbering pages from 1 to n --/
def count_fives (n : ℕ) : ℕ := sorry

/-- The highest page number that can be labeled with 16 '5' digits --/
def max_page_number : ℕ := 75

theorem max_page_number_proof :
  count_fives max_page_number ≤ 16 ∧
  ∀ m : ℕ, m > max_page_number → count_fives m > 16 :=
sorry

end NUMINAMATH_CALUDE_max_page_number_proof_l133_13392


namespace NUMINAMATH_CALUDE_distance_between_points_l133_13308

theorem distance_between_points (max_speed : ℝ) (total_time : ℝ) (stream_speed_ab : ℝ) (stream_speed_ba : ℝ) (speed_percentage_ab : ℝ) (speed_percentage_ba : ℝ) (D : ℝ) :
  max_speed = 5 →
  total_time = 5 →
  stream_speed_ab = 1 →
  stream_speed_ba = 2 →
  speed_percentage_ab = 0.9 →
  speed_percentage_ba = 0.8 →
  D / (speed_percentage_ab * max_speed + stream_speed_ab) + D / (speed_percentage_ba * max_speed - stream_speed_ba) = total_time →
  26 * D = 110 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l133_13308


namespace NUMINAMATH_CALUDE_bake_sale_goal_l133_13368

def brownie_count : ℕ := 4
def brownie_price : ℕ := 3
def lemon_square_count : ℕ := 5
def lemon_square_price : ℕ := 2
def cookie_count : ℕ := 7
def cookie_price : ℕ := 4

def total_goal : ℕ := 50

theorem bake_sale_goal :
  brownie_count * brownie_price +
  lemon_square_count * lemon_square_price +
  cookie_count * cookie_price = total_goal :=
by
  sorry

end NUMINAMATH_CALUDE_bake_sale_goal_l133_13368


namespace NUMINAMATH_CALUDE_claire_cooking_time_l133_13326

/-- Represents Claire's daily schedule -/
structure DailySchedule where
  total_hours : ℕ
  sleep_hours : ℕ
  clean_hours : ℕ
  craft_hours : ℕ
  tailor_hours : ℕ
  cook_hours : ℕ

/-- Claire's schedule satisfies the given conditions -/
def is_valid_schedule (s : DailySchedule) : Prop :=
  s.total_hours = 24 ∧
  s.sleep_hours = 8 ∧
  s.clean_hours = 4 ∧
  s.craft_hours = 5 ∧
  s.tailor_hours = s.craft_hours ∧
  s.total_hours = s.sleep_hours + s.clean_hours + s.craft_hours + s.tailor_hours + s.cook_hours

theorem claire_cooking_time (s : DailySchedule) (h : is_valid_schedule s) : s.cook_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_claire_cooking_time_l133_13326


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l133_13339

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = r + s →        -- c is divided into r and s
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  a / b = 2 / 5 →    -- Given ratio of a to b
  r / s = 4 / 25     -- Conclusion: ratio of r to s
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l133_13339


namespace NUMINAMATH_CALUDE_train_length_l133_13376

/-- The length of a train given its speed and time to pass a point --/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 36) (h2 : time = 16) :
  speed * time * (5 / 18) = 160 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l133_13376


namespace NUMINAMATH_CALUDE_integral_problem_1_l133_13375

theorem integral_problem_1 (x : ℝ) (h : x > 0) :
  (deriv (fun x => 4 * (x^(1/2)/2 - x^(1/4) + Real.log (1 + x^(1/4)))) x) = 1 / (x^(1/2) + x^(1/4)) :=
sorry

end NUMINAMATH_CALUDE_integral_problem_1_l133_13375


namespace NUMINAMATH_CALUDE_carnation_bouquet_combinations_l133_13318

def distribute_carnations (total : ℕ) (types : ℕ) (extras : ℕ) : ℕ :=
  Nat.choose (extras + types - 1) (types - 1)

theorem carnation_bouquet_combinations :
  distribute_carnations 5 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_carnation_bouquet_combinations_l133_13318


namespace NUMINAMATH_CALUDE_average_after_removing_numbers_l133_13381

theorem average_after_removing_numbers (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) :
  n = 50 →
  initial_avg = 38 →
  removed1 = 45 →
  removed2 = 55 →
  (n : ℚ) * initial_avg - (removed1 + removed2) = ((n - 2) : ℚ) * 37.5 :=
by sorry

end NUMINAMATH_CALUDE_average_after_removing_numbers_l133_13381


namespace NUMINAMATH_CALUDE_stamp_collection_difference_l133_13305

theorem stamp_collection_difference (kylie_stamps nelly_stamps : ℕ) : 
  kylie_stamps = 34 →
  nelly_stamps > kylie_stamps →
  kylie_stamps + nelly_stamps = 112 →
  nelly_stamps - kylie_stamps = 44 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_difference_l133_13305


namespace NUMINAMATH_CALUDE_condition_type_l133_13363

theorem condition_type (A B : Prop) 
  (h1 : ¬B → ¬A) 
  (h2 : ¬(¬A → ¬B)) : 
  (A → B) ∧ ¬(B → A) := by sorry

end NUMINAMATH_CALUDE_condition_type_l133_13363


namespace NUMINAMATH_CALUDE_village_population_l133_13329

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  percentage = 80 / 100 →
  partial_population = 32000 →
  percentage * (total_population : ℚ) = partial_population →
  total_population = 40000 :=
by
  sorry

end NUMINAMATH_CALUDE_village_population_l133_13329


namespace NUMINAMATH_CALUDE_last_year_honey_harvest_l133_13358

/-- 
Given Diane's honey harvest information:
- This year's harvest: 8564 pounds
- Increase from last year: 6085 pounds

Prove that last year's harvest was 2479 pounds.
-/
theorem last_year_honey_harvest 
  (this_year : ℕ) 
  (increase : ℕ) 
  (h1 : this_year = 8564)
  (h2 : increase = 6085) :
  this_year - increase = 2479 := by
sorry

end NUMINAMATH_CALUDE_last_year_honey_harvest_l133_13358


namespace NUMINAMATH_CALUDE_min_stamps_proof_l133_13321

/-- The minimum number of stamps needed to make 60 cents using only 5 cent and 6 cent stamps -/
def min_stamps : ℕ := 10

/-- The value of stamps in cents -/
def total_value : ℕ := 60

/-- Proves that the minimum number of stamps needed to make 60 cents using only 5 cent and 6 cent stamps is 10 -/
theorem min_stamps_proof :
  ∀ c f : ℕ, 5 * c + 6 * f = total_value → c + f ≥ min_stamps :=
sorry

end NUMINAMATH_CALUDE_min_stamps_proof_l133_13321


namespace NUMINAMATH_CALUDE_prob_girl_given_boy_specific_l133_13369

/-- Represents a club with members -/
structure Club where
  total_members : ℕ
  girls : ℕ
  boys : ℕ

/-- The probability of choosing a girl given that at least one boy is chosen -/
def prob_girl_given_boy (c : Club) : ℚ :=
  (c.girls * c.boys : ℚ) / ((c.girls * c.boys + (c.boys * (c.boys - 1)) / 2) : ℚ)

theorem prob_girl_given_boy_specific :
  let c : Club := { total_members := 12, girls := 7, boys := 5 }
  prob_girl_given_boy c = 7/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_girl_given_boy_specific_l133_13369


namespace NUMINAMATH_CALUDE_picture_coverage_percentage_l133_13361

theorem picture_coverage_percentage (poster_width poster_height picture_width picture_height : ℝ) 
  (hw_poster : poster_width = 50 ∧ poster_height = 100)
  (hw_picture : picture_width = 20 ∧ picture_height = 40) :
  (picture_width * picture_height) / (poster_width * poster_height) * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_picture_coverage_percentage_l133_13361


namespace NUMINAMATH_CALUDE_arithmetic_sequence_exponents_l133_13378

theorem arithmetic_sequence_exponents (a b : ℝ) (m : ℝ) : 
  a > 0 → b > 0 → 
  2^a = m → 3^b = m → 
  2 * a * b = a + b → 
  m = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_exponents_l133_13378


namespace NUMINAMATH_CALUDE_probability_three_primes_in_six_rolls_l133_13360

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def count_primes_on_12_sided_die : ℕ := 5

def probability_prime_on_12_sided_die : ℚ := 5 / 12

def probability_not_prime_on_12_sided_die : ℚ := 7 / 12

def number_of_ways_to_choose_3_out_of_6 : ℕ := 20

theorem probability_three_primes_in_six_rolls : 
  (probability_prime_on_12_sided_die ^ 3 * 
   probability_not_prime_on_12_sided_die ^ 3 * 
   number_of_ways_to_choose_3_out_of_6 : ℚ) = 3575 / 124416 := by sorry

end NUMINAMATH_CALUDE_probability_three_primes_in_six_rolls_l133_13360


namespace NUMINAMATH_CALUDE_all_students_accounted_for_no_unsatisfactory_grades_l133_13342

theorem all_students_accounted_for (top_marks : ℚ) (average_marks : ℚ) (good_marks : ℚ)
  (h1 : top_marks = 1 / 6)
  (h2 : average_marks = 1 / 3)
  (h3 : good_marks = 1 / 2) :
  top_marks + average_marks + good_marks = 1 :=
by
  sorry

theorem no_unsatisfactory_grades (total_fraction : ℚ)
  (h : total_fraction = 1) :
  1 - total_fraction = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_all_students_accounted_for_no_unsatisfactory_grades_l133_13342


namespace NUMINAMATH_CALUDE_triangle_median_and_symmetric_point_l133_13357

/-- Triangle OAB with vertices O(0,0), A(2,0), and B(3,2) -/
structure Triangle :=
  (O : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)

/-- Line l containing the median on side OA -/
structure MedianLine :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Point symmetric to A with respect to line l -/
structure SymmetricPoint :=
  (x : ℝ)
  (y : ℝ)

theorem triangle_median_and_symmetric_point 
  (t : Triangle)
  (l : MedianLine)
  (h1 : t.O = (0, 0))
  (h2 : t.A = (2, 0))
  (h3 : t.B = (3, 2))
  (h4 : l.slope = 1)
  (h5 : l.intercept = -1)
  : ∃ (p : SymmetricPoint), 
    (∀ (x y : ℝ), y = l.slope * x + l.intercept ↔ y = x - 1) ∧
    p.x = 1 ∧ p.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_and_symmetric_point_l133_13357


namespace NUMINAMATH_CALUDE_max_flowers_grown_l133_13313

theorem max_flowers_grown (total_seeds : ℕ) (seeds_per_bed : ℕ) : 
  total_seeds = 55 → seeds_per_bed = 15 → ∃ (max_flowers : ℕ), max_flowers ≤ 55 ∧ 
  ∀ (actual_flowers : ℕ), actual_flowers ≤ max_flowers := by
  sorry

end NUMINAMATH_CALUDE_max_flowers_grown_l133_13313


namespace NUMINAMATH_CALUDE_problem_statement_l133_13328

theorem problem_statement (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) :
  a^2006 + (a + b)^2007 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l133_13328


namespace NUMINAMATH_CALUDE_trenton_fixed_earnings_l133_13343

/-- Trenton's weekly earnings structure -/
structure WeeklyEarnings where
  fixed : ℝ
  commissionRate : ℝ
  salesGoal : ℝ
  totalEarningsGoal : ℝ

/-- Trenton's actual weekly earnings -/
def actualEarnings (w : WeeklyEarnings) : ℝ :=
  w.fixed + w.commissionRate * w.salesGoal

/-- Theorem: Trenton's fixed weekly earnings are $190 -/
theorem trenton_fixed_earnings :
  ∀ w : WeeklyEarnings,
  w.commissionRate = 0.04 →
  w.salesGoal = 7750 →
  w.totalEarningsGoal = 500 →
  actualEarnings w ≥ w.totalEarningsGoal →
  w.fixed = 190 := by
sorry

end NUMINAMATH_CALUDE_trenton_fixed_earnings_l133_13343


namespace NUMINAMATH_CALUDE_find_MN_length_l133_13371

-- Define the triangles
structure Triangle :=
  (a b c : ℝ)

-- Define similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the triangles
def PQR : Triangle := ⟨4, 8, sorry⟩
def XYZ : Triangle := ⟨sorry, 24, sorry⟩
def MNO : Triangle := ⟨sorry, sorry, 32⟩

-- State the theorem
theorem find_MN_length :
  similar PQR XYZ →
  similar XYZ MNO →
  MNO.a = 16 := by sorry

end NUMINAMATH_CALUDE_find_MN_length_l133_13371


namespace NUMINAMATH_CALUDE_basketball_team_selection_l133_13301

theorem basketball_team_selection (girls boys called_back : ℕ) : 
  girls = 15 → boys = 25 → called_back = 7 → 
  girls + boys - called_back = 33 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l133_13301


namespace NUMINAMATH_CALUDE_cos_alpha_value_l133_13337

theorem cos_alpha_value (α : Real) : 
  (∃ x y : Real, x ≤ 0 ∧ y = -4/3 * x ∧ 
   x = Real.cos α ∧ y = Real.sin α) → 
  Real.cos α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l133_13337


namespace NUMINAMATH_CALUDE_sin_2α_minus_π_over_6_l133_13356

theorem sin_2α_minus_π_over_6 (α : Real) 
  (h : Real.cos (α + 2 * Real.pi / 3) = 3 / 5) : 
  Real.sin (2 * α - Real.pi / 6) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2α_minus_π_over_6_l133_13356


namespace NUMINAMATH_CALUDE_find_A_l133_13394

theorem find_A : ∃ A : ℕ, A = 38 ∧ A / 7 = 5 ∧ A % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l133_13394
