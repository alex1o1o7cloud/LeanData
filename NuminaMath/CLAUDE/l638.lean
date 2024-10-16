import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_ratio_l638_63826

theorem system_solution_ratio (x y c d : ℝ) : 
  (4 * x - 2 * y = c) →
  (5 * y - 10 * x = d) →
  d ≠ 0 →
  c / d = 0 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l638_63826


namespace NUMINAMATH_CALUDE_wheel_distance_l638_63886

/-- The distance covered by a wheel given its circumference and number of revolutions -/
theorem wheel_distance (circumference : ℝ) (revolutions : ℝ) :
  circumference = 56 →
  revolutions = 3.002729754322111 →
  circumference * revolutions = 168.1528670416402 := by
  sorry

end NUMINAMATH_CALUDE_wheel_distance_l638_63886


namespace NUMINAMATH_CALUDE_fencing_cost_9m_square_l638_63896

/-- Cost of fencing for each side of a square -/
structure FencingCost where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculate the total cost of fencing a square -/
def totalCost (cost : FencingCost) (sideLength : ℕ) : ℕ :=
  (cost.first + cost.second + cost.third + cost.fourth) * sideLength

/-- The fencing costs for the problem -/
def givenCost : FencingCost :=
  { first := 79
    second := 92
    third := 85
    fourth := 96 }

/-- Theorem: The total cost of fencing the square with side length 9 meters is $3168 -/
theorem fencing_cost_9m_square (cost : FencingCost := givenCost) :
  totalCost cost 9 = 3168 := by
  sorry

#eval totalCost givenCost 9

end NUMINAMATH_CALUDE_fencing_cost_9m_square_l638_63896


namespace NUMINAMATH_CALUDE_cookies_given_proof_l638_63807

/-- The number of cookies Paco gave to his friend -/
def cookies_given_to_friend : ℕ := sorry

/-- The initial number of cookies Paco had -/
def initial_cookies : ℕ := 41

/-- The number of cookies Paco ate -/
def cookies_eaten : ℕ := 18

theorem cookies_given_proof :
  cookies_given_to_friend = 9 ∧
  initial_cookies = cookies_given_to_friend + cookies_eaten + cookies_given_to_friend ∧
  cookies_eaten = cookies_given_to_friend + 9 :=
sorry

end NUMINAMATH_CALUDE_cookies_given_proof_l638_63807


namespace NUMINAMATH_CALUDE_roberto_outfits_l638_63819

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 8
  let jackets : ℕ := 3
  let shoes : ℕ := 4
  number_of_outfits trousers shirts jackets shoes = 480 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l638_63819


namespace NUMINAMATH_CALUDE_forgotten_and_doubled_sum_l638_63866

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem forgotten_and_doubled_sum (luke_sum carissa_sum : ℕ) : 
  sum_first_n 20 = 210 →
  luke_sum = 207 →
  carissa_sum = 225 →
  (sum_first_n 20 - luke_sum) + (carissa_sum - sum_first_n 20) = 18 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_and_doubled_sum_l638_63866


namespace NUMINAMATH_CALUDE_rectangular_plot_length_difference_l638_63812

theorem rectangular_plot_length_difference (breadth : ℝ) (x : ℝ) : 
  breadth > 0 →
  x > 0 →
  breadth + x = 60 →
  4 * breadth + 2 * x = 200 →
  x = 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_difference_l638_63812


namespace NUMINAMATH_CALUDE_fraction_problem_l638_63850

theorem fraction_problem : ∃ x : ℚ, (0.60 * 40 : ℚ) = x * 25 + 4 :=
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l638_63850


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l638_63833

-- Define the proportional function
def proportional_function (k : ℝ) (x : ℝ) : ℝ := k * x

-- Theorem statement
theorem quadratic_roots_distinct 
  (k : ℝ) 
  (h1 : k ≠ 0)
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → proportional_function k x1 > proportional_function k x2) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - x1 + k - 1 = 0 ∧ x2^2 - x2 + k - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l638_63833


namespace NUMINAMATH_CALUDE_min_distance_sum_l638_63810

theorem min_distance_sum (x : ℝ) : 
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_l638_63810


namespace NUMINAMATH_CALUDE_circle_properties_l638_63865

/-- Given a circle with diameter endpoints (2, 1) and (8, 7), prove its center and diameter length -/
theorem circle_properties :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (8, 7)
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let diameter_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  center = (5, 4) ∧ diameter_length = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l638_63865


namespace NUMINAMATH_CALUDE_chapter_page_difference_l638_63894

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 48) 
  (h2 : second_chapter_pages = 11) : 
  first_chapter_pages - second_chapter_pages = 37 := by
  sorry

end NUMINAMATH_CALUDE_chapter_page_difference_l638_63894


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l638_63841

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 12) % 12 = 0 ∧
  (n - 12) % 24 = 0 ∧
  (n - 12) % 36 = 0 ∧
  (n - 12) % 48 = 0 ∧
  (n - 12) % 56 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 1020 ∧
  ∀ m : ℕ, m < 1020 → ¬(is_divisible_by_all m) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l638_63841


namespace NUMINAMATH_CALUDE_find_S_l638_63803

theorem find_S (R S : ℕ) (h1 : 111111111111 - 222222 = (R + S)^2) (h2 : S > 0) : S = 333332 := by
  sorry

end NUMINAMATH_CALUDE_find_S_l638_63803


namespace NUMINAMATH_CALUDE_marias_trip_distance_l638_63877

/-- Proves that the total distance of Maria's trip is 450 miles -/
theorem marias_trip_distance (D : ℝ) 
  (first_stop : D - D/3 = 2/3 * D)
  (second_stop : 2/3 * D - 1/4 * (2/3 * D) = 1/2 * D)
  (third_stop : 1/2 * D - 1/5 * (1/2 * D) = 2/5 * D)
  (final_distance : 2/5 * D = 180) :
  D = 450 := by sorry

end NUMINAMATH_CALUDE_marias_trip_distance_l638_63877


namespace NUMINAMATH_CALUDE_parabola_translation_l638_63837

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation :
  let p : Parabola := { a := 1, b := 2, c := -1 }
  let translated_p := translate p 2 1
  translated_p = { a := 1, b := -2, c := -3 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l638_63837


namespace NUMINAMATH_CALUDE_equivalent_representations_l638_63857

theorem equivalent_representations (x y z w : ℚ) : 
  x = 1 / 8 ∧ 
  y = 2 / 16 ∧ 
  z = 3 / 24 ∧ 
  w = 125 / 1000 → 
  x = y ∧ y = z ∧ z = w := by
sorry

end NUMINAMATH_CALUDE_equivalent_representations_l638_63857


namespace NUMINAMATH_CALUDE_max_value_at_two_l638_63816

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem max_value_at_two (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 2) → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_at_two_l638_63816


namespace NUMINAMATH_CALUDE_max_contribution_l638_63804

theorem max_contribution (n : ℕ) (total : ℝ) (min_contrib : ℝ) (h1 : n = 15) (h2 : total = 30) (h3 : min_contrib = 1) :
  let max_single := total - (n - 1) * min_contrib
  max_single = 16 := by
sorry

end NUMINAMATH_CALUDE_max_contribution_l638_63804


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l638_63806

theorem cubic_sum_theorem (x y : ℝ) (h : x^3 + 21*x*y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l638_63806


namespace NUMINAMATH_CALUDE_subscription_savings_l638_63888

def category_A_cost : ℝ := 520
def category_B_cost : ℝ := 860
def category_C_cost : ℝ := 620

def category_A_cut_percentage : ℝ := 0.25
def category_B_cut_percentage : ℝ := 0.35
def category_C_cut_percentage : ℝ := 0.30

def total_savings : ℝ :=
  category_A_cost * category_A_cut_percentage +
  category_B_cost * category_B_cut_percentage +
  category_C_cost * category_C_cut_percentage

theorem subscription_savings : total_savings = 617 := by
  sorry

end NUMINAMATH_CALUDE_subscription_savings_l638_63888


namespace NUMINAMATH_CALUDE_volume_common_tetrahedra_l638_63873

/-- Given a parallelepiped ABCDA₁B₁C₁D₁ with volume V, the volume of the common part
    of tetrahedra AB₁CD₁ and A₁BC₁D is V/12 -/
theorem volume_common_tetrahedra (V : ℝ) : ℝ :=
  let parallelepiped_volume := V
  let common_volume := V / 12
  common_volume

#check volume_common_tetrahedra

end NUMINAMATH_CALUDE_volume_common_tetrahedra_l638_63873


namespace NUMINAMATH_CALUDE_luncheon_attendance_l638_63811

theorem luncheon_attendance (invited : ℕ) (table_capacity : ℕ) (tables_used : ℕ) 
  (h1 : invited = 24) 
  (h2 : table_capacity = 7) 
  (h3 : tables_used = 2) : 
  invited - (table_capacity * tables_used) = 10 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_attendance_l638_63811


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l638_63885

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 1 → b = Real.sqrt 2 → B = π / 4 → A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l638_63885


namespace NUMINAMATH_CALUDE_square_minus_two_x_plus_2023_l638_63891

theorem square_minus_two_x_plus_2023 :
  let x : ℝ := 1 + Real.sqrt 3
  x^2 - 2*x + 2023 = 2025 := by sorry

end NUMINAMATH_CALUDE_square_minus_two_x_plus_2023_l638_63891


namespace NUMINAMATH_CALUDE_quadratic_transformation_impossibility_l638_63875

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic trinomial -/
def discriminant (f : QuadraticTrinomial) : ℝ :=
  f.b^2 - 4*f.a*f.c

/-- Represents the allowed operations on quadratic trinomials -/
inductive QuadraticOperation
  | op1 : QuadraticOperation  -- f(x) → x^2 f(1 + 1/x)
  | op2 : QuadraticOperation  -- f(x) → (x-1)^2 f(1/(x-1))

/-- Applies a quadratic operation to a quadratic trinomial -/
def applyOperation (f : QuadraticTrinomial) (op : QuadraticOperation) : QuadraticTrinomial :=
  match op with
  | QuadraticOperation.op1 => QuadraticTrinomial.mk f.a (2*f.a + f.b) (f.a + f.b + f.c)
  | QuadraticOperation.op2 => QuadraticTrinomial.mk f.c (f.b - 2*f.c) (f.a - f.b + f.c)

/-- Theorem stating that it's impossible to transform x^2 + 4x + 3 into x^2 + 10x + 9
    using only the allowed operations -/
theorem quadratic_transformation_impossibility :
  ∀ (ops : List QuadraticOperation),
  let f := QuadraticTrinomial.mk 1 4 3
  let g := QuadraticTrinomial.mk 1 10 9
  let result := ops.foldl applyOperation f
  result ≠ g := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_impossibility_l638_63875


namespace NUMINAMATH_CALUDE_problem_solution_l638_63876

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem problem_solution (a : ℕ → ℝ) : 
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l638_63876


namespace NUMINAMATH_CALUDE_min_distance_exp_ln_curves_l638_63847

/-- The minimum distance between a point on y = e^x and a point on y = ln x is √2 -/
theorem min_distance_exp_ln_curves : ∃ (d : ℝ),
  d = Real.sqrt 2 ∧
  ∀ (x₁ x₂ : ℝ),
    let P := (x₁, Real.exp x₁)
    let Q := (x₂, Real.log x₂)
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_exp_ln_curves_l638_63847


namespace NUMINAMATH_CALUDE_equation_solution_l638_63849

theorem equation_solution : 
  {x : ℝ | x - 12 ≥ 0 ∧ 
    (8 / (Real.sqrt (x - 12) - 10) + 
     2 / (Real.sqrt (x - 12) - 5) + 
     10 / (Real.sqrt (x - 12) + 5) + 
     16 / (Real.sqrt (x - 12) + 10) = 0)} = 
  {208/9, 62} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l638_63849


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l638_63815

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A parallelogram defined by four lattice points -/
structure Parallelogram where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint
  v4 : LatticePoint

/-- Checks if a point is inside or on the edges of a parallelogram (excluding vertices) -/
def isInsideOrOnEdge (p : LatticePoint) (para : Parallelogram) : Prop :=
  sorry

/-- Calculates the area of a parallelogram -/
def area (para : Parallelogram) : ℚ :=
  sorry

theorem parallelogram_area_theorem (para : Parallelogram) :
  (∃ p : LatticePoint, isInsideOrOnEdge p para) → area para > 1 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l638_63815


namespace NUMINAMATH_CALUDE_range_of_half_difference_l638_63802

theorem range_of_half_difference (α β : Real) 
  (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-π/2) 0 ↔ ∃ α' β', -π/2 ≤ α' ∧ α' < β' ∧ β' ≤ π/2 ∧ x = (α' - β') / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_half_difference_l638_63802


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l638_63893

theorem sqrt_3_irrational (numbers : Set ℝ) (h1 : numbers = {-1, 0, (1/2 : ℝ), Real.sqrt 3}) :
  ∃ x ∈ numbers, Irrational x ∧ ∀ y ∈ numbers, y ≠ x → ¬ Irrational y :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l638_63893


namespace NUMINAMATH_CALUDE_hash_12_6_l638_63831

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ := 
  r * s + 2 * r

-- State the theorem
theorem hash_12_6 : hash 12 6 = 96 := by
  -- Identity property
  have h1 : ∀ r : ℝ, hash r 0 = r := by sorry
  
  -- Commutativity property
  have h2 : ∀ r s : ℝ, hash r s = hash s r := by sorry
  
  -- Increment rule
  have h3 : ∀ r s : ℝ, hash (r + 1) s = hash r s + s + 2 := by sorry
  
  -- Prove the main statement
  sorry


end NUMINAMATH_CALUDE_hash_12_6_l638_63831


namespace NUMINAMATH_CALUDE_fourth_group_number_l638_63856

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  sample_size : Nat
  second_group_number : Nat

/-- The number drawn from a specific group in systematic sampling -/
def number_in_group (setup : SystematicSampling) (group : Nat) : Nat :=
  setup.second_group_number + (group - 2) * (setup.total_students / setup.num_groups)

/-- Theorem stating the relationship between the numbers drawn from different groups -/
theorem fourth_group_number (setup : SystematicSampling) 
  (h1 : setup.total_students = 72)
  (h2 : setup.num_groups = 6)
  (h3 : setup.sample_size = 6)
  (h4 : setup.second_group_number = 16) :
  number_in_group setup 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_number_l638_63856


namespace NUMINAMATH_CALUDE_choose_two_from_three_l638_63835

theorem choose_two_from_three : Nat.choose 3 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l638_63835


namespace NUMINAMATH_CALUDE_trig_identity_l638_63881

theorem trig_identity (α : Real) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) :
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / 
  (1 + Real.sin (4 * α) + Real.cos (4 * α)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l638_63881


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l638_63863

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

theorem ninth_term_of_arithmetic_sequence 
  (a₁ a₁₇ : ℚ) 
  (h₁ : a₁ = 3/4) 
  (h₁₇ : a₁₇ = 6/7) 
  (h_seq : ∃ d, ∀ n, arithmetic_sequence a₁ d n = a₁ + (n - 1) * d) :
  ∃ d, arithmetic_sequence a₁ d 9 = 45/56 :=
sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l638_63863


namespace NUMINAMATH_CALUDE_correlation_coefficient_is_negative_one_l638_63830

/-- A structure representing a set of sample data points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  h_n : n ≥ 2
  h_not_all_equal : ∃ i j, i ≠ j ∧ x i ≠ x j
  h_on_line : ∀ i, y i = -1/2 * x i + 1

/-- The correlation coefficient of a set of sample data -/
def correlationCoefficient (data : SampleData) : ℝ := sorry

/-- Theorem stating that the correlation coefficient is -1 for the given conditions -/
theorem correlation_coefficient_is_negative_one (data : SampleData) :
  correlationCoefficient data = -1 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_is_negative_one_l638_63830


namespace NUMINAMATH_CALUDE_line_of_symmetry_l638_63884

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

-- Define the line of symmetry
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem line_of_symmetry :
  ∀ (x y : ℝ), 
    (∃ (x' y' : ℝ), circle_O x' y' ∧ circle_C x y ∧ 
      line_l ((x + x')/2) ((y + y')/2) ∧
      (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2) :=
sorry

end NUMINAMATH_CALUDE_line_of_symmetry_l638_63884


namespace NUMINAMATH_CALUDE_dasha_ate_one_bowl_l638_63898

/-- The number of bowls of porridge eaten by each monkey -/
structure MonkeyPorridge where
  masha : ℕ
  dasha : ℕ
  glasha : ℕ
  natasha : ℕ

/-- The conditions of the monkey porridge problem -/
def MonkeyPorridgeConditions (mp : MonkeyPorridge) : Prop :=
  mp.masha + mp.dasha + mp.glasha + mp.natasha = 16 ∧
  mp.glasha + mp.natasha = 9 ∧
  mp.masha > mp.dasha ∧
  mp.masha > mp.glasha ∧
  mp.masha > mp.natasha

theorem dasha_ate_one_bowl (mp : MonkeyPorridge) 
  (h : MonkeyPorridgeConditions mp) : mp.dasha = 1 := by
  sorry

end NUMINAMATH_CALUDE_dasha_ate_one_bowl_l638_63898


namespace NUMINAMATH_CALUDE_parabola_ellipse_shared_focus_l638_63882

/-- Given a parabola and an ellipse with shared focus, prove p = 8 -/
theorem parabola_ellipse_shared_focus (p : ℝ) : 
  p > 0 → 
  (∃ x y, y^2 = 2*p*x) →  -- parabola equation
  (∃ x y, x^2/(3*p) + y^2/p = 1) →  -- ellipse equation
  (∃ x, x = p/2 ∧ x^2 = p^2/4) →  -- focus of parabola
  (∃ x, x^2 = 3*p^2/4) →  -- focus of ellipse
  p = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_ellipse_shared_focus_l638_63882


namespace NUMINAMATH_CALUDE_sequence_general_term_l638_63897

theorem sequence_general_term (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, (2 * n - 1 : ℝ) * a (n + 1) = (2 * n + 1 : ℝ) * a n) →
  ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l638_63897


namespace NUMINAMATH_CALUDE_circle_radius_l638_63824

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l638_63824


namespace NUMINAMATH_CALUDE_min_perimeter_of_triangle_l638_63809

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- A point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- The center of the ellipse -/
def center : ℝ × ℝ := (0, 0)

/-- A line passing through the center of the ellipse -/
structure LineThroughCenter where
  slope : ℝ

/-- Intersection points of the line with the ellipse -/
def intersectionPoints (l : LineThroughCenter) : PointOnEllipse × PointOnEllipse := sorry

/-- One of the foci of the ellipse -/
def focus : ℝ × ℝ := (3, 0)

/-- The perimeter of the triangle formed by two points on the ellipse and the focus -/
def trianglePerimeter (p q : PointOnEllipse) : ℝ := sorry

/-- The statement to be proved -/
theorem min_perimeter_of_triangle : 
  ∀ l : LineThroughCenter, 
  let (p, q) := intersectionPoints l
  18 ≤ trianglePerimeter p q ∧ 
  ∃ l₀ : LineThroughCenter, trianglePerimeter (intersectionPoints l₀).1 (intersectionPoints l₀).2 = 18 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_of_triangle_l638_63809


namespace NUMINAMATH_CALUDE_benton_school_earnings_l638_63814

/-- Represents the total earnings of students from a school -/
def school_earnings (students : ℕ) (days : ℕ) (daily_wage : ℚ) : ℚ :=
  students * days * daily_wage

/-- Calculates the daily wage per student given the total amount and total student-days -/
def calculate_daily_wage (total_amount : ℚ) (total_student_days : ℕ) : ℚ :=
  total_amount / total_student_days

theorem benton_school_earnings :
  let adams_students : ℕ := 4
  let adams_days : ℕ := 4
  let benton_students : ℕ := 5
  let benton_days : ℕ := 6
  let camden_students : ℕ := 6
  let camden_days : ℕ := 7
  let total_amount : ℚ := 780

  let total_student_days : ℕ := 
    adams_students * adams_days + 
    benton_students * benton_days + 
    camden_students * camden_days

  let daily_wage : ℚ := calculate_daily_wage total_amount total_student_days

  let benton_earnings : ℚ := school_earnings benton_students benton_days daily_wage

  ⌊benton_earnings⌋ = 266 :=
by sorry

end NUMINAMATH_CALUDE_benton_school_earnings_l638_63814


namespace NUMINAMATH_CALUDE_average_difference_l638_63801

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 170) : 
  a - c = -120 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l638_63801


namespace NUMINAMATH_CALUDE_calculate_unknown_interest_rate_l638_63895

/-- Proves that for a given principal, time period, and interest rate difference, 
    the unknown rate can be calculated. -/
theorem calculate_unknown_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (known_rate : ℝ) 
  (interest_difference : ℝ) 
  (unknown_rate : ℝ)
  (h1 : principal = 7000)
  (h2 : time = 2)
  (h3 : known_rate = 18)
  (h4 : interest_difference = 840)
  (h5 : principal * (known_rate / 100) * time - principal * (unknown_rate / 100) * time = interest_difference) :
  unknown_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_calculate_unknown_interest_rate_l638_63895


namespace NUMINAMATH_CALUDE_internet_speed_calculation_l638_63860

/-- Represents the internet speed calculation problem -/
theorem internet_speed_calculation 
  (file1 : ℝ) 
  (file2 : ℝ) 
  (file3 : ℝ) 
  (download_time : ℝ) 
  (h1 : file1 = 80) 
  (h2 : file2 = 90) 
  (h3 : file3 = 70) 
  (h4 : download_time = 2) :
  (file1 + file2 + file3) / download_time = 120 := by
  sorry

#check internet_speed_calculation

end NUMINAMATH_CALUDE_internet_speed_calculation_l638_63860


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l638_63843

/-- Represents the possible states of a cell in the game grid -/
inductive Cell
| Empty : Cell
| S : Cell
| O : Cell

/-- Represents the game state -/
structure GameState where
  grid : Vector Cell 2000
  currentPlayer : Nat

/-- Checks if a player has won by forming SOS pattern -/
def hasWon (state : GameState) : Bool :=
  sorry

/-- Checks if the game is a draw -/
def isDraw (state : GameState) : Bool :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Nat

/-- Checks if a strategy is winning for the given player -/
def isWinningStrategy (player : Nat) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy 2 strategy :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l638_63843


namespace NUMINAMATH_CALUDE_remainder_count_l638_63846

theorem remainder_count : 
  (Finset.filter (fun n => Nat.mod 2017 n = 1 ∨ Nat.mod 2017 n = 2) (Finset.range 2018)).card = 43 := by
  sorry

end NUMINAMATH_CALUDE_remainder_count_l638_63846


namespace NUMINAMATH_CALUDE_alyona_floor_l638_63887

/-- Represents a multi-story building with multiple entrances -/
structure Building where
  stories : ℕ
  apartments_per_floor : ℕ
  entrances : ℕ

/-- Calculates the floor number given an apartment number and building structure -/
def floor_number (b : Building) (apartment : ℕ) : ℕ :=
  let apartments_per_entrance := b.stories * b.apartments_per_floor
  let apartments_before_entrance := ((apartment - 1) / apartments_per_entrance) * apartments_per_entrance
  let remaining_apartments := apartment - apartments_before_entrance
  ((remaining_apartments - 1) / b.apartments_per_floor) + 1

/-- Theorem stating that Alyona lives on the 3rd floor -/
theorem alyona_floor :
  ∀ (b : Building),
    b.stories = 9 →
    b.entrances ≥ 10 →
    floor_number b 333 = 3 :=
by sorry

end NUMINAMATH_CALUDE_alyona_floor_l638_63887


namespace NUMINAMATH_CALUDE_sum_of_squares_unique_l638_63861

theorem sum_of_squares_unique (x y z : ℕ+) : 
  (x : ℕ) + y + z = 24 → 
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 → 
  x^2 + y^2 + z^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_unique_l638_63861


namespace NUMINAMATH_CALUDE_player_catches_ball_l638_63828

/-- Represents the motion of an object with uniform acceleration --/
structure UniformMotion where
  initial_velocity : ℝ
  acceleration : ℝ

/-- Calculates the distance traveled by an object with uniform motion --/
def distance (m : UniformMotion) (t : ℝ) : ℝ :=
  m.initial_velocity * t + 0.5 * m.acceleration * t^2

theorem player_catches_ball (ball_motion player_motion : UniformMotion)
  (initial_distance sideline_distance : ℝ) : 
  ball_motion.initial_velocity = 4.375 ∧ 
  ball_motion.acceleration = -0.75 ∧
  player_motion.initial_velocity = 3.25 ∧
  player_motion.acceleration = 0.5 ∧
  initial_distance = 10 ∧
  sideline_distance = 23 →
  ∃ (t : ℝ), t = 5 ∧ 
  distance ball_motion t = distance player_motion t + initial_distance ∧
  distance ball_motion t < sideline_distance :=
by sorry

#check player_catches_ball

end NUMINAMATH_CALUDE_player_catches_ball_l638_63828


namespace NUMINAMATH_CALUDE_sin_210_degrees_l638_63890

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l638_63890


namespace NUMINAMATH_CALUDE_f_is_odd_iff_a_eq_one_l638_63855

/-- A function f is odd if f(-x) = -f(x) for all x in its domain. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x(x-1)(x+a) -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - 1) * (x + a)

/-- Theorem: f(x) = x(x-1)(x+a) is an odd function if and only if a = 1 -/
theorem f_is_odd_iff_a_eq_one (a : ℝ) : IsOdd (f a) ↔ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_f_is_odd_iff_a_eq_one_l638_63855


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l638_63871

theorem floor_of_expression_equals_eight :
  ⌊(1005^3 : ℝ) / (1003 * 1004) - (1003^3 : ℝ) / (1004 * 1005)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l638_63871


namespace NUMINAMATH_CALUDE_total_supplies_is_1260_l638_63808

/-- The total number of supplies given the number of rows and items per row -/
def total_supplies (rows : ℕ) (crayons_per_row : ℕ) (colored_pencils_per_row : ℕ) (graphite_pencils_per_row : ℕ) : ℕ :=
  rows * (crayons_per_row + colored_pencils_per_row + graphite_pencils_per_row)

/-- Theorem stating that the total number of supplies is 1260 -/
theorem total_supplies_is_1260 : total_supplies 28 12 15 18 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_total_supplies_is_1260_l638_63808


namespace NUMINAMATH_CALUDE_indexCardsCostForCarl_l638_63845

/-- Represents the cost of index cards for Carl's students. -/
def indexCardsCost (
  sixthGradeCards : ℕ
  ) (seventhGradeCards : ℕ
  ) (eighthGradeCards : ℕ
  ) (periodsPerDay : ℕ
  ) (sixthGradersPerPeriod : ℕ
  ) (seventhGradersPerPeriod : ℕ
  ) (eighthGradersPerPeriod : ℕ
  ) (cardsPerPack : ℕ
  ) (costPerPack : ℕ
  ) : ℕ :=
  let totalCards := 
    (sixthGradeCards * sixthGradersPerPeriod + 
     seventhGradeCards * seventhGradersPerPeriod + 
     eighthGradeCards * eighthGradersPerPeriod) * periodsPerDay
  let packsNeeded := (totalCards + cardsPerPack - 1) / cardsPerPack
  packsNeeded * costPerPack

/-- Theorem stating the total cost of index cards for Carl's students. -/
theorem indexCardsCostForCarl : 
  indexCardsCost 8 10 12 6 20 25 30 50 3 = 279 := by
  sorry

end NUMINAMATH_CALUDE_indexCardsCostForCarl_l638_63845


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l638_63899

theorem last_two_digits_sum (n : ℕ) : n = 7^15 + 13^15 → n % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l638_63899


namespace NUMINAMATH_CALUDE_mean_median_difference_l638_63862

/-- Represents the frequency distribution of days missed --/
structure FrequencyDistribution :=
  (zero_days : Nat)
  (one_day : Nat)
  (two_days : Nat)
  (three_days : Nat)
  (four_days : Nat)
  (five_days : Nat)

/-- Calculates the median of the dataset --/
def median (fd : FrequencyDistribution) : Rat :=
  2

/-- Calculates the mean of the dataset --/
def mean (fd : FrequencyDistribution) : Rat :=
  (0 * fd.zero_days + 1 * fd.one_day + 2 * fd.two_days + 
   3 * fd.three_days + 4 * fd.four_days + 5 * fd.five_days) / 20

/-- The main theorem to prove --/
theorem mean_median_difference 
  (fd : FrequencyDistribution) 
  (h1 : fd.zero_days = 4)
  (h2 : fd.one_day = 2)
  (h3 : fd.two_days = 5)
  (h4 : fd.three_days = 3)
  (h5 : fd.four_days = 2)
  (h6 : fd.five_days = 4)
  (h7 : fd.zero_days + fd.one_day + fd.two_days + fd.three_days + fd.four_days + fd.five_days = 20) :
  mean fd - median fd = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l638_63862


namespace NUMINAMATH_CALUDE_print_shop_charge_l638_63838

/-- The charge per color copy at print shop X -/
def charge_X : ℝ := 1.25

/-- The charge per color copy at print shop Y -/
def charge_Y : ℝ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 60

/-- The difference in total charge between print shops Y and X -/
def charge_difference : ℝ := 90

theorem print_shop_charge : 
  charge_X * num_copies + charge_difference = charge_Y * num_copies := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_l638_63838


namespace NUMINAMATH_CALUDE_square_minus_two_x_plus_one_l638_63800

theorem square_minus_two_x_plus_one (x : ℝ) : x = Real.sqrt 3 + 1 → x^2 - 2*x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_two_x_plus_one_l638_63800


namespace NUMINAMATH_CALUDE_average_people_moving_per_hour_l638_63825

/-- The number of people moving to Texas in 5 days -/
def people_moving : ℕ := 3500

/-- The number of days -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem average_people_moving_per_hour :
  round_to_nearest average_per_hour = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_people_moving_per_hour_l638_63825


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l638_63821

theorem stratified_sampling_theorem (first_grade second_grade third_grade total_selected : ℕ) 
  (h1 : first_grade = 120)
  (h2 : second_grade = 180)
  (h3 : third_grade = 150)
  (h4 : total_selected = 90) :
  let total_students := first_grade + second_grade + third_grade
  let sampling_ratio := total_selected / total_students
  (sampling_ratio * first_grade : ℕ) = 24 ∧
  (sampling_ratio * second_grade : ℕ) = 36 ∧
  (sampling_ratio * third_grade : ℕ) = 30 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l638_63821


namespace NUMINAMATH_CALUDE_joe_weight_loss_l638_63892

/-- Represents Joe's weight loss problem --/
theorem joe_weight_loss 
  (initial_weight : ℝ) 
  (months_on_diet : ℝ) 
  (future_weight : ℝ) 
  (months_until_future_weight : ℝ) 
  (h1 : initial_weight = 222)
  (h2 : months_on_diet = 3)
  (h3 : future_weight = 170)
  (h4 : months_until_future_weight = 3.5)
  : ∃ (current_weight : ℝ), 
    current_weight = initial_weight - (initial_weight - future_weight) * (months_on_diet / (months_on_diet + months_until_future_weight))
    ∧ current_weight = 198 :=
by sorry

end NUMINAMATH_CALUDE_joe_weight_loss_l638_63892


namespace NUMINAMATH_CALUDE_ratio_chain_l638_63813

theorem ratio_chain (a b c d e f g h : ℝ) 
  (hab : a / b = 7 / 3)
  (hbc : b / c = 5 / 2)
  (hcd : c / d = 2)
  (hde : d / e = 3 / 2)
  (hef : e / f = 4 / 3)
  (hfg : f / g = 1 / 4)
  (hgh : g / h = 3 / 5) :
  a * b * c * d * e * f * g / (d * e * f * g * h * i * j) = 15.75 :=
by sorry

end NUMINAMATH_CALUDE_ratio_chain_l638_63813


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l638_63869

theorem sqrt_equation_solution (z : ℝ) :
  (Real.sqrt 1.5 / Real.sqrt 0.81 + Real.sqrt z / Real.sqrt 0.49 = 3.0751133491652576) →
  z = 1.44 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l638_63869


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l638_63879

theorem imaginary_part_of_z (i : ℂ) : i * i = -1 → Complex.im ((1 + i) / i) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l638_63879


namespace NUMINAMATH_CALUDE_cube_edge_length_is_5_l638_63872

/-- The edge length of a cube immersed in water --/
def cube_edge_length (base_length base_width water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that the edge length of the cube is 5 cm --/
theorem cube_edge_length_is_5 :
  cube_edge_length 10 5 2.5 = 5 := by sorry

end NUMINAMATH_CALUDE_cube_edge_length_is_5_l638_63872


namespace NUMINAMATH_CALUDE_smallest_quotient_by_18_l638_63851

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_quotient_by_18 (U : ℕ) (hU : is_binary_number U) (hDiv : U % 18 = 0) :
  ∃ Y : ℕ, Y = U / 18 ∧ Y ≥ 61728395 ∧ (∀ Z : ℕ, (∃ V : ℕ, is_binary_number V ∧ V % 18 = 0 ∧ Z = V / 18) → Z ≥ Y) :=
sorry

end NUMINAMATH_CALUDE_smallest_quotient_by_18_l638_63851


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l638_63870

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem quadratic_inequality_negation : 
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l638_63870


namespace NUMINAMATH_CALUDE_max_value_of_f_l638_63839

-- Define the function f(x) = x(4 - x)
def f (x : ℝ) := x * (4 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ x, 0 < x ∧ x < 4 → f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l638_63839


namespace NUMINAMATH_CALUDE_stratified_sampling_type_D_l638_63868

/-- Calculates the number of units to be selected from a specific product type in stratified sampling -/
def stratifiedSampleSize (totalProduction : ℕ) (typeProduction : ℕ) (totalSample : ℕ) : ℕ :=
  (typeProduction * totalSample) / totalProduction

/-- The problem statement -/
theorem stratified_sampling_type_D :
  let totalProduction : ℕ := 100 + 200 + 300 + 400
  let typeDProduction : ℕ := 400
  let totalSample : ℕ := 50
  stratifiedSampleSize totalProduction typeDProduction totalSample = 20 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_type_D_l638_63868


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l638_63823

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, n > 0 ∧
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 ∧ m < n →
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ,
      Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l638_63823


namespace NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l638_63805

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle for rotational symmetry in degrees -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ p : RegularPolygon 17,
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 649 / 17 := by
  sorry

end NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l638_63805


namespace NUMINAMATH_CALUDE_subset_intersection_implies_empty_complement_l638_63853

theorem subset_intersection_implies_empty_complement
  (A B : Set ℝ) (h : A ⊆ A ∩ B) : A ∩ (Set.univ \ B) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_implies_empty_complement_l638_63853


namespace NUMINAMATH_CALUDE_sin_cos_alpha_abs_value_l638_63864

theorem sin_cos_alpha_abs_value (α : Real) 
  (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  |Real.sin α * Real.cos α| = 2/5 := by sorry

end NUMINAMATH_CALUDE_sin_cos_alpha_abs_value_l638_63864


namespace NUMINAMATH_CALUDE_madan_age_is_five_l638_63854

-- Define the ages as natural numbers
def arun_age : ℕ := 60

-- Define Gokul's age as a function of Arun's age
def gokul_age (a : ℕ) : ℕ := (a - 6) / 18

-- Define Madan's age as a function of Gokul's age
def madan_age (g : ℕ) : ℕ := g + 2

-- Theorem to prove
theorem madan_age_is_five :
  madan_age (gokul_age arun_age) = 5 := by
  sorry

end NUMINAMATH_CALUDE_madan_age_is_five_l638_63854


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l638_63832

/-- The number of perfect square factors of 360 -/
def perfectSquareFactors : ℕ := 4

/-- The prime factorization of 360 -/
def primeFactorization : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

/-- Theorem stating that the number of perfect square factors of 360 is 4 -/
theorem count_perfect_square_factors :
  (List.sum (List.map (fun (p : ℕ × ℕ) => (p.2 / 2 + 1)) primeFactorization)) = perfectSquareFactors := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l638_63832


namespace NUMINAMATH_CALUDE_tau_fraction_values_l638_63820

/-- The number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The number of positive divisors of n which have remainders 1 when divided by 3 -/
def τ₁ (n : ℕ+) : ℕ := sorry

/-- A number is composite if it's greater than 1 and not prime -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

/-- The set of possible values for τ(10n) / τ₁(10n) -/
def possibleValues : Set ℕ := {n | n % 2 = 0 ∨ isComposite n}

/-- The main theorem -/
theorem tau_fraction_values (n : ℕ+) : 
  ∃ (k : ℕ), k ∈ possibleValues ∧ (τ (10 * n) : ℚ) / τ₁ (10 * n) = k := by sorry

end NUMINAMATH_CALUDE_tau_fraction_values_l638_63820


namespace NUMINAMATH_CALUDE_fraction_equals_93_l638_63858

theorem fraction_equals_93 : (3025 - 2880)^2 / 225 = 93 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_93_l638_63858


namespace NUMINAMATH_CALUDE_exists_subsequences_forming_2520_l638_63840

def infinite_sequence : ℕ → ℕ
  | n => match n % 6 with
         | 0 => 2
         | 1 => 0
         | 2 => 1
         | 3 => 5
         | 4 => 2
         | 5 => 0
         | _ => 0  -- This case should never occur

def is_subsequence (s : List ℕ) : Prop :=
  ∃ start : ℕ, ∀ i : ℕ, i < s.length → s.get ⟨i, by sorry⟩ = infinite_sequence (start + i)

def concatenate_to_number (s1 s2 : List ℕ) : ℕ :=
  (s1 ++ s2).foldl (λ acc d => acc * 10 + d) 0

theorem exists_subsequences_forming_2520 :
  ∃ (s1 s2 : List ℕ),
    s1 ≠ [] ∧ s2 ≠ [] ∧
    is_subsequence s1 ∧
    is_subsequence s2 ∧
    concatenate_to_number s1 s2 = 2520 ∧
    2520 % 45 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_subsequences_forming_2520_l638_63840


namespace NUMINAMATH_CALUDE_algebraic_arithmetic_equivalence_l638_63880

theorem algebraic_arithmetic_equivalence (a b : ℕ) (h : a > b) :
  (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_arithmetic_equivalence_l638_63880


namespace NUMINAMATH_CALUDE_complex_multiplication_l638_63874

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : i * (2 * i + 1) = -2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l638_63874


namespace NUMINAMATH_CALUDE_range_of_x_plus_y_l638_63848

theorem range_of_x_plus_y (x y : ℝ) (h : x - Real.sqrt (x + 1) = Real.sqrt (y + 1) - y) :
  ∃ (s : ℝ), s ∈ Set.Icc (1 - Real.sqrt 5) (1 + Real.sqrt 5) ∧ x + y = s :=
sorry

end NUMINAMATH_CALUDE_range_of_x_plus_y_l638_63848


namespace NUMINAMATH_CALUDE_digits_1198_to_1200_form_473_l638_63889

/-- A function that generates the list of positive integers with first digit 1 or 2 -/
def firstDigitOneOrTwo : ℕ → Bool := sorry

/-- The number of digits written before reaching a given position in the list -/
def digitCount (n : ℕ) : ℕ := sorry

/-- The number at a given position in the list -/
def numberAtPosition (n : ℕ) : ℕ := sorry

theorem digits_1198_to_1200_form_473 :
  let pos := 1198
  ∃ (n : ℕ), 
    firstDigitOneOrTwo n ∧ 
    digitCount n ≤ pos ∧ 
    digitCount (n + 1) > pos + 2 ∧
    numberAtPosition n = 473 := by sorry

end NUMINAMATH_CALUDE_digits_1198_to_1200_form_473_l638_63889


namespace NUMINAMATH_CALUDE_second_tank_volume_l638_63836

/-- Represents the capacity of each tank in liters -/
def tank_capacity : ℝ := 1000

/-- Represents the volume of water in the first tank in liters -/
def first_tank_volume : ℝ := 300

/-- Represents the fraction of the second tank that is filled -/
def second_tank_fill_ratio : ℝ := 0.45

/-- Represents the additional water needed to fill both tanks in liters -/
def additional_water_needed : ℝ := 1250

/-- Theorem stating that the second tank contains 450 liters of water -/
theorem second_tank_volume :
  let second_tank_volume := second_tank_fill_ratio * tank_capacity
  second_tank_volume = 450 := by sorry

end NUMINAMATH_CALUDE_second_tank_volume_l638_63836


namespace NUMINAMATH_CALUDE_angle_value_proof_l638_63883

theorem angle_value_proof (ABC : ℝ) (x : ℝ) 
  (h1 : ABC = 90)
  (h2 : ABC = 44 + x) : 
  x = 46 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_proof_l638_63883


namespace NUMINAMATH_CALUDE_mike_washed_nine_cars_l638_63878

/-- Represents the time in minutes for various car maintenance tasks -/
structure CarMaintenanceTimes where
  washTime : ℕ
  oilChangeTime : ℕ
  tireChangeTime : ℕ

/-- Represents the tasks Mike performed -/
structure MikesTasks where
  oilChanges : ℕ
  tireChanges : ℕ
  totalWorkTime : ℕ

/-- Calculates the number of cars Mike washed given the maintenance times and his tasks -/
def carsWashed (times : CarMaintenanceTimes) (tasks : MikesTasks) : ℕ :=
  let remainingTime := tasks.totalWorkTime - 
    (tasks.oilChanges * times.oilChangeTime + tasks.tireChanges * times.tireChangeTime)
  remainingTime / times.washTime

/-- Theorem stating that Mike washed 9 cars given the problem conditions -/
theorem mike_washed_nine_cars : 
  let times : CarMaintenanceTimes := ⟨10, 15, 30⟩
  let tasks : MikesTasks := ⟨6, 2, 4 * 60⟩
  carsWashed times tasks = 9 := by
  sorry

end NUMINAMATH_CALUDE_mike_washed_nine_cars_l638_63878


namespace NUMINAMATH_CALUDE_matildas_chocolate_bars_l638_63834

/-- Proves that Matilda initially had 4 chocolate bars given the problem conditions -/
theorem matildas_chocolate_bars (total_people : ℕ) (sisters : ℕ) (fathers_remaining : ℕ) 
  (mothers_share : ℕ) (fathers_eaten : ℕ) :
  total_people = sisters + 1 →
  sisters = 4 →
  fathers_remaining = 5 →
  mothers_share = 3 →
  fathers_eaten = 2 →
  ∃ (initial_bars : ℕ),
    initial_bars = (fathers_remaining + mothers_share + fathers_eaten) * 2 / total_people ∧
    initial_bars = 4 :=
by sorry

end NUMINAMATH_CALUDE_matildas_chocolate_bars_l638_63834


namespace NUMINAMATH_CALUDE_newspaper_cost_8_weeks_l638_63827

/-- The cost of newspapers over a period of weeks -/
def newspaper_cost (weekday_price : ℚ) (sunday_price : ℚ) (num_weeks : ℕ) : ℚ :=
  (3 * weekday_price + sunday_price) * num_weeks

/-- Proof that the total cost of newspapers for 8 weeks is $28.00 -/
theorem newspaper_cost_8_weeks :
  newspaper_cost 0.5 2 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_cost_8_weeks_l638_63827


namespace NUMINAMATH_CALUDE_milk_container_problem_l638_63829

/-- Proves that the initial quantity of milk in container B was 37.5% of container A's capacity -/
theorem milk_container_problem (A B C : ℝ) : 
  A = 1200 →  -- Container A's capacity is 1200 liters
  B + C = A →  -- All milk from A was poured into B and C
  (B + 150 = C - 150) →  -- After transferring 150 liters from C to B, both containers have equal quantities
  B / A = 0.375 :=  -- The initial quantity in B was 37.5% of A's capacity
by sorry

end NUMINAMATH_CALUDE_milk_container_problem_l638_63829


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_progression_sides_and_perimeter_15_l638_63818

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_with_arithmetic_progression_sides_and_perimeter_15 :
  ∀ a b c : ℕ,
    a + b + c = 15 →
    is_arithmetic_progression a b c →
    is_valid_triangle a b c →
    ((a = 5 ∧ b = 5 ∧ c = 5) ∨
     (a = 4 ∧ b = 5 ∧ c = 6) ∨
     (a = 3 ∧ b = 5 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_progression_sides_and_perimeter_15_l638_63818


namespace NUMINAMATH_CALUDE_sixth_term_is_geometric_mean_l638_63817

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

/-- The second term is the geometric mean of the first and fourth terms -/
def SecondTermIsGeometricMean (a : ℕ → ℝ) : Prop :=
  a 2 = Real.sqrt (a 1 * a 4)

theorem sixth_term_is_geometric_mean
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_second : SecondTermIsGeometricMean a) :
  a 6 = Real.sqrt (a 4 * a 9) :=
by sorry

end NUMINAMATH_CALUDE_sixth_term_is_geometric_mean_l638_63817


namespace NUMINAMATH_CALUDE_tan_90_degrees_undefined_l638_63844

theorem tan_90_degrees_undefined :
  let θ : Real := 90 * Real.pi / 180  -- Convert 90 degrees to radians
  ∀ (tan sin cos : Real → Real),
    (∀ α, tan α = sin α / cos α) →    -- Definition of tangent
    sin θ = 1 →                       -- Given: sin 90° = 1
    cos θ = 0 →                       -- Given: cos 90° = 0
    ¬∃ (x : Real), tan θ = x          -- tan 90° is undefined
  := by sorry

end NUMINAMATH_CALUDE_tan_90_degrees_undefined_l638_63844


namespace NUMINAMATH_CALUDE_tan_equality_implies_75_l638_63822

theorem tan_equality_implies_75 (n : ℤ) (h1 : -90 < n) (h2 : n < 90) :
  Real.tan (n • π / 180) = Real.tan (255 • π / 180) → n = 75 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_implies_75_l638_63822


namespace NUMINAMATH_CALUDE_hot_dog_sales_l638_63867

theorem hot_dog_sales (total : ℕ) (first_innings : ℕ) (left_unsold : ℕ) :
  total = 91 →
  first_innings = 19 →
  left_unsold = 45 →
  total - (first_innings + left_unsold) = 27 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_sales_l638_63867


namespace NUMINAMATH_CALUDE_probability_is_four_twentyfirsts_l638_63859

/-- Represents a person with a unique age --/
structure Person :=
  (age : ℕ)

/-- The set of all possible orderings of people leaving the meeting --/
def Orderings : Type := List Person

/-- Checks if the youngest person leaves before the oldest in a given ordering --/
def youngest_before_oldest (ordering : Orderings) : Prop :=
  sorry

/-- Checks if the 3rd, 4th, and 5th people in the ordering are in ascending age order --/
def middle_three_ascending (ordering : Orderings) : Prop :=
  sorry

/-- The set of all valid orderings (where youngest leaves before oldest) --/
def valid_orderings (people : Finset Person) : Finset Orderings :=
  sorry

/-- The probability of the event occurring --/
def probability (people : Finset Person) : ℚ :=
  sorry

theorem probability_is_four_twentyfirsts 
  (people : Finset Person) 
  (h1 : people.card = 7) 
  (h2 : ∀ p q : Person, p ∈ people → q ∈ people → p ≠ q → p.age ≠ q.age) : 
  probability people = 4 / 21 :=
sorry

end NUMINAMATH_CALUDE_probability_is_four_twentyfirsts_l638_63859


namespace NUMINAMATH_CALUDE_f_relationship_l638_63842

/-- A quadratic function f(x) = 3x^2 + ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + a * x + b

/-- f(x-1) is an even function -/
def f_shifted_is_even (a b : ℝ) : Prop :=
  ∀ x, f a b (x - 1) = f a b (-x - 1)

theorem f_relationship (a b : ℝ) (h : f_shifted_is_even a b) :
  f a b (-1) < f a b (-3/2) ∧ f a b (-3/2) < f a b (3/2) := by
  sorry

end NUMINAMATH_CALUDE_f_relationship_l638_63842


namespace NUMINAMATH_CALUDE_camp_acquaintances_l638_63852

/-- Represents the number of acquaintances of a child -/
def Acquaintances : Type := ℕ

/-- Represents a child in the group -/
structure Child :=
  (name : String)
  (acquaintances : Acquaintances)

/-- The fraction of one child's acquaintances who are also acquainted with another child -/
def mutualAcquaintanceFraction (a b : Child) : ℚ := sorry

/-- Petya, one of the children in the group -/
def petya : Child := ⟨"Petya", sorry⟩

/-- Vasya, one of the children in the group -/
def vasya : Child := ⟨"Vasya", sorry⟩

/-- Timofey, one of the children in the group -/
def timofey : Child := ⟨"Timofey", sorry⟩

theorem camp_acquaintances :
  (mutualAcquaintanceFraction petya vasya = 1/2) →
  (mutualAcquaintanceFraction petya timofey = 1/7) →
  (mutualAcquaintanceFraction vasya petya = 1/3) →
  (mutualAcquaintanceFraction vasya timofey = 1/6) →
  (mutualAcquaintanceFraction timofey petya = 1/5) →
  (mutualAcquaintanceFraction timofey vasya = 7/20) :=
by sorry

end NUMINAMATH_CALUDE_camp_acquaintances_l638_63852
