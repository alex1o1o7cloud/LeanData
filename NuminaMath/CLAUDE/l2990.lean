import Mathlib

namespace g_equiv_l2990_299005

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- Define the function g using f
def g (x : ℝ) : ℝ := 2 * (f x) - 19

-- Theorem stating that g(x) is equivalent to 6x - 29
theorem g_equiv : ∀ x : ℝ, g x = 6 * x - 29 := by
  sorry

end g_equiv_l2990_299005


namespace complex_fraction_sum_l2990_299007

/-- Given that i is the imaginary unit and (2+i)/(1+i) = a + bi where a and b are real numbers,
    prove that a + b = 1 -/
theorem complex_fraction_sum (i : ℂ) (a b : ℝ) 
    (h1 : i * i = -1) 
    (h2 : (2 + i) / (1 + i) = a + b * i) : 
  a + b = 1 := by
  sorry

end complex_fraction_sum_l2990_299007


namespace parabola_max_vertex_sum_l2990_299009

theorem parabola_max_vertex_sum (a S : ℤ) (h : S ≠ 0) :
  let parabola (x y : ℚ) := ∃ b c : ℚ, y = a * x^2 + b * x + c
  let passes_through (x y : ℚ) := parabola x y
  let vertex_sum := 
    let x₀ : ℚ := (3 * S : ℚ) / 2
    let y₀ : ℚ := -((9 * S^2 : ℚ) / 4) * a
    x₀ + y₀
  (passes_through 0 0) ∧ 
  (passes_through (3 * S) 0) ∧ 
  (passes_through (3 * S - 2) 35) →
  (∀ M : ℚ, vertex_sum ≤ M → M ≤ 1485/4)
  :=
by sorry

end parabola_max_vertex_sum_l2990_299009


namespace binomial_distribution_p_value_l2990_299002

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_p_value 
  (X : BinomialDistribution) 
  (h_exp : expectedValue X = 300)
  (h_var : variance X = 200) :
  X.p = 1/3 := by
sorry

end binomial_distribution_p_value_l2990_299002


namespace triangle_area_l2990_299016

/-- Given vectors m and n, and function f, prove the area of triangle ABC -/
theorem triangle_area (x : ℝ) :
  let m : ℝ × ℝ := (Real.sqrt 3 * Real.sin x - Real.cos x, 1)
  let n : ℝ × ℝ := (Real.cos x, 1/2)
  let f : ℝ → ℝ := λ x => m.1 * n.1 + m.2 * n.2
  let a : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 4
  ∀ A : ℝ, f A = 1 →
    ∃ b : ℝ, 
      let s := (a + b + c) / 2
      2 * Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 2 * Real.sqrt 3 :=
by sorry

end triangle_area_l2990_299016


namespace probability_even_product_l2990_299077

def setA : Finset ℕ := {1, 2, 3, 4}
def setB : Finset ℕ := {5, 6, 7, 8}

def isEven (n : ℕ) : Bool := n % 2 = 0

def evenProductPairs : Finset (ℕ × ℕ) :=
  setA.product setB |>.filter (fun (a, b) => isEven (a * b))

theorem probability_even_product :
  (evenProductPairs.card : ℚ) / ((setA.card * setB.card) : ℚ) = 3/4 := by
  sorry

end probability_even_product_l2990_299077


namespace andrew_payment_l2990_299099

/-- The total amount Andrew paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_weight : ℕ) (grape_rate : ℕ) (mango_weight : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_weight * grape_rate + mango_weight * mango_rate

/-- Theorem stating that Andrew paid 908 to the shopkeeper -/
theorem andrew_payment : total_amount 7 68 9 48 = 908 := by
  sorry

end andrew_payment_l2990_299099


namespace segment_length_l2990_299027

/-- Given a line segment CD with points R and S on it, prove that CD has length 146.2/11 -/
theorem segment_length (C D R S : ℝ) : 
  (R > C) →  -- R is to the right of C
  (S > R) →  -- S is to the right of R
  (D > S) →  -- D is to the right of S
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 4 / 7 →  -- S divides CD in ratio 4:7
  S - R = 1 →  -- RS = 1
  D - C = 146.2 / 11 := by
sorry

end segment_length_l2990_299027


namespace original_number_proof_l2990_299087

theorem original_number_proof (x : ℝ) (h : x * 1.25 = 250) : x = 200 := by
  sorry

end original_number_proof_l2990_299087


namespace combination_distinctness_and_divisor_count_l2990_299086

theorem combination_distinctness_and_divisor_count (n : ℕ) (hn : n > 3) :
  -- Part (a)
  (∀ x y z : ℕ, x > n / 2 → y > n / 2 → z > n / 2 → x < y → y < z → z ≤ n →
    (let exprs := [x + y + z, x + y * z, x * y + z, y + z * x, (x + y) * z, (z + x) * y, (y + z) * x, x * y * z]
     exprs.Pairwise (·≠·))) ∧
  -- Part (b)
  (∀ p : ℕ, Nat.Prime p → p ≤ Real.sqrt n →
    (Finset.filter (fun i => i > 1 ∧ (p - 1) % i = 0) (Finset.range (p - 1))).card =
    (Finset.filter (fun pair : ℕ × ℕ =>
      let (y, z) := pair
      p < y ∧ y < z ∧ z ≤ n ∧
      ¬(let exprs := [p + y + z, p + y * z, p * y + z, y + z * p, (p + y) * z, (z + p) * y, (y + z) * p, p * y * z]
        exprs.Pairwise (·≠·)))
     (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card) :=
by sorry

end combination_distinctness_and_divisor_count_l2990_299086


namespace problem_statement_l2990_299075

theorem problem_statement (x y : ℝ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end problem_statement_l2990_299075


namespace seven_arithmetic_to_hundred_l2990_299088

theorem seven_arithmetic_to_hundred : (777 / 7) - (77 / 7) = 100 := by sorry

end seven_arithmetic_to_hundred_l2990_299088


namespace cara_don_meeting_l2990_299030

/-- The distance Cara walks before meeting Don -/
def distance_cara_walks : ℝ := 18

/-- The total distance between Cara's and Don's homes -/
def total_distance : ℝ := 45

/-- Cara's walking speed in km/h -/
def cara_speed : ℝ := 6

/-- Don's walking speed in km/h -/
def don_speed : ℝ := 5

/-- The time Don starts walking after Cara (in hours) -/
def don_start_delay : ℝ := 2

theorem cara_don_meeting :
  distance_cara_walks = 18 ∧
  distance_cara_walks + don_speed * (distance_cara_walks / cara_speed) =
    total_distance - cara_speed * don_start_delay :=
sorry

end cara_don_meeting_l2990_299030


namespace height_edge_relationship_l2990_299006

/-- A triangular pyramid with mutually perpendicular edges -/
structure TriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h
  perpendicular : True  -- Represents that SA, SB, and SC are mutually perpendicular

/-- The theorem about the relationship between height and edge lengths in a triangular pyramid -/
theorem height_edge_relationship (p : TriangularPyramid) : 
  1 / p.h^2 = 1 / p.a^2 + 1 / p.b^2 + 1 / p.c^2 := by
  sorry

end height_edge_relationship_l2990_299006


namespace scooter_cost_l2990_299057

/-- The cost of a scooter given the amount saved and the additional amount needed. -/
theorem scooter_cost (saved : ℕ) (needed : ℕ) (cost : ℕ) 
  (h1 : saved = 57) 
  (h2 : needed = 33) : 
  cost = saved + needed := by
  sorry

end scooter_cost_l2990_299057


namespace sum_of_legs_is_462_l2990_299068

/-- A right triangle with two inscribed squares -/
structure RightTriangleWithSquares where
  -- The right triangle
  AC : ℝ
  CB : ℝ
  -- The two inscribed squares
  S1 : ℝ
  S2 : ℝ
  -- Conditions
  right_triangle : AC^2 + CB^2 = (AC + CB)^2 / 2
  area_S1 : S1^2 = 441
  area_S2 : S2^2 = 440

/-- The sum of the legs of the right triangle is 462 -/
theorem sum_of_legs_is_462 (t : RightTriangleWithSquares) : t.AC + t.CB = 462 := by
  sorry

end sum_of_legs_is_462_l2990_299068


namespace power_four_mod_nine_l2990_299020

theorem power_four_mod_nine : 4^215 % 9 = 7 := by
  sorry

end power_four_mod_nine_l2990_299020


namespace at_least_one_geq_two_l2990_299065

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) := by
  sorry

end at_least_one_geq_two_l2990_299065


namespace power_value_from_equation_l2990_299001

theorem power_value_from_equation (x y : ℝ) 
  (h : |x - 2| + Real.sqrt (y + 3) = 0) : 
  y ^ x = 9 := by sorry

end power_value_from_equation_l2990_299001


namespace no_five_circle_arrangement_l2990_299076

-- Define a structure for a point in a plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is the circumcenter of a triangle
def isCircumcenter (p : Point2D) (a b c : Point2D) : Prop :=
  (p.x - a.x)^2 + (p.y - a.y)^2 = (p.x - b.x)^2 + (p.y - b.y)^2 ∧
  (p.x - b.x)^2 + (p.y - b.y)^2 = (p.x - c.x)^2 + (p.y - c.y)^2

-- Theorem statement
theorem no_five_circle_arrangement :
  ¬ ∃ (p₁ p₂ p₃ p₄ p₅ : Point2D),
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    (isCircumcenter p₁ p₂ p₃ p₄ ∨ isCircumcenter p₁ p₂ p₃ p₅ ∨ isCircumcenter p₁ p₂ p₄ p₅ ∨ isCircumcenter p₁ p₃ p₄ p₅) ∧
    (isCircumcenter p₂ p₁ p₃ p₄ ∨ isCircumcenter p₂ p₁ p₃ p₅ ∨ isCircumcenter p₂ p₁ p₄ p₅ ∨ isCircumcenter p₂ p₃ p₄ p₅) ∧
    (isCircumcenter p₃ p₁ p₂ p₄ ∨ isCircumcenter p₃ p₁ p₂ p₅ ∨ isCircumcenter p₃ p₁ p₄ p₅ ∨ isCircumcenter p₃ p₂ p₄ p₅) ∧
    (isCircumcenter p₄ p₁ p₂ p₃ ∨ isCircumcenter p₄ p₁ p₂ p₅ ∨ isCircumcenter p₄ p₁ p₃ p₅ ∨ isCircumcenter p₄ p₂ p₃ p₅) ∧
    (isCircumcenter p₅ p₁ p₂ p₃ ∨ isCircumcenter p₅ p₁ p₂ p₄ ∨ isCircumcenter p₅ p₁ p₃ p₄ ∨ isCircumcenter p₅ p₂ p₃ p₄) :=
by
  sorry


end no_five_circle_arrangement_l2990_299076


namespace speed_difference_calc_l2990_299071

/-- Calculates the speed difference between return and outbound trips --/
theorem speed_difference_calc (outbound_time outbound_speed return_time : ℝ) 
  (h1 : outbound_time = 6)
  (h2 : outbound_speed = 60)
  (h3 : return_time = 5)
  (h4 : outbound_time * outbound_speed = return_time * (outbound_speed + speed_diff)) :
  speed_diff = 12 := by
  sorry

#check speed_difference_calc

end speed_difference_calc_l2990_299071


namespace sqrt_x_plus_reciprocal_l2990_299051

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 150) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 152 := by
  sorry

end sqrt_x_plus_reciprocal_l2990_299051


namespace least_integer_greater_than_sqrt_750_l2990_299026

theorem least_integer_greater_than_sqrt_750 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 750 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 750 → m ≥ n :=
sorry

end least_integer_greater_than_sqrt_750_l2990_299026


namespace fraction_evaluation_l2990_299053

theorem fraction_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/5) = 5/8 := by sorry

end fraction_evaluation_l2990_299053


namespace smallest_valid_m_l2990_299046

def is_valid_partition (m : ℕ) (partition : Fin 14 → Set ℕ) : Prop :=
  (∀ i, partition i ⊆ Finset.range (m + 1)) ∧
  (∀ x, x ∈ Finset.range (m + 1) → ∃ i, x ∈ partition i) ∧
  (∀ i j, i ≠ j → partition i ∩ partition j = ∅)

def has_valid_subset (m : ℕ) (partition : Fin 14 → Set ℕ) : Prop :=
  ∃ i : Fin 14, 1 < i.val ∧ i.val < 14 ∧
    ∃ a b : ℕ, a ∈ partition i ∧ b ∈ partition i ∧
      b < a ∧ (a : ℚ) ≤ 4/3 * (b : ℚ)

theorem smallest_valid_m :
  (∀ m < 56, ∃ partition : Fin 14 → Set ℕ,
    is_valid_partition m partition ∧ ¬has_valid_subset m partition) ∧
  (∀ partition : Fin 14 → Set ℕ,
    is_valid_partition 56 partition → has_valid_subset 56 partition) := by
  sorry

end smallest_valid_m_l2990_299046


namespace smallest_positive_angle_l2990_299093

theorem smallest_positive_angle (α : Real) : 
  let P : Real × Real := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (∃ t : Real, t > 0 ∧ P.1 = t * Real.sin α ∧ P.2 = t * Real.cos α) →
  (∀ β : Real, β > 0 ∧ (∃ s : Real, s > 0 ∧ P.1 = s * Real.sin β ∧ P.2 = s * Real.cos β) → α ≤ β) →
  α = 11 * Real.pi / 6 := by
sorry

end smallest_positive_angle_l2990_299093


namespace no_solution_when_x_is_five_l2990_299008

theorem no_solution_when_x_is_five (x : ℝ) (y : ℝ) :
  x = 5 → ¬∃y, 1 / (x + 5) + y = 1 / (x - 5) := by
sorry

end no_solution_when_x_is_five_l2990_299008


namespace vacation_cost_split_vacation_cost_equalization_l2990_299092

theorem vacation_cost_split (X Y : ℝ) (h : X > Y) : 
  (X - Y) / 2 = (X + Y) / 2 - Y := by sorry

theorem vacation_cost_equalization (X Y : ℝ) (h : X > Y) : 
  (X - Y) / 2 > 0 := by sorry

end vacation_cost_split_vacation_cost_equalization_l2990_299092


namespace order_of_abc_l2990_299015

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := 1 / Real.sin 1

theorem order_of_abc : c > a ∧ a > b := by sorry

end order_of_abc_l2990_299015


namespace min_value_expression_min_value_achievable_l2990_299078

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ≥ 7.5 :=
sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) = 7.5 :=
sorry

end min_value_expression_min_value_achievable_l2990_299078


namespace union_intersection_equality_union_subset_l2990_299044

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Theorem for part (1)
theorem union_intersection_equality (a : ℝ) :
  A ∪ B a = A ∩ B a → a = 1 := by sorry

-- Theorem for part (2)
theorem union_subset (a : ℝ) :
  A ∪ B a = A → a ≤ -1 ∨ a = 1 := by sorry

end union_intersection_equality_union_subset_l2990_299044


namespace ratio_of_arithmetic_sequences_l2990_299023

def arithmetic_sequence_sum (a₁ : ℚ) (d : ℚ) (l : ℚ) : ℚ :=
  let n := (l - a₁) / d + 1
  n / 2 * (a₁ + l)

theorem ratio_of_arithmetic_sequences :
  let seq1_sum := arithmetic_sequence_sum 3 3 96
  let seq2_sum := arithmetic_sequence_sum 4 4 64
  seq1_sum / seq2_sum = 99 / 34 := by
sorry

end ratio_of_arithmetic_sequences_l2990_299023


namespace window_side_length_is_five_l2990_299083

/-- Represents the dimensions of a glass pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the window configuration -/
structure Window where
  pane : Pane
  rows : ℕ
  columns : ℕ
  border_width : ℝ

/-- Calculates the side length of a square window -/
def window_side_length (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating that the window's side length is 5 inches -/
theorem window_side_length_is_five (w : Window) 
  (h_square : window_side_length w = w.rows * w.pane.height + (w.rows + 1) * w.border_width)
  (h_rows : w.rows = 2)
  (h_columns : w.columns = 3)
  (h_border : w.border_width = 1) :
  window_side_length w = 5 := by
  sorry

#check window_side_length_is_five

end window_side_length_is_five_l2990_299083


namespace specific_pyramid_volume_l2990_299034

/-- A right pyramid with a square base -/
structure SquarePyramid where
  base_area : ℝ
  total_surface_area : ℝ
  triangular_face_area : ℝ

/-- The volume of a square pyramid -/
def volume (p : SquarePyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∀ (p : SquarePyramid),
  p.total_surface_area = 648 ∧
  p.triangular_face_area = (1/3) * p.base_area ∧
  p.total_surface_area = p.base_area + 4 * p.triangular_face_area →
  volume p = (4232 * Real.sqrt 6) / 9 := by
  sorry

end specific_pyramid_volume_l2990_299034


namespace border_area_l2990_299060

/-- The area of a border around a rectangular picture --/
theorem border_area (picture_height picture_width border_width : ℝ) : 
  picture_height = 12 →
  picture_width = 15 →
  border_width = 3 →
  (picture_height + 2 * border_width) * (picture_width + 2 * border_width) - picture_height * picture_width = 198 := by
  sorry

end border_area_l2990_299060


namespace equilateral_triangle_side_length_l2990_299062

/-- Given a triangle with sides 6, 10, and 11, prove that an equilateral triangle
    with the same perimeter has side length 9. -/
theorem equilateral_triangle_side_length : 
  ∀ (a b c s : ℝ), 
    a = 6 → b = 10 → c = 11 →  -- Given triangle side lengths
    3 * s = a + b + c →        -- Equilateral triangle has same perimeter
    s = 9 :=                   -- Side length of equilateral triangle is 9
by
  sorry


end equilateral_triangle_side_length_l2990_299062


namespace sum_of_x_and_y_l2990_299074

theorem sum_of_x_and_y (x y : ℝ) 
  (eq1 : x^2 + y^2 = 16*x - 10*y + 14) 
  (eq2 : x - y = 6) : 
  x + y = 3 := by
  sorry

end sum_of_x_and_y_l2990_299074


namespace cube_volume_surface_area_l2990_299017

/-- A cube with volume 5x and surface area x has x equal to 5400 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 5*x ∧ 6*s^2 = x) → x = 5400 := by
  sorry

end cube_volume_surface_area_l2990_299017


namespace sqrt_product_equals_24_l2990_299097

theorem sqrt_product_equals_24 (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (6 * x) * Real.sqrt (24 * x) = 24) : 
  x = Real.sqrt (3 / 22) := by
sorry

end sqrt_product_equals_24_l2990_299097


namespace f_decreasing_on_interval_l2990_299056

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, StrictMonoOn f (Set.Ioo 0 2) := by
  sorry

end f_decreasing_on_interval_l2990_299056


namespace total_quarters_l2990_299037

def initial_quarters : ℕ := 8
def additional_quarters : ℕ := 3

theorem total_quarters : initial_quarters + additional_quarters = 11 := by
  sorry

end total_quarters_l2990_299037


namespace greatest_integer_fraction_inequality_l2990_299021

theorem greatest_integer_fraction_inequality : 
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 := by sorry

end greatest_integer_fraction_inequality_l2990_299021


namespace tangent_point_coordinates_l2990_299049

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  h : y = x^2 - 2*x - 3

/-- Predicate for a circle being tangent to x-axis or y-axis -/
def is_tangent_to_axis (p : ParabolaPoint) : Prop :=
  (p.y = 2 ∨ p.y = -2) ∨ (p.x = 2 ∨ p.x = -2)

/-- The set of points where the circle is tangent to an axis -/
def tangent_points : Set ParabolaPoint :=
  { p | is_tangent_to_axis p }

/-- Theorem stating the coordinates of tangent points -/
theorem tangent_point_coordinates :
  ∀ p ∈ tangent_points,
    (p.x = 1 + Real.sqrt 6 ∧ p.y = 2) ∨
    (p.x = 1 - Real.sqrt 6 ∧ p.y = 2) ∨
    (p.x = 1 + Real.sqrt 2 ∧ p.y = -2) ∨
    (p.x = 1 - Real.sqrt 2 ∧ p.y = -2) ∨
    (p.x = 2 ∧ p.y = -3) ∨
    (p.x = -2 ∧ p.y = 5) :=
  sorry

end tangent_point_coordinates_l2990_299049


namespace largest_p_value_l2990_299073

theorem largest_p_value (m n p : ℕ) : 
  m ≤ n → n ≤ p → 
  2 * m * n * p = (m + 2) * (n + 2) * (p + 2) → 
  p ≤ 130 :=
by sorry

end largest_p_value_l2990_299073


namespace dot_path_length_on_rolling_cube_l2990_299031

/-- The path length traced by a dot on a rolling cube. -/
theorem dot_path_length_on_rolling_cube : 
  ∀ (cube_edge_length : ℝ) (dot_distance_from_edge : ℝ),
    cube_edge_length = 2 →
    dot_distance_from_edge = 1 →
    ∃ (path_length : ℝ),
      path_length = 2 * Real.pi * Real.sqrt 2 :=
by sorry

end dot_path_length_on_rolling_cube_l2990_299031


namespace square_sum_given_conditions_l2990_299084

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x^2 + y^2 = 4) 
  (h2 : (x - y)^2 = 5) : 
  (x + y)^2 = 3 := by
  sorry

end square_sum_given_conditions_l2990_299084


namespace money_and_costs_problem_l2990_299090

/-- The problem of determining the original amounts of money and costs of wine and wheat. -/
theorem money_and_costs_problem 
  (x y z u : ℚ) -- x: A's money, y: B's money, z: cost of 1 hl wine, u: cost of 1 hl wheat
  (h1 : y / 4 = 6 * z)
  (h2 : x / 5 = 8 * z)
  (h3 : (x + 46) + (y - 46) / 3 = 30 * u)
  (h4 : (y - 46) + (x + 46) / 3 = 36 * u)
  : x = 520 ∧ y = 312 ∧ z = 13 ∧ u = 50/3 := by
  sorry

end money_and_costs_problem_l2990_299090


namespace inscribed_trapezoid_median_l2990_299025

-- Define the trapezoid and its properties
structure InscribedTrapezoid where
  radius : ℝ
  baseAngle : ℝ
  leg : ℝ

-- Define the median (midsegment) of the trapezoid
def median (t : InscribedTrapezoid) : ℝ :=
  sorry

-- Theorem statement
theorem inscribed_trapezoid_median
  (t : InscribedTrapezoid)
  (h1 : t.radius = 13)
  (h2 : t.baseAngle = 30 * π / 180)  -- Convert degrees to radians
  (h3 : t.leg = 10) :
  median t = 12 :=
sorry

end inscribed_trapezoid_median_l2990_299025


namespace count_triangles_including_center_l2990_299029

/-- Given a regular polygon with 2n + 1 sides, this function calculates the number of triangles
    formed by its vertices that include the center of the polygon. -/
def trianglesIncludingCenter (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

/-- Theorem stating that the number of triangles including the center of a regular polygon
    with 2n + 1 sides is equal to n(n+1)(2n+1)/6 -/
theorem count_triangles_including_center (n : ℕ) :
  trianglesIncludingCenter n = n * (n + 1) * (2 * n + 1) / 6 := by
  sorry

end count_triangles_including_center_l2990_299029


namespace vacation_class_ratio_l2990_299014

theorem vacation_class_ratio :
  ∀ (grant_vacations : ℕ) (kelvin_classes : ℕ),
    kelvin_classes = 90 →
    grant_vacations + kelvin_classes = 450 →
    (grant_vacations : ℚ) / kelvin_classes = 4 / 1 :=
by
  sorry

end vacation_class_ratio_l2990_299014


namespace largest_t_value_l2990_299045

theorem largest_t_value : ∃ (t_max : ℝ), 
  (∀ t : ℝ, (15 * t^2 - 40 * t + 18) / (4 * t - 3) + 3 * t = 4 * t + 2 → t ≤ t_max) ∧
  ((15 * t_max^2 - 40 * t_max + 18) / (4 * t_max - 3) + 3 * t_max = 4 * t_max + 2) ∧
  t_max = 3 :=
by sorry

end largest_t_value_l2990_299045


namespace no_integer_solution_for_z_l2990_299047

theorem no_integer_solution_for_z :
  ¬ ∃ (z : ℤ), (2 : ℚ) / z = 2 / (z + 1) + 2 / (z + 25) :=
by sorry

end no_integer_solution_for_z_l2990_299047


namespace quadratic_inequality_range_l2990_299035

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 0 → x^2 - a*x + a + 3 ≥ 0) → 
  a ≥ -2 :=
by sorry

end quadratic_inequality_range_l2990_299035


namespace equal_roots_quadratic_l2990_299079

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x, x^2 - (p + 1) * x + p = 0 → (∃! x, x^2 - (p + 1) * x + p = 0)) :=
by sorry

end equal_roots_quadratic_l2990_299079


namespace overall_gain_percentage_l2990_299024

def item_a_cost : ℚ := 700
def item_b_cost : ℚ := 500
def item_c_cost : ℚ := 300
def item_a_gain : ℚ := 70
def item_b_gain : ℚ := 50
def item_c_gain : ℚ := 30

def total_cost : ℚ := item_a_cost + item_b_cost + item_c_cost
def total_gain : ℚ := item_a_gain + item_b_gain + item_c_gain

theorem overall_gain_percentage :
  (total_gain / total_cost) * 100 = 10 := by sorry

end overall_gain_percentage_l2990_299024


namespace sum_of_squared_ratios_bound_l2990_299043

/-- Given positive real numbers a, b, and c, 
    the sum of three terms in the form (2x+y+z)²/(2x²+(y+z)²) 
    where x, y, z are cyclic permutations of a, b, c, is less than or equal to 8 -/
theorem sum_of_squared_ratios_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) + 
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) + 
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end sum_of_squared_ratios_bound_l2990_299043


namespace relationship_abc_l2990_299081

theorem relationship_abc : 3^(1/10) > (1/2)^(1/10) ∧ (1/2)^(1/10) > (-1/2)^3 := by
  sorry

end relationship_abc_l2990_299081


namespace jaysons_mom_age_at_birth_l2990_299039

theorem jaysons_mom_age_at_birth (jayson_age : ℕ) (dad_age : ℕ) (mom_age : ℕ) : 
  jayson_age = 10 →
  dad_age = 4 * jayson_age →
  mom_age = dad_age - 2 →
  mom_age - jayson_age = 28 := by
  sorry

end jaysons_mom_age_at_birth_l2990_299039


namespace parallel_implies_m_half_perpendicular_implies_m_seven_fourths_l2990_299096

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define vector operations
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def AB : ℝ × ℝ := vector_sub OB OA
def BC (m : ℝ) : ℝ × ℝ := vector_sub (OC m) OB
def AC (m : ℝ) : ℝ × ℝ := vector_sub (OC m) OA

-- Define parallel and perpendicular conditions
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the theorems
theorem parallel_implies_m_half :
  parallel AB (BC (1/2)) :=
sorry

theorem perpendicular_implies_m_seven_fourths :
  perpendicular AB (AC (7/4)) :=
sorry

end parallel_implies_m_half_perpendicular_implies_m_seven_fourths_l2990_299096


namespace josh_cheese_purchase_cost_l2990_299066

/-- Calculates the total cost of string cheese purchase including tax -/
def total_cost_with_tax (packs : ℕ) (pieces_per_pack : ℕ) (cost_per_piece : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := packs * pieces_per_pack * cost_per_piece
  let tax := total_cost * tax_rate
  total_cost + tax

/-- The total cost of Josh's string cheese purchase including tax is $6.72 -/
theorem josh_cheese_purchase_cost :
  total_cost_with_tax 3 20 (10 / 100) (12 / 100) = 672 / 100 := by
  sorry

#eval total_cost_with_tax 3 20 (10 / 100) (12 / 100)

end josh_cheese_purchase_cost_l2990_299066


namespace blind_box_equations_l2990_299000

/-- Represents the blind box production scenario -/
structure BlindBoxProduction where
  total_fabric : ℝ
  fabric_for_a : ℝ
  fabric_for_b : ℝ

/-- Conditions for the blind box production -/
def valid_production (p : BlindBoxProduction) : Prop :=
  p.total_fabric = 135 ∧
  p.fabric_for_a + p.fabric_for_b = p.total_fabric ∧
  2 * p.fabric_for_a = 3 * p.fabric_for_b

/-- Theorem stating the correct system of equations for the blind box production -/
theorem blind_box_equations (p : BlindBoxProduction) :
  valid_production p →
  p.fabric_for_a + p.fabric_for_b = 135 ∧ 2 * p.fabric_for_a = 3 * p.fabric_for_b := by
  sorry

end blind_box_equations_l2990_299000


namespace june_sales_increase_l2990_299085

def normal_monthly_sales : ℕ := 21122
def june_july_combined_sales : ℕ := 46166

theorem june_sales_increase : 
  (june_july_combined_sales - 2 * normal_monthly_sales) = 3922 := by
  sorry

end june_sales_increase_l2990_299085


namespace largest_value_in_interval_l2990_299058

theorem largest_value_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) :
  max (max (max (max x (x^3)) (3*x)) (x^(1/3))) (1/x) = 1/x := by sorry

end largest_value_in_interval_l2990_299058


namespace triangle_area_at_most_half_parallelogram_l2990_299082

-- Define a parallelogram
structure Parallelogram where
  area : ℝ
  area_pos : area > 0

-- Define a triangle inscribed in the parallelogram
structure InscribedTriangle (p : Parallelogram) where
  area : ℝ
  area_pos : area > 0
  inscribed : True  -- This represents that the triangle is inscribed in the parallelogram

-- Theorem statement
theorem triangle_area_at_most_half_parallelogram (p : Parallelogram) (t : InscribedTriangle p) :
  t.area ≤ p.area / 2 := by
  sorry

end triangle_area_at_most_half_parallelogram_l2990_299082


namespace max_abs_sum_on_circle_l2990_299042

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 :=
sorry

end max_abs_sum_on_circle_l2990_299042


namespace train_length_l2990_299038

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 16 → speed_kmh * (5/18) * time_s = 240 := by
  sorry

#check train_length

end train_length_l2990_299038


namespace tournament_team_b_matches_l2990_299004

/-- Represents a team in the tournament -/
structure Team where
  city : Fin 16
  type : Bool -- false for A, true for B

/-- The tournament setup -/
structure Tournament where
  teams : Fin 32 → Team
  matches_played : Fin 32 → Nat
  different_matches : ∀ i j, i ≠ j → matches_played i ≠ matches_played j ∨ (teams i).city = 0 ∧ (teams i).type = false
  no_self_city_matches : ∀ i j, (teams i).city = (teams j).city → matches_played i + matches_played j ≤ 30
  max_one_match : ∀ i, matches_played i ≤ 30

theorem tournament_team_b_matches (t : Tournament) : 
  ∃ i, (t.teams i).city = 0 ∧ (t.teams i).type = true ∧ t.matches_played i = 15 :=
sorry

end tournament_team_b_matches_l2990_299004


namespace barbed_wire_cost_l2990_299022

theorem barbed_wire_cost (area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (cost_per_meter : ℝ) : area = 3136 ∧ gate_width = 1 ∧ num_gates = 2 ∧ cost_per_meter = 1 → (4 * Real.sqrt area - num_gates * gate_width) * cost_per_meter = 222 := by
  sorry

end barbed_wire_cost_l2990_299022


namespace fedya_statement_possible_l2990_299048

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents Fedya's age on a given date -/
def age (birthdate : Date) (currentDate : Date) : ℕ := sorry

/-- Returns the date one year after the given date -/
def nextYear (d : Date) : Date := sorry

/-- Returns the date two days before the given date -/
def twoDaysAgo (d : Date) : Date := sorry

/-- Theorem stating that Fedya's statement could be true -/
theorem fedya_statement_possible : ∃ (birthdate currentDate : Date),
  age birthdate (twoDaysAgo currentDate) = 10 ∧
  age birthdate (nextYear currentDate) = 13 :=
sorry

end fedya_statement_possible_l2990_299048


namespace equation_solutions_equation_solutions_unique_l2990_299041

theorem equation_solutions :
  (∃ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3)) ∧
  (∃ x : ℝ, x^2 - 4*x + 2 = 0) :=
by
  constructor
  · use -3/2
    sorry
  · use 2 + Real.sqrt 2
    sorry

theorem equation_solutions_unique :
  (∀ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3) ↔ (x = -3/2 ∨ x = 1/2)) ∧
  (∀ x : ℝ, x^2 - 4*x + 2 = 0 ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2)) :=
by
  constructor
  · intro x
    sorry
  · intro x
    sorry

end equation_solutions_equation_solutions_unique_l2990_299041


namespace probability_one_white_one_black_is_eight_fifteenths_l2990_299080

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7

def probability_one_white_one_black : ℚ := (white_balls * black_balls) / (total_balls.choose 2)

theorem probability_one_white_one_black_is_eight_fifteenths :
  probability_one_white_one_black = 8 / 15 := by
  sorry

end probability_one_white_one_black_is_eight_fifteenths_l2990_299080


namespace hotel_room_charge_comparison_l2990_299052

theorem hotel_room_charge_comparison (G P R : ℝ) 
  (hP_G : P = G * 0.8)
  (hR_G : R = G * 1.6) :
  (R - P) / R * 100 = 50 := by
  sorry

end hotel_room_charge_comparison_l2990_299052


namespace family_ages_correct_l2990_299011

-- Define the family members' ages as natural numbers
def son_age : Nat := 7
def daughter_age : Nat := 12
def man_age : Nat := 27
def wife_age : Nat := 22
def father_age : Nat := 59

-- State the theorem
theorem family_ages_correct :
  -- Man is 20 years older than son
  man_age = son_age + 20 ∧
  -- Man is 15 years older than daughter
  man_age = daughter_age + 15 ∧
  -- In two years, man's age will be twice son's age
  man_age + 2 = 2 * (son_age + 2) ∧
  -- In two years, man's age will be three times daughter's age
  man_age + 2 = 3 * (daughter_age + 2) ∧
  -- Wife is 5 years younger than man
  wife_age = man_age - 5 ∧
  -- In 6 years, wife will be twice as old as daughter
  wife_age + 6 = 2 * (daughter_age + 6) ∧
  -- Father is 32 years older than man
  father_age = man_age + 32 := by
  sorry


end family_ages_correct_l2990_299011


namespace reflection_after_translation_l2990_299061

/-- Given a point A with coordinates (-3, -2), prove that translating it 5 units
    to the right and then reflecting across the y-axis results in a point with
    coordinates (-2, -2). -/
theorem reflection_after_translation :
  let A : ℝ × ℝ := (-3, -2)
  let B : ℝ × ℝ := (A.1 + 5, A.2)
  let B' : ℝ × ℝ := (-B.1, B.2)
  B' = (-2, -2) := by
sorry

end reflection_after_translation_l2990_299061


namespace work_division_proof_l2990_299055

/-- The number of days it takes x to finish the entire work -/
def x_total_days : ℝ := 18

/-- The number of days it takes y to finish the entire work -/
def y_total_days : ℝ := 15

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining_days : ℝ := 12

/-- The number of days y worked before leaving the job -/
def y_worked_days : ℝ := 5

theorem work_division_proof :
  let total_work : ℝ := 1
  let x_rate : ℝ := total_work / x_total_days
  let y_rate : ℝ := total_work / y_total_days
  y_worked_days * y_rate + x_remaining_days * x_rate = total_work :=
by sorry

end work_division_proof_l2990_299055


namespace at_least_one_not_less_than_one_l2990_299050

theorem at_least_one_not_less_than_one (x : ℝ) :
  let a := |Real.cos x - Real.sin x|
  let b := |Real.cos x + Real.sin x|
  max a b ≥ 1 := by
sorry

end at_least_one_not_less_than_one_l2990_299050


namespace residue_problem_l2990_299091

theorem residue_problem : (198 * 6 - 16 * 8^2 + 5) % 16 = 9 := by
  sorry

end residue_problem_l2990_299091


namespace cos_sqrt3_over_2_necessary_not_sufficient_l2990_299059

theorem cos_sqrt3_over_2_necessary_not_sufficient (α : ℝ) :
  (∃ k : ℤ, α = 2 * k * π + 5 * π / 6 → Real.cos α = -Real.sqrt 3 / 2) ∧
  (∃ α : ℝ, Real.cos α = -Real.sqrt 3 / 2 ∧ ∀ k : ℤ, α ≠ 2 * k * π + 5 * π / 6) :=
by sorry

end cos_sqrt3_over_2_necessary_not_sufficient_l2990_299059


namespace carpet_shaded_area_l2990_299069

/-- Calculates the total shaded area of a carpet design with given ratios and square counts. -/
theorem carpet_shaded_area (carpet_side : ℝ) (ratio_12_S : ℝ) (ratio_S_T : ℝ) (ratio_T_U : ℝ)
  (count_S : ℕ) (count_T : ℕ) (count_U : ℕ) :
  carpet_side = 12 →
  ratio_12_S = 4 →
  ratio_S_T = 2 →
  ratio_T_U = 2 →
  count_S = 1 →
  count_T = 4 →
  count_U = 8 →
  let S := carpet_side / ratio_12_S
  let T := S / ratio_S_T
  let U := T / ratio_T_U
  count_S * S^2 + count_T * T^2 + count_U * U^2 = 22.5 := by
  sorry

end carpet_shaded_area_l2990_299069


namespace money_sharing_l2990_299010

theorem money_sharing (jane_share : ℕ) (total : ℕ) : 
  jane_share = 30 →
  (2 : ℕ) * total = jane_share * (2 + 3 + 8) →
  total = 195 := by
sorry

end money_sharing_l2990_299010


namespace sum_equals_rounded_sum_l2990_299018

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  let remainder := x % 5
  if remainder < 3 then x - remainder else x + (5 - remainder)

def sum_rounded_to_five (n : ℕ) : ℕ :=
  List.range n |> List.map (λ x => round_to_nearest_five (x + 1)) |> List.sum

theorem sum_equals_rounded_sum (n : ℕ) : sum_to_n n = sum_rounded_to_five n := by
  sorry

end sum_equals_rounded_sum_l2990_299018


namespace circle_center_sum_l2990_299019

theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = 10) →
  h + k = 7 := by
sorry

end circle_center_sum_l2990_299019


namespace no_roots_equation_l2990_299063

theorem no_roots_equation : ¬∃ (x : ℝ), x - 8 / (x - 4) = 4 - 8 / (x - 4) := by sorry

end no_roots_equation_l2990_299063


namespace sqrt_6_bounds_l2990_299094

theorem sqrt_6_bounds : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end sqrt_6_bounds_l2990_299094


namespace average_temperature_l2990_299054

def temperatures : List ℝ := [60, 59, 56, 53, 49, 48, 46]

theorem average_temperature : 
  (List.sum temperatures) / temperatures.length = 53 :=
by sorry

end average_temperature_l2990_299054


namespace max_min_product_l2990_299003

theorem max_min_product (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b + c = 12 →
  a * b + b * c + c * a = 20 →
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧
             m ≤ 12 ∧
             ∀ (k : ℝ), (∃ (x y z : ℝ), 
               0 < x ∧ 0 < y ∧ 0 < z ∧
               x + y + z = 12 ∧
               x * y + y * z + z * x = 20 ∧
               k = min (x * y) (min (y * z) (z * x))) →
             k ≤ 12 :=
by sorry

end max_min_product_l2990_299003


namespace max_trailing_zeros_1003_sum_l2990_299012

/-- Given three natural numbers that sum to 1003, the maximum number of trailing zeros in their product is 7. -/
theorem max_trailing_zeros_1003_sum (a b c : ℕ) (h_sum : a + b + c = 1003) :
  ∃ (n : ℕ), n ≤ 7 ∧
  ∀ (m : ℕ), (a * b * c) % (10^m) = 0 → m ≤ n :=
by sorry

end max_trailing_zeros_1003_sum_l2990_299012


namespace loan_B_is_5000_l2990_299036

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  rate : ℚ  -- Interest rate per annum
  time_B : ℕ  -- Time for loan B in years
  time_C : ℕ  -- Time for loan C in years
  amount_C : ℚ  -- Amount lent to C
  total_interest : ℚ  -- Total interest received from both loans

/-- Calculates the amount lent to B given the loan details --/
def calculate_loan_B (loan : LoanDetails) : ℚ :=
  (loan.total_interest - loan.amount_C * loan.rate * loan.time_C) / (loan.rate * loan.time_B)

/-- Theorem stating that the amount lent to B is 5000 --/
theorem loan_B_is_5000 (loan : LoanDetails) 
  (h1 : loan.rate = 9 / 100)
  (h2 : loan.time_B = 2)
  (h3 : loan.time_C = 4)
  (h4 : loan.amount_C = 3000)
  (h5 : loan.total_interest = 1980) :
  calculate_loan_B loan = 5000 := by
  sorry

#eval calculate_loan_B { rate := 9/100, time_B := 2, time_C := 4, amount_C := 3000, total_interest := 1980 }

end loan_B_is_5000_l2990_299036


namespace resulting_figure_sides_l2990_299040

/-- Represents a polygon in the construction --/
structure Polygon :=
  (sides : ℕ)
  (adjacentSides : ℕ)

/-- The construction of polygons --/
def construction : List Polygon :=
  [{ sides := 3, adjacentSides := 1 },  -- isosceles triangle
   { sides := 4, adjacentSides := 2 },  -- rectangle
   { sides := 6, adjacentSides := 2 },  -- first hexagon
   { sides := 7, adjacentSides := 2 },  -- heptagon
   { sides := 6, adjacentSides := 2 },  -- second hexagon
   { sides := 9, adjacentSides := 1 }]  -- nonagon

theorem resulting_figure_sides :
  (construction.map (λ p => p.sides - p.adjacentSides)).sum = 25 := by
  sorry

end resulting_figure_sides_l2990_299040


namespace tangent_line_slope_l2990_299089

theorem tangent_line_slope (k : ℝ) : 
  (∃ x : ℝ, k * x = x^3 - x^2 + x ∧ 
   k = 3 * x^2 - 2 * x + 1) → 
  k = 1 ∨ k = 3/4 := by
sorry

end tangent_line_slope_l2990_299089


namespace total_workers_count_l2990_299098

def num_other_workers : ℕ := 5

def probability_jack_and_jill : ℚ := 1 / 21

theorem total_workers_count (num_selected : ℕ) (h1 : num_selected = 2) :
  ∃ (total_workers : ℕ),
    total_workers = num_other_workers + 2 ∧
    probability_jack_and_jill = 1 / (total_workers.choose num_selected) :=
by sorry

end total_workers_count_l2990_299098


namespace series_sum_equals_three_halves_l2990_299064

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-3)/(3^n) is equal to 3/2 -/
theorem series_sum_equals_three_halves :
  ∑' n, (4 * n - 3 : ℝ) / (3 : ℝ) ^ n = 3 / 2 := by
  sorry

end series_sum_equals_three_halves_l2990_299064


namespace shoe_probabilities_l2990_299032

-- Define the type for shoes
inductive Shoe
| left : Shoe
| right : Shoe

-- Define a pair of shoes
structure ShoePair :=
  (left : Shoe)
  (right : Shoe)

-- Define the cabinet with 3 pairs of shoes
def cabinet : Finset ShoePair := sorry

-- Define the sample space of choosing 2 shoes
def sampleSpace : Finset (Shoe × Shoe) := sorry

-- Event A: The taken out shoes do not form a pair
def eventA : Finset (Shoe × Shoe) := sorry

-- Event B: Both taken out shoes are for the same foot
def eventB : Finset (Shoe × Shoe) := sorry

-- Event C: One shoe is for the left foot and the other is for the right foot, but they do not form a pair
def eventC : Finset (Shoe × Shoe) := sorry

theorem shoe_probabilities :
  (Finset.card eventA : ℚ) / Finset.card sampleSpace = 4 / 5 ∧
  (Finset.card eventB : ℚ) / Finset.card sampleSpace = 2 / 5 ∧
  (Finset.card eventC : ℚ) / Finset.card sampleSpace = 2 / 5 :=
sorry

end shoe_probabilities_l2990_299032


namespace product_of_specific_primes_l2990_299095

theorem product_of_specific_primes : 
  let largest_one_digit_prime := 7
  let smallest_two_digit_prime1 := 11
  let smallest_two_digit_prime2 := 13
  largest_one_digit_prime * smallest_two_digit_prime1 * smallest_two_digit_prime2 = 1001 := by
sorry

end product_of_specific_primes_l2990_299095


namespace sixth_grade_boys_l2990_299067

theorem sixth_grade_boys (total_students : ℕ) (boys : ℕ) : 
  total_students = 152 →
  boys * 10 = (total_students - boys - 5) * 11 →
  boys = 77 := by
sorry

end sixth_grade_boys_l2990_299067


namespace gcd_and_sum_divisibility_l2990_299070

theorem gcd_and_sum_divisibility : 
  (Nat.gcd 42558 29791 = 3) ∧ 
  ¬(72349 % 3 = 0) := by sorry

end gcd_and_sum_divisibility_l2990_299070


namespace min_reciprocal_sum_min_reciprocal_sum_achievable_l2990_299028

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_reciprocal_sum_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end min_reciprocal_sum_min_reciprocal_sum_achievable_l2990_299028


namespace y_intercept_of_specific_line_l2990_299033

/-- A line in the xy-plane is defined by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- 
Given a line with slope 4 passing through the point (199, 800),
prove that its y-intercept is 4.
-/
theorem y_intercept_of_specific_line :
  let l : Line := { slope := 4, point := (199, 800) }
  y_intercept l = 4 := by
  sorry

end y_intercept_of_specific_line_l2990_299033


namespace isosceles_trapezoid_area_l2990_299013

/-- An isosceles trapezoid with given base lengths and perpendicular diagonals -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  perpendicular_diagonals : Prop

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The area of an isosceles trapezoid with bases 40 and 24, 
    and mutually perpendicular diagonals, is 1024 -/
theorem isosceles_trapezoid_area : 
  ∀ (t : IsoscelesTrapezoid), 
  t.base1 = 40 ∧ t.base2 = 24 ∧ t.perpendicular_diagonals → 
  area t = 1024 := by
  sorry

end isosceles_trapezoid_area_l2990_299013


namespace associate_prof_charts_l2990_299072

theorem associate_prof_charts (
  associate_profs : ℕ) 
  (assistant_profs : ℕ) 
  (charts_per_associate : ℕ) 
  (h1 : associate_profs + assistant_profs = 7)
  (h2 : 2 * associate_profs + assistant_profs = 10)
  (h3 : charts_per_associate * associate_profs + 2 * assistant_profs = 11)
  : charts_per_associate = 1 := by
  sorry

end associate_prof_charts_l2990_299072
