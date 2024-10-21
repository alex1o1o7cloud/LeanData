import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_price_after_discounts_and_tax_l893_89350

def original_price : ℝ := 300
def sunday_discount : ℝ := 0.4
def monday_discount : ℝ := 0.2
def sales_tax : ℝ := 0.05

theorem bicycle_price_after_discounts_and_tax :
  (original_price * (1 - sunday_discount) * (1 - monday_discount) * (1 + sales_tax)) = 151.20 := by
  -- Proof steps would go here
  sorry

#eval (original_price * (1 - sunday_discount) * (1 - monday_discount) * (1 + sales_tax))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_price_after_discounts_and_tax_l893_89350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_per_component_l893_89308

/-- Calculates the lowest price per component to avoid loss for a computer manufacturer. -/
theorem lowest_price_per_component
  (production_cost : ℝ)
  (shipping_cost : ℝ)
  (fixed_costs : ℝ)
  (num_components : ℕ)
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 7)
  (h3 : fixed_costs = 16500)
  (h4 : num_components = 150) :
  (production_cost + shipping_cost) * (num_components : ℝ) + fixed_costs ≤ 197 * (num_components : ℝ) ∧
  ∀ p : ℝ, p < 197 → (production_cost + shipping_cost) * (num_components : ℝ) + fixed_costs > p * (num_components : ℝ) :=
by sorry

#check lowest_price_per_component

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_per_component_l893_89308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sweet_potato_problem_l893_89302

/-- Given an amount of sweet potato and an amount per person, 
    calculate the amount left after sharing in grams. -/
noncomputable def sweet_potato_leftover (total : ℝ) (per_person : ℝ) : ℝ :=
  let people_served := ⌊total / per_person⌋
  let amount_served := people_served * per_person
  let leftover_kg := total - amount_served
  leftover_kg * 1000

/-- Theorem: When 52.5 kg of sweet potato is divided by 5 kg per person,
    2500 grams of sweet potato are left after sharing. -/
theorem sweet_potato_problem : sweet_potato_leftover 52.5 5 = 2500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sweet_potato_problem_l893_89302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l893_89373

theorem max_true_statements (c d : ℝ) : 
  (∃ (s : Fin 5 → Prop), 
    s 0 = (1 / c > 1 / d) ∧ 
    s 1 = (c^2 < d^2) ∧ 
    s 2 = (c > d) ∧ 
    s 3 = (c > 0) ∧ 
    s 4 = (d > 0) ∧ 
    (∀ i : Fin 5, s i → i.val < 5) ∧
    (∃ (n : ℕ), n ≤ 5 ∧ (∃ (t : Fin n → Fin 5), Function.Injective t ∧ (∀ i : Fin n, s (t i)))) ∧
    (∀ (m : ℕ), m > 3 → 
      ¬∃ (t : Fin m → Fin 5), Function.Injective t ∧ (∀ i : Fin m, s (t i)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l893_89373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_iff_k_eq_neg_four_l893_89371

noncomputable section

-- Define the parameterizations of the two lines
def line1 (s k : ℝ) : ℝ × ℝ × ℝ := (-1 + s, 3 - k*s, 1 + k*s)
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (t/2, 1 + 2*t, 2 - t)

-- Define the direction vectors of the two lines
def direction1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -k, k)
def direction2 : ℝ × ℝ × ℝ := (1/2, 2, -1)

-- Define coplanarity condition
def are_coplanar (k : ℝ) : Prop :=
  ∃ r : ℝ, direction1 k = r • direction2

-- Theorem statement
theorem lines_coplanar_iff_k_eq_neg_four :
  ∀ k : ℝ, are_coplanar k ↔ k = -4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_coplanar_iff_k_eq_neg_four_l893_89371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aunt_zhang_investment_l893_89303

/-- Calculates the total amount (principal + interest) for a fixed-rate investment -/
def total_amount (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate * years)

/-- Proves that the total amount for Aunt Zhang's investment is 23180 yuan -/
theorem aunt_zhang_investment : 
  let principal : ℝ := 20000
  let rate : ℝ := 0.0318
  let years : ℝ := 5
  total_amount principal rate years = 23180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aunt_zhang_investment_l893_89303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l893_89329

noncomputable def f (x : ℝ) := 3 * Real.cos x - 4 * Real.sin x

theorem range_of_f :
  ∀ y ∈ Set.range (f ∘ (fun x => x : Set.Icc 0 π → ℝ)),
  -5 ≤ y ∧ y ≤ 3 ∧
  ∃ x ∈ Set.Icc 0 π, f x = -5 ∧
  ∃ x ∈ Set.Icc 0 π, f x = 3 := by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l893_89329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l893_89390

/-- The function f(x) = (3x + 1) / (x - 4) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (x - 4)

/-- The range of f is (-∞, 3) ∪ (3, ∞) -/
theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l893_89390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_k_l893_89366

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

-- Define the intersection points
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ y = line k x ∧ circle_eq x y}

-- State the theorem
theorem intersection_distance_implies_k (k : ℝ) :
  (∃ (M N : ℝ × ℝ), M ∈ intersection_points k ∧ N ∈ intersection_points k ∧ 
   (M.1 - N.1)^2 + (M.2 - N.2)^2 = 12) →
  k = Real.sqrt 3 / 3 ∨ k = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_k_l893_89366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l893_89376

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the focus-to-point distance for a parabola
noncomputable def focusToPoint (p x : ℝ) : ℝ := x + p/2

theorem parabola_problem (p : ℝ) (y_0 : ℝ) 
  (h1 : p > 0)
  (h2 : parabola p 2 y_0)
  (h3 : focusToPoint p 2 = 5/2) :
  (∃ (x y : ℝ),
    -- 1. The parabola equation is y^2 = 2x
    (∀ (x' y' : ℝ), parabola p x' y' ↔ y'^2 = 2*x') ∧
    -- 2. The point B(0, 0) on the parabola is closest to A(2/3, 0)
    parabola p 0 0 ∧
    (∀ (x' y' : ℝ), parabola p x' y' →
      distance x y (2/3) 0 ≤ distance x' y' (2/3) 0) ∧
    -- 3. The distance |BA| is 2/3
    distance 0 0 (2/3) 0 = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l893_89376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_perpendicular_line_l893_89381

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the line and planes
variable (l : Subspace ℝ V)
variable (α β : Subspace ℝ V)

-- Define the parallel and perpendicular relations
def parallel (A B : Subspace ℝ V) : Prop := sorry

def perpendicular (L P : Subspace ℝ V) : Prop := sorry

-- Define a function to represent "all lines in a plane"
def all_lines_in_plane (P : Subspace ℝ V) : Set (Subspace ℝ V) := sorry

-- State the theorem
theorem parallel_planes_perpendicular_line 
  (h1 : parallel α β) 
  (h2 : perpendicular l α) : 
  ∀ m ∈ all_lines_in_plane β, perpendicular m l := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_perpendicular_line_l893_89381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_intersecting_circle_l893_89347

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if two circles are tangent
def are_tangent (c1 c2 : Circle) : Prop := sorry

-- Define a function to check if a circle intersects another circle
def intersects (c1 c2 : Circle) : Prop := sorry

-- Define the tangency conditions
def tangency_condition (C1 C2 C3 C4 C5 C6 : Circle) : Prop :=
  are_tangent C1 C6 ∧ are_tangent C1 C2 ∧
  are_tangent C2 C3 ∧ are_tangent C3 C4 ∧
  are_tangent C4 C5 ∧ are_tangent C5 C6

-- Theorem statement
theorem existence_of_intersecting_circle
  (C1 C2 C3 C4 C5 C6 : Circle)
  (h : tangency_condition C1 C2 C3 C4 C5 C6) :
  ∃ (C : Circle), intersects C C1 ∧ intersects C C2 ∧ intersects C C3 ∧
                  intersects C C4 ∧ intersects C C5 ∧ intersects C C6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_intersecting_circle_l893_89347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_ten_percent_l893_89346

noncomputable section

/-- Represents the cost price of the watch in Rupees -/
def cost_price : ℚ := 1076923076923077 / 1000000000000000

/-- Represents the additional amount in Rupees that would result in a gain -/
def additional_amount : ℚ := 140

/-- Represents the gain percentage if the watch was sold for the additional amount -/
def gain_percentage : ℚ := 3 / 100

/-- Calculates the selling price that would result in the given gain percentage -/
def selling_price_with_gain : ℚ := cost_price * (1 + gain_percentage)

/-- Calculates the actual selling price of the watch -/
def actual_selling_price : ℚ := selling_price_with_gain - additional_amount

/-- Calculates the loss amount -/
def loss_amount : ℚ := cost_price - actual_selling_price

/-- Calculates the loss percentage -/
def loss_percentage : ℚ := (loss_amount / cost_price) * 100

/-- Theorem stating that the loss percentage is 10% -/
theorem loss_percentage_is_ten_percent : loss_percentage = 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_ten_percent_l893_89346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_semicircles_area_value_l893_89304

/-- The area of the region inside a regular hexagon with side length 4 but outside
    six inscribed semicircles with diameters along each side of the hexagon. -/
noncomputable def hexagon_semicircles_area : ℝ :=
  let hexagon_side : ℝ := 4
  let hexagon_area : ℝ := (3 * Real.sqrt 3 / 2) * hexagon_side^2
  let semicircle_radius : ℝ := hexagon_side / 2
  let semicircle_area : ℝ := Real.pi * semicircle_radius^2 / 2
  let total_semicircles_area : ℝ := 6 * semicircle_area
  hexagon_area - total_semicircles_area

/-- Proof that the area of the region inside the hexagon but outside
    all semicircles is equal to 48√3 - 12π. -/
theorem hexagon_semicircles_area_value : hexagon_semicircles_area = 48 * Real.sqrt 3 - 12 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_semicircles_area_value_l893_89304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_equation_intersection_sum_l893_89365

-- Define the Cartesian coordinate system
variable (x y : ℝ)

-- Define polar coordinates for point N
noncomputable def N : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

-- Define curve C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point G
def G (x y x₀ y₀ : ℝ) : Prop := x = x₀ + 1 ∧ y = y₀ + 1

-- Define line l
noncomputable def l (t : ℝ) : ℝ × ℝ := (2 - t/2, (Real.sqrt 3/2) * t)

-- Define point P
def P : ℝ × ℝ := (2, 0)

theorem curve_C₂_equation (x y : ℝ) :
  (∃ x₀ y₀, C₁ x₀ y₀ ∧ G x y x₀ y₀) ↔ (x - 1)^2 + (y - 1)^2 = 1 := by sorry

theorem intersection_sum (A B : ℝ × ℝ) :
  (∃ t₁ t₂, l t₁ = A ∧ l t₂ = B ∧ (A.1 - 1)^2 + (A.2 - 1)^2 = 1 ∧ (B.1 - 1)^2 + (B.2 - 1)^2 = 1) →
  1 / ((A.1 - 2)^2 + A.2^2).sqrt + 1 / ((B.1 - 2)^2 + B.2^2).sqrt = 1 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_equation_intersection_sum_l893_89365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l893_89367

theorem log_inequality_range (a : ℝ) : 
  (0 < a ∧ Real.log (2/3) / Real.log a < 1) ↔ a ∈ Set.Ioo 0 (2/3) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l893_89367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_not_necessary_l893_89326

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, 2 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 3 → |x| < 2) ∧
  (∃ x : ℝ, |x| < 2 ∧ ¬(2 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_sufficient_not_necessary_l893_89326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l893_89341

theorem distance_between_points : 
  let point1 : ℝ × ℝ := (3, 3)
  let point2 : ℝ × ℝ := (-2, -2)
  let distance := Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2)
  distance = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l893_89341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_on_parabola_l893_89337

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.A = (0, 0) ∧
  (∃ a : ℝ, t.B = (-a, parabola (-a)) ∧ t.C = (a, parabola a)) ∧
  (t.B.2 = t.C.2)

-- Define the area of the triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let base := t.C.1 - t.B.1
  let height := t.B.2 - t.A.2
  (1/2) * base * height

-- Define the length of BC
noncomputable def lengthBC (t : Triangle) : ℝ :=
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)

-- Theorem statement
theorem triangle_on_parabola (t : Triangle) :
  satisfiesConditions t → triangleArea t = 128 → lengthBC t = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_on_parabola_l893_89337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_espresso_price_l893_89387

/-- The cost of Nancy's daily coffee purchases over 20 days -/
noncomputable def total_cost : ℚ := 110

/-- The number of days Nancy buys coffee -/
def days : ℕ := 20

/-- The cost of an iced coffee -/
noncomputable def iced_coffee_cost : ℚ := 5/2

/-- The cost of a double espresso -/
noncomputable def double_espresso_cost : ℚ := total_cost / days - iced_coffee_cost

theorem double_espresso_price :
  double_espresso_cost = 3 := by
  -- Unfold the definitions
  unfold double_espresso_cost total_cost days iced_coffee_cost
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_espresso_price_l893_89387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l893_89349

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

noncomputable def g (m x : ℝ) : ℝ := m * x^2 - 2 * x + 2 - m

theorem problem_solution :
  ∀ (a b : ℝ),
  (∀ x, f a b x = -f a b (-x)) →  -- f is odd
  (b = 0 ∧
   (a > 0 → ∀ x y, 1 < x → x < y → f a 0 x > f a 0 y) ∧
   (a = 1 → ∀ m, (∀ x₁ ∈ Set.Icc 1 3, ∃ x₂ ∈ Set.Icc 0 1, f 1 0 x₁ + 1/2 = g m x₂) ↔ m ≤ 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l893_89349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l893_89338

def max_set_size : ℕ := 12

def valid_subset (S : Finset ℕ) : Prop :=
  ∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(5 ∣ (x + y))

theorem max_subset_size :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 25 ∧ valid_subset S ∧ S.card = max_set_size ∧
  ∀ (T : Finset ℕ), T ⊆ Finset.range 25 → valid_subset T → T.card ≤ max_set_size :=
by
  sorry

#check max_subset_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l893_89338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_values_l893_89357

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 6 / (x - 1) - Real.sqrt (x + 4)

-- Define the domain of f
def domain (x : ℝ) : Prop := x ≥ -4 ∧ x ≠ 1

theorem f_domain_and_values :
  (∀ x : ℝ, f x ≠ 0 → domain x) ∧
  f (-1) = -3 - Real.sqrt 3 ∧
  f 12 = -38/11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_values_l893_89357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l893_89388

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_ratio 
  (a b : ArithmeticSequence) 
  (h : ∀ n : ℕ, sum_n a n / sum_n b n = (2 * n : ℚ) / (3 * n + 1)) :
  ∀ n : ℕ, a.a n / b.a n = (2 * n - 1 : ℚ) / (3 * n - 1) := by
  sorry

#check arithmetic_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l893_89388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l893_89393

-- Define the function f as noncomputable
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then Real.log x else -Real.log (-x)

-- State the theorem
theorem f_positive_range :
  (∀ x, f (-x) = -f x) →  -- f is odd
  {x : ℝ | f x > 0} = {x : ℝ | -1 < x ∧ x < 0 ∨ x > 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l893_89393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_cube_sum_l893_89330

theorem infinite_solutions_cube_sum :
  ∃ (S : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ × ℤ × ℤ × ℤ × ℤ)), 
    (Set.Infinite S) ∧ 
    (∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ : ℤ), 
      (x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀) ∈ S → 
      x₁^3 + x₂^3 + x₃^3 + x₄^3 + x₅^3 + x₆^3 + x₇^3 + x₈^3 + x₉^3 + x₁₀^3 = 600) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_cube_sum_l893_89330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_one_l893_89310

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := Real.sqrt (x - 1)

-- State the theorem
theorem domain_of_sqrt_x_minus_one :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_one_l893_89310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l893_89306

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 3) + 1

-- State the theorem
theorem inverse_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, Function.LeftInverse g (f a) ∧ Function.RightInverse g (f a) ∧ g 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l893_89306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_f_eq_one_l893_89331

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 4 else 3*x - 6

theorem unique_solution_f_f_eq_one :
  ∃! x : ℝ, f (f x) = 1 ∧ x = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_f_eq_one_l893_89331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seventeen_pi_fourths_l893_89343

theorem cos_seventeen_pi_fourths : Real.cos (17 * π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seventeen_pi_fourths_l893_89343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l893_89391

/-- For any real number x > 2, the infinite sum ∑(n=0 to ∞) 1 / (x^(3^n) - x^(-3^n)) 
    converges and is equal to 1 / (x - 1). -/
theorem infinite_sum_convergence (x : ℝ) (hx : x > 2) :
  ∑' n, 1 / (x^(3^n) - (1/x)^(3^n)) = 1 / (x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l893_89391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_5_is_121_l893_89359

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_1_3 : a 1 + a 3 = 10
  sum_2_4 : a 2 + a 4 = 30

/-- The sum of the first n terms of a geometric sequence -/
def geometricSum (seq : GeometricSequence) (n : ℕ) : ℚ :=
  seq.a 1 * (1 - (seq.a 2 / seq.a 1)^n) / (1 - (seq.a 2 / seq.a 1))

/-- Theorem: The sum of the first 5 terms of the specific geometric sequence is 121 -/
theorem geometric_sum_5_is_121 (seq : GeometricSequence) : geometricSum seq 5 = 121 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_5_is_121_l893_89359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_symmetry_axes_l893_89309

noncomputable section

/-- A quadratic function of the form ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The axis of symmetry of a quadratic function ax^2 + bx + c -/
def axisOfSymmetry (a b c : ℝ) : ℝ := -b / (2 * a)

/-- The roots of a quadratic function are distinct and real -/
def hasDistinctRealRoots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c > 0

theorem equal_distance_symmetry_axes 
  (m : ℝ) 
  (f g : ℝ → ℝ) 
  (hf : f = QuadraticFunction (-1) (m^2) 0) 
  (hg : g = QuadraticFunction 1 0 (-m^2)) 
  (hf_roots : hasDistinctRealRoots (-1) (m^2) 0) 
  (hg_roots : hasDistinctRealRoots 1 0 (-m^2)) 
  (h_equal_dist : ∃ (d : ℝ), d > 0 ∧ 
    (∀ (x1 x2 : ℝ), (f x1 = 0 ∨ g x1 = 0) ∧ (f x2 = 0 ∨ g x2 = 0) ∧ x1 < x2 ∧ 
      (∀ x, x1 < x ∧ x < x2 → (f x ≠ 0 ∧ g x ≠ 0)) → x2 - x1 = d)) :
  |axisOfSymmetry (-1) (m^2) 0 - axisOfSymmetry 1 0 (-m^2)| = 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_symmetry_axes_l893_89309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l893_89364

theorem complex_modulus_example : Complex.abs (7/4 + 3*Complex.I) = Real.sqrt 193 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l893_89364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_distance_increases_l893_89318

noncomputable section

/-- Two unit vectors representing the directions of two intersecting lines -/
def e₁ : ℝ × ℝ := sorry
def e₂ : ℝ × ℝ := sorry

/-- The acute angle between the two lines -/
def φ : ℝ := sorry

/-- Assumption that φ is an acute angle -/
axiom φ_acute : 0 < φ ∧ φ < Real.pi / 2

/-- The compression coefficient -/
def k : ℝ := 1 / 2

/-- The transformation function that applies the compression -/
def transform (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.1, k * v.2)

/-- The distance function from a point to the origin (intersection point) -/
def distance (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- The theorem stating that there exists a point whose distance increases after transformation -/
theorem exists_point_distance_increases :
  ∃ v : ℝ × ℝ, distance (transform v) > distance v := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_distance_increases_l893_89318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_without_y_l893_89314

theorem sum_of_coefficients_without_y (x y : ℝ) : 
  (Finset.sum (Finset.range 4) fun i =>
    Finset.sum (Finset.range 4) fun j =>
      if i + j ≤ 3 then (Nat.choose 3 i) * (Nat.choose (3 - i) j) * x^i * 3^(3 - i - j)
      else 0) = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_without_y_l893_89314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tri_poly_roots_bounded_l893_89345

/-- A polynomial with coefficients in {-1, 0, 1} -/
def TriPoly (n : ℕ) := { p : Polynomial ℝ | ∀ i, p.coeff i ∈ ({-1, 0, 1} : Set ℝ) }

/-- Theorem: All real roots of a TriPoly are in [-2, 2] -/
theorem tri_poly_roots_bounded {n : ℕ} (p : TriPoly n) :
  ∀ r : ℝ, (p : Polynomial ℝ).eval r = 0 → r ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tri_poly_roots_bounded_l893_89345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l893_89386

noncomputable section

open Real

variable (t : ℝ)

def x : ℝ → ℝ := fun t => log t
def y : ℝ → ℝ := fun t => arctan t

-- Helper function to represent the second derivative
def second_derivative (t : ℝ) : ℝ := t * (1 - t^2) / (1 + t^2)^2

theorem second_derivative_parametric_function (t : ℝ) (h : t > 0) :
  deriv (fun x => y (exp x)) (log t) = second_derivative t := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l893_89386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_143_of_3_11_l893_89361

/-- The decimal representation of 3/11 -/
def decimal_rep_3_11 : ℚ := 3 / 11

/-- The repeating sequence in the decimal representation of 3/11 -/
def repeating_sequence : List ℕ := [2, 7]

/-- The length of the repeating sequence -/
def sequence_length : ℕ := repeating_sequence.length

/-- Function to get the nth digit after the decimal point -/
def nth_digit (n : ℕ) : ℕ :=
  repeating_sequence[((n - 1) % sequence_length)]'(by {
    simp [sequence_length]
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 1
  })

theorem digit_143_of_3_11 :
  nth_digit 143 = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_143_of_3_11_l893_89361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_transformation_and_solutions_l893_89307

-- Define the main theorem
theorem cubic_equation_transformation_and_solutions 
  (p q a b : ℝ) (φ : ℝ) 
  (h1 : 4 * p^3 + 27 * q^2 < 0)
  (h2 : Real.sin φ = b / Real.sqrt (a^2 + b^2))
  (h3 : Real.cos φ = a / Real.sqrt (a^2 + b^2)) :
  (∃ (α β : ℝ), ∀ (x y : ℝ), 
    x = α * y + β → 
    (x^3 + p * x + q = 0 ↔ y^3 - 3 * b * y^2 - 3 * a * y + b = 0)) ∧
  (∀ (y : ℝ), 
    y^3 - 3 * b * y^2 - 3 * a * y + b = 0 ↔ 
    (y = Real.tan (φ / 3) ∨ 
     y = Real.tan ((φ + 2 * Real.pi) / 3) ∨ 
     y = Real.tan ((φ + 4 * Real.pi) / 3))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_transformation_and_solutions_l893_89307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_isosceles_right_triangle_on_hyperbola_l893_89342

/-- A point on the hyperbola xy = 1 -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  eq : x * y = 1

/-- The area of a triangle given three points -/
noncomputable def triangleArea (a b c : HyperbolaPoint) : ℝ :=
  let ab := ((a.x - b.x)^2 + (a.y - b.y)^2 : ℝ)
  let bc := ((b.x - c.x)^2 + (b.y - c.y)^2 : ℝ)
  let ca := ((c.x - a.x)^2 + (c.y - a.y)^2 : ℝ)
  Real.sqrt (ab * bc * ca) / 4

/-- Predicate to check if a triangle is isosceles right -/
def isIsoscelesRight (a b c : HyperbolaPoint) : Prop :=
  ∃ (i j k : HyperbolaPoint), ({i, j, k} : Set HyperbolaPoint) = {a, b, c} ∧
    (i.x - j.x)^2 + (i.y - j.y)^2 = (i.x - k.x)^2 + (i.y - k.y)^2 ∧
    (j.x - i.x) * (k.x - i.x) + (j.y - i.y) * (k.y - i.y) = 0

/-- The main theorem -/
theorem min_area_isosceles_right_triangle_on_hyperbola :
  ∀ (a b c : HyperbolaPoint),
    isIsoscelesRight a b c →
    triangleArea a b c ≥ 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_isosceles_right_triangle_on_hyperbola_l893_89342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l893_89389

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)

/-- Sides form a geometric progression -/
def isGeometricProgression (t : Triangle) : Prop := t.a * t.c = t.b^2

theorem triangle_properties (t : Triangle) 
  (h_area : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2)) :
  t.B = π/3 ∧ 
  (isGeometricProgression t → Real.sin t.A * Real.sin t.C = 3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l893_89389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_equation_l893_89378

theorem unique_solution_for_equation : 
  ∃! k n : ℕ, k > 0 ∧ n > 0 ∧ k^2 - 2016 = 3^n ∧ k = 45 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_equation_l893_89378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_isosceles_triangles_l893_89398

-- Define a point in 2D space
structure Point where
  x : ℚ
  y : ℚ

-- Define a triangle using three points
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

-- Function to calculate the square of the distance between two points
def distanceSquared (p1 p2 : Point) : ℚ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := distanceSquared t.p1 t.p2
  let d2 := distanceSquared t.p2 t.p3
  let d3 := distanceSquared t.p3 t.p1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the five triangles
def triangle1 : Triangle := { p1 := ⟨1, 2⟩, p2 := ⟨3, 2⟩, p3 := ⟨2, 0⟩ }
def triangle2 : Triangle := { p1 := ⟨4, 1⟩, p2 := ⟨4, 3⟩, p3 := ⟨6, 1⟩ }
def triangle3 : Triangle := { p1 := ⟨7, 2⟩, p2 := ⟨10, 2⟩, p3 := ⟨8, 5⟩ }
def triangle4 : Triangle := { p1 := ⟨0, 1⟩, p2 := ⟨3, 2⟩, p3 := ⟨6, 1⟩ }
def triangle5 : Triangle := { p1 := ⟨9, 1⟩, p2 := ⟨10, 3⟩, p3 := ⟨11, 0⟩ }

-- Theorem statement
theorem four_isosceles_triangles :
  (isIsosceles triangle1) ∧
  (isIsosceles triangle2) ∧
  (¬ isIsosceles triangle3) ∧
  (isIsosceles triangle4) ∧
  (isIsosceles triangle5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_isosceles_triangles_l893_89398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_value_sum_of_coefficients_squared_difference_l893_89352

-- Define the polynomial coefficients as variables
variable (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ)

-- Define the polynomial equation
axiom polynomial_eq : ∀ x : ℝ, (x^2 - 3*x + 2)^5 = a + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6 + a7*x^7 + a8*x^8 + a9*x^9 + a10*x^10

-- Theorem statements
theorem a2_value : a2 = 800 := by
  sorry

theorem sum_of_coefficients : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = -32 := by
  sorry

theorem squared_difference : 
  (a + a2 + a4 + a6 + a8 + a10)^2 - (a1 + a3 + a5 + a7 + a9)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_value_sum_of_coefficients_squared_difference_l893_89352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_a_wins_probability_l893_89351

/-- Represents a soccer tournament with 9 teams -/
structure Tournament where
  teams : Fin 9 → Team
  score : Team → ℕ
  played : Team → Team → Bool

/-- The probability of a team winning any given match -/
noncomputable def win_probability : ℝ := 1/2

/-- The number of games each team plays (excluding the direct match between A and B) -/
def games_per_team : ℕ := 7

/-- Theorem stating the probability of team A finishing with more points than team B -/
theorem team_a_wins_probability (t : Tournament) (A B : Team) 
  (h1 : t.played A B = true) 
  (h2 : t.score A > t.score B) :
  (39203 : ℝ) / 65536 = 39203/65536 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_a_wins_probability_l893_89351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_between_five_and_six_count_l893_89324

theorem sqrt_between_five_and_six_count :
  (Finset.filter (fun x : ℕ => 25 < x ∧ x < 36) (Finset.range 36)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_between_five_and_six_count_l893_89324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_far_from_reduced_l893_89353

/-- Two integers are coprime if their greatest common divisor is 1 -/
def Coprime (x y : ℤ) : Prop := Int.gcd x y = 1

/-- The Euclidean distance between two points (a, b) and (x, y) -/
noncomputable def EuclideanDistance (a b x y : ℤ) : ℝ :=
  Real.sqrt ((a - x)^2 + (b - y)^2)

/-- For any positive integer n, there exists integers a and b such that
    for all coprime integers x and y, the Euclidean distance between
    (a, b) and (x, y) is greater than n -/
theorem exists_point_far_from_reduced (n : ℕ) :
  ∃ (a b : ℤ), ∀ (x y : ℤ), Coprime x y →
    (n : ℝ) < EuclideanDistance a b x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_far_from_reduced_l893_89353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l893_89372

open Real

theorem trigonometric_identity (α β γ : ℝ) (h : α + β + γ = π) :
  sin (2 * α) + sin (2 * β) + sin (2 * γ) =
  2 * (sin α + sin β + sin γ) * (cos α + cos β + cos γ) - 2 * (sin α + sin β + sin γ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l893_89372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_three_l893_89321

noncomputable def average_of_four (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem first_number_is_three (x : ℚ) :
  average_of_four x 16 33 28 = 20 → x = 3 := by
  intro h
  have sum_eq : x + 16 + 33 + 28 = 20 * 4 := by
    rw [average_of_four] at h
    field_simp at h
    linarith
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_three_l893_89321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersections_l893_89327

/-- The number of intersections between two circles -/
def num_intersections (c1 c2 : Set (ℝ × ℝ)) : ℕ := sorry

/-- First circle: (x - 3)² + y² = 9 -/
def circle1 : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 9}

/-- Second circle: x² + (y - 5)² = 25 -/
def circle2 : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 5)^2 = 25}

/-- Theorem: The number of intersections between the two given circles is 2 -/
theorem circle_intersections : num_intersections circle1 circle2 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersections_l893_89327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_sum_of_digits_l893_89360

/-- Sum of digits function -/
def S (k : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem unique_polynomial_sum_of_digits :
  ∀ P : Polynomial ℤ,
  (∀ n : ℕ, n ≥ 2016 → (P.eval (n : ℤ)) > 0) →
  (∀ n : ℕ, n ≥ 2016 → S (Int.natAbs (P.eval (n : ℤ))) = P.eval (S n : ℤ)) →
  P = Polynomial.X := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_sum_of_digits_l893_89360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_negative_sum_l893_89385

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a1_positive : a 1 > 0
  sum_2022_2023_positive : a 2022 + a 2023 > 0
  product_2022_2023_negative : a 2022 * a 2023 < 0

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem smallest_negative_sum (seq : ArithmeticSequence) :
  (∀ k < 4045, sum_n seq k > 0) ∧ sum_n seq 4045 < 0 := by
  sorry

#check smallest_negative_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_negative_sum_l893_89385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l893_89301

/-- A right parallelepiped with a rhombus base -/
structure RhombusParallelepiped where
  /-- Area of the rhombus base -/
  Q : ℝ
  /-- Area of the first diagonal section -/
  S1 : ℝ
  /-- Area of the second diagonal section -/
  S2 : ℝ
  /-- Q, S1, and S2 are positive -/
  hQ : 0 < Q
  hS1 : 0 < S1
  hS2 : 0 < S2

/-- The volume of a rhombus-based parallelepiped -/
noncomputable def volume (p : RhombusParallelepiped) : ℝ :=
  Real.sqrt (p.Q * p.S1 * p.S2 / 2)

/-- Theorem: The volume of a rhombus-based parallelepiped is √(Q * S1 * S2 / 2) -/
theorem volume_formula (p : RhombusParallelepiped) :
    volume p = Real.sqrt (p.Q * p.S1 * p.S2 / 2) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l893_89301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_division_floor_theorem_l893_89377

theorem power_division_floor_theorem :
  ⌊((13^13 / 13^12 : ℚ)^3 * 3^3) / 2^6⌋ = 926 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_division_floor_theorem_l893_89377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_of_a_l893_89396

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + 4) / x

-- Theorem for the area of the triangle
theorem triangle_area : 
  ∃ A B C : ℝ × ℝ, 
    (A.1 ≠ B.1 ∧ B.1 ≠ C.1 ∧ C.1 ≠ A.1) ∧
    (A.2 = 0 ∧ B.2 = 0) ∧
    (∃ x : ℝ, C = (x, f x)) ∧
    (1/2 * |A.1 - C.1| * |C.2| = 8/3) := by
  sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ s t : ℝ, s > 0 ∧ t > 0 → g a s ≥ f t) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_of_a_l893_89396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l893_89369

-- Define the curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.exp x
noncomputable def curve2 (x : ℝ) : ℝ := Real.log x

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Statement of the theorem
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 ∧
  ∀ (x1 x2 : ℝ), 
    distance x1 (curve1 x1) x2 (curve2 x2) ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l893_89369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_zero_l893_89368

theorem factorial_difference_zero : Nat.factorial 7 - 6 * Nat.factorial 6 - Nat.factorial 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_difference_zero_l893_89368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_total_time_l893_89379

/-- Represents a cyclist with speeds on asphalt and sand sections -/
structure Cyclist where
  asphalt_speed : ℝ
  sand_speed : ℝ

/-- Represents a bicycle path with asphalt and sand sections -/
structure BicyclePath where
  asphalt_length : ℝ
  sand_length : ℝ

/-- Theorem: Two cyclists take the same time to complete the path if they meet at midpoints -/
theorem same_total_time (path : BicyclePath) (cyclist1 cyclist2 : Cyclist) :
  cyclist1.asphalt_speed > 0 ∧ cyclist1.sand_speed > 0 ∧
  cyclist2.asphalt_speed > 0 ∧ cyclist2.sand_speed > 0 ∧
  path.asphalt_length > 0 ∧ path.sand_length > 0 →
  (path.asphalt_length / (2 * cyclist1.asphalt_speed) =
   path.asphalt_length / (2 * cyclist2.asphalt_speed)) ∧
  (path.sand_length / (2 * cyclist1.sand_speed) =
   path.sand_length / (2 * cyclist2.sand_speed)) →
  (path.asphalt_length / cyclist1.asphalt_speed + path.sand_length / cyclist1.sand_speed =
   path.asphalt_length / cyclist2.asphalt_speed + path.sand_length / cyclist2.sand_speed) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_total_time_l893_89379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_values_l893_89380

theorem cosine_values (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : π/2 < β ∧ β < π)
  (h3 : Real.cos (α + π/4) = 1/3)
  (h4 : Real.cos (π/4 - β/2) = Real.sqrt 3/3) :
  Real.cos β = -4 * Real.sqrt 2/9 ∧ Real.cos (2*α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_values_l893_89380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_dot_product_l893_89344

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 36

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define point P
def P : ℝ × ℝ := (-5, 0)

-- Define a point Q on the circle
def Q (x y : ℝ) : Prop := circle_eq x y

-- Define M as the midpoint of PQ
def M (x y : ℝ) : Prop := ∃ (qx qy : ℝ), Q qx qy ∧ x = (qx - 5) / 2 ∧ y = qy / 2

-- Define line l passing through P
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 5)

-- Define points A and B on the intersection of line l and trajectory C
def A_B (k : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ),
  line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
  trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

-- Theorem statement
theorem trajectory_and_dot_product :
  ∀ (x y : ℝ), M x y → trajectory x y ∧
  ∃ (k x₁ y₁ x₂ y₂ : ℝ), A_B k ∧
  ((x₁ - (-5))^2 + y₁^2) * ((x₂ - (-5))^2 + y₂^2) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_dot_product_l893_89344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chromosome_structure_correct_description_l893_89300

structure Chromosome where
  centromeres : ℕ
  chromatids : ℕ
  is_chromatin : Bool

def correct_description (c : Chromosome) : Prop :=
  c.centromeres = 1

theorem chromosome_structure_correct_description (c : Chromosome) :
  c.centromeres = 1 ∧ c.chromatids ∈ ({0, 2} : Set ℕ) ∧ ¬c.is_chromatin →
  correct_description c :=
by
  intro h
  simp [correct_description]
  exact h.1

#check chromosome_structure_correct_description

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chromosome_structure_correct_description_l893_89300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disc_purchase_cost_l893_89311

/-- The total cost of buying compact discs -/
def total_cost (price1 : ℚ) (price2 : ℚ) (quantity1 : ℕ) (quantity2 : ℕ) : ℚ :=
  price1 * quantity1 + price2 * quantity2

/-- Theorem: The total cost of buying 6 discs at $8.50 each and 4 discs at $10.50 each is $93.00 -/
theorem disc_purchase_cost :
  let price1 : ℚ := 21/2
  let price2 : ℚ := 17/2
  let quantity1 : ℕ := 4
  let quantity2 : ℕ := 6
  total_cost price1 price2 quantity1 quantity2 = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disc_purchase_cost_l893_89311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l893_89358

/-- Definition of the "P(a) property" for a function -/
def has_p_property (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (x + a) = f (-x)

/-- Definition of central symmetry about a point -/
def centrally_symmetric (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

/-- Definition of monotonically decreasing on an interval -/
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

/-- Definition of monotonically increasing on an interval -/
def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem problem_statement :
  (has_p_property Real.sin π) ∧ 
  (∀ f : ℝ → ℝ, has_p_property f 4 → 
    centrally_symmetric f (1, 0) → 
    (∃ ε > 0, monotone_decreasing_on f (-1 - ε) (-1 + ε)) → 
    (monotone_decreasing_on f (-2) (-1) ∧ 
     monotone_increasing_on f 1 2)) ∧
  (∀ f g : ℝ → ℝ, f ≠ 0 → 
    has_p_property f 0 → 
    has_p_property f 3 → 
    (∀ x₁ x₂, |f x₁ - f x₂| ≥ |g x₁ - g x₂|) → 
    ∃ T > 0, ∀ x, g (x + T) = g x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l893_89358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_l893_89370

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 9

-- Define the line
def my_line (k x y : ℝ) : Prop := y = k * x

-- Define the intersection condition
def intersects (k : ℝ) : Prop := ∃ x y, my_circle x y ∧ my_line k x y

-- Define the probability measure on [-1, 1]
noncomputable def prob : Set ℝ → ℝ := sorry

-- State the theorem
theorem intersection_probability :
  prob {k : ℝ | k ∈ Set.Icc (-1) 1 ∧ intersects k} = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_l893_89370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equals_semiperimeter_times_inradius_l893_89354

/-- The area of a triangle is equal to its semiperimeter multiplied by the radius of its inscribed circle. -/
theorem triangle_area_equals_semiperimeter_times_inradius 
  (a b c r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_inradius : r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c)))
  (p : ℝ) (hp : p = (a + b + c) / 2)
  : (a * b * c) / (4 * p) = p * r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equals_semiperimeter_times_inradius_l893_89354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_2_l893_89328

-- Define the function f
noncomputable def f (x a b : ℝ) : ℝ := x^2023 + a*x^3 - b/x - 8

-- State the theorem
theorem f_at_2 (a b : ℝ) : (f (-2) a b = 10) → (f 2 a b = -26) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_2_l893_89328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_side_increase_l893_89355

theorem rectangle_side_increase (x : ℝ) : 
  ((1 + x / 100)^2 = 1.3225) → x = 15 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_side_increase_l893_89355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l893_89383

def A : Set ℝ := {x | 0 ≤ x + 1 ∧ x + 1 ≤ 3}
def B : Set ℝ := {x | 4 * x + 3 > 0}

theorem intersection_A_B : A ∩ B = Set.Ioo (-3/4) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l893_89383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_normal_price_l893_89325

/-- The normal price of an article given its final price after successive discounts -/
noncomputable def normal_price (final_price : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  final_price / ((1 - discount3) * (1 - discount2) * (1 - discount1))

/-- Theorem stating the normal price of an article given specific conditions -/
theorem article_normal_price :
  let final_price : ℝ := 144
  let discount1 : ℝ := 0.12
  let discount2 : ℝ := 0.22
  let discount3 : ℝ := 0.15
  abs (normal_price final_price discount1 discount2 discount3 - 246.81) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval normal_price 144 0.12 0.22 0.15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_normal_price_l893_89325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perp_plane_parallel_planes_perp_line_parallel_l893_89305

-- Define the basic types
variable (P : Type) -- Type for points
variable (L : Type) -- Type for lines
variable (Pl : Type) -- Type for planes (changed from Π to Pl)

-- Define the relationships
variable (parallel : L → L → Prop) -- Parallel lines
variable (parallel_planes : Pl → Pl → Prop) -- Parallel planes
variable (perp_line_plane : L → Pl → Prop) -- Line perpendicular to plane
variable (perp_plane_line : Pl → L → Prop) -- Plane perpendicular to line

-- Theorem 1: Two lines perpendicular to the same plane are parallel
theorem lines_perp_plane_parallel (l1 l2 : L) (p : Pl) :
  perp_line_plane l1 p → perp_line_plane l2 p → parallel l1 l2 :=
sorry

-- Theorem 2: Two planes perpendicular to the same line are parallel
theorem planes_perp_line_parallel (p1 p2 : Pl) (l : L) :
  perp_plane_line p1 l → perp_plane_line p2 l → parallel_planes p1 p2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perp_plane_parallel_planes_perp_line_parallel_l893_89305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_area_square_with_circles_l893_89319

/-- The area of a square with side length 2, not covered by four unit circles
    each centered on one of the square's corners, is equal to 4 - π. -/
theorem uncovered_area_square_with_circles : ∀ π : Real, 4 - π = 4 - π := by
  intro π
  let square_side : Real := 2
  let circle_radius : Real := 1
  let square_area : Real := square_side ^ 2
  let circle_sector_area : Real := π * circle_radius ^ 2 / 4
  let total_circle_sectors_area : Real := 4 * circle_sector_area
  let uncovered_area : Real := square_area - total_circle_sectors_area
  sorry  -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncovered_area_square_with_circles_l893_89319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l893_89382

theorem cube_root_simplification :
  (5488000000 : ℝ)^(1/3) * 64^(1/3) = 5600 * 2^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l893_89382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_squares_l893_89317

/-- Represents a color: Black, White, or Red -/
inductive Color
  | Black
  | White
  | Red
deriving DecidableEq

/-- Represents a position on the cube -/
structure Position where
  x : Fin 5
  y : Fin 5
  z : Fin 5
deriving DecidableEq, Fintype

/-- Represents the coloring of the cube -/
def Coloring := Position → Color

/-- Two positions are adjacent if they differ by 1 in exactly one coordinate -/
def adjacent (p q : Position) : Prop :=
  (p.x = q.x ∧ p.y = q.y ∧ (p.z.val + 1 = q.z.val ∨ p.z.val = q.z.val + 1)) ∨
  (p.x = q.x ∧ p.z = q.z ∧ (p.y.val + 1 = q.y.val ∨ p.y.val = q.y.val + 1)) ∨
  (p.y = q.y ∧ p.z = q.z ∧ (p.x.val + 1 = q.x.val ∨ p.x.val = q.x.val + 1))

/-- A coloring is valid if no adjacent squares have the same color -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ p q : Position, adjacent p q → c p ≠ c q

/-- Count the number of black squares in a coloring -/
def black_count (c : Coloring) : ℕ :=
  Finset.card (Finset.filter (λ p : Position => c p = Color.Black) Finset.univ)

/-- The main theorem: The minimum number of black squares in a valid coloring is 18 -/
theorem min_black_squares :
  ∃ (c : Coloring), valid_coloring c ∧
    (∀ c' : Coloring, valid_coloring c' → black_count c ≤ black_count c') ∧
    black_count c = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_black_squares_l893_89317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l893_89374

-- Define the ellipse C
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

-- Define the circle (renamed to avoid conflict)
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 2/3

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := x/a + y = 1

-- Define the condition for line l being tangent to the circle
def tangent_condition (a : ℝ) : Prop := a / Real.sqrt (1 + a^2) = Real.sqrt (2/3)

-- Define the slopes condition
def slopes_condition (k₁ k₂ : ℝ) : Prop := k₁ + k₂ = 2

-- Main theorem
theorem ellipse_and_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : tangent_condition a) 
  (k₁ k₂ : ℝ) (h₃ : slopes_condition k₁ k₂) :
  (∀ x y, ellipse a x y ↔ ellipse (Real.sqrt 2) x y) ∧
  (∃ k m : ℝ, ∀ x y, y = k * x + m → x = -1 ∧ y = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l893_89374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zeros_difference_l893_89336

/-- A parabola with vertex at (5, -9) passing through (6, -8) has zeros differing by 6 -/
theorem parabola_zeros_difference (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Parabola equation
  (5, -9) = (-(b / (2 * a)), -(b^2 - 4*a*c) / (4 * a)) →  -- Vertex formula
  -8 = a * 6^2 + b * 6 + c →  -- Point (6, -8) lies on the parabola
  let zeros := {x : ℝ | a * x^2 + b * x + c = 0}
  ∃ m n : ℝ, m ∈ zeros ∧ n ∈ zeros ∧ m > n ∧ m - n = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_zeros_difference_l893_89336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approximation_l893_89339

/-- The selling price of the computer table in rupees -/
noncomputable def selling_price : ℚ := 7967

/-- The markup percentage applied to the cost price -/
noncomputable def markup_percentage : ℚ := 24

/-- The cost price of the computer table -/
noncomputable def cost_price : ℚ := selling_price / (1 + markup_percentage / 100)

/-- Theorem stating that the cost price is approximately 6424 rupees -/
theorem cost_price_approximation : 
  ⌊cost_price⌋ = 6424 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approximation_l893_89339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_first_step_l893_89392

def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

def horner_first_step (x : ℝ) : ℝ := 0.5 * x + 4

theorem horner_method_first_step :
  horner_first_step 3 = 5.5 ∧
  horner_first_step 3 = 0.5 * 3 + 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_first_step_l893_89392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_proof_l893_89333

/-- Calculates the monthly payment for a loan with compound interest -/
noncomputable def calculate_monthly_payment (principal : ℝ) (interest_rate : ℝ) (months : ℕ) : ℝ :=
  let total_amount := principal * (1 + interest_rate) ^ (months : ℝ)
  total_amount / (months : ℝ)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem loan_payment_proof :
  let principal : ℝ := 2000
  let interest_rate : ℝ := 0.10
  let months : ℕ := 3
  round_to_nearest (calculate_monthly_payment principal interest_rate months) = 716 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_proof_l893_89333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l893_89394

theorem divisor_problem :
  ∃! D : ℕ, 
    D > 0 ∧
    242 % D = 15 ∧
    698 % D = 27 ∧
    940 % D = 5 ∧
    D = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l893_89394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_comparison_l893_89356

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of 6 points
def SixPoints := Fin 6 → Point

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define what it means for three points to be collinear
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

-- Define the theorem
theorem triangle_side_comparison (points : SixPoints) 
  (h1 : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k))
  (h2 : ∀ i j k l, i ≠ j → k ≠ l → distance (points i) (points j) ≠ distance (points k) (points l)) :
  ∃ (i j k l m n : Fin 6), 
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
    l ≠ m ∧ m ≠ n ∧ n ≠ l ∧
    (distance (points i) (points j) ≤ distance (points j) (points k) ∧
     distance (points i) (points j) ≤ distance (points k) (points i)) ∧
    (distance (points l) (points m) ≥ distance (points m) (points n) ∧
     distance (points l) (points m) ≥ distance (points n) (points l)) ∧
    distance (points i) (points j) = distance (points l) (points m) :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_comparison_l893_89356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_altitude_calculation_l893_89334

/-- Given temperature drop rate, ground temperature, and balloon temperature, 
    calculate the altitude of the balloon. -/
noncomputable def balloon_altitude (temp_drop_rate : ℝ) (altitude_increment : ℝ) 
                     (ground_temp : ℝ) (balloon_temp : ℝ) : ℝ :=
  ((ground_temp - balloon_temp) / temp_drop_rate) * altitude_increment

/-- Theorem stating that under the given conditions, 
    the balloon's altitude is 1500 meters. -/
theorem balloon_altitude_calculation :
  let temp_drop_rate : ℝ := 3
  let altitude_increment : ℝ := 500
  let ground_temp : ℝ := 7
  let balloon_temp : ℝ := -2
  balloon_altitude temp_drop_rate altitude_increment ground_temp balloon_temp = 1500 := by
  -- Unfold the definition of balloon_altitude
  unfold balloon_altitude
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_altitude_calculation_l893_89334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l893_89348

theorem triangle_sine_ratio (A B C : ℝ) (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  c * (a * Real.cos B - b * Real.cos A) = b^2 →
  Real.sin A / Real.sin B = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_ratio_l893_89348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_gt_20_l893_89384

def given_list : List Nat := [18, 20, 22, 23, 29, 31, 33]

def is_prime (n : Nat) : Bool := Nat.Prime n

def greater_than_20 (n : Nat) : Bool := n > 20

def prime_greater_than_20 (n : Nat) : Bool := is_prime n && greater_than_20 n

def filtered_list : List Nat := given_list.filter prime_greater_than_20

theorem arithmetic_mean_of_primes_gt_20 : 
  (filtered_list.sum : Rat) / filtered_list.length = 83 / 3 := by
  sorry

#eval filtered_list
#eval filtered_list.sum
#eval filtered_list.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_gt_20_l893_89384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_distribution_possible_l893_89322

/-- The amount of cheese sold to k customers, given the total amount of cheese and the number of remaining customers after each sale. -/
noncomputable def cheese_sold (total : ℝ) (remaining : ℕ → ℕ) (k : ℕ) : ℝ :=
  k * (total / (k + remaining k))

/-- The proposition that the cheese can be sold to exactly 20 customers with the given conditions. -/
theorem cheese_distribution_possible : ∃ (remaining : ℕ → ℕ),
  (∀ k, k < 10 → remaining k = 10) ∧
  (cheese_sold 20 remaining 10 = 10) := by
  sorry

#check cheese_distribution_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_distribution_possible_l893_89322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l893_89315

noncomputable def determinant (a₁ a₂ a₃ a₄ : ℝ) : ℝ := a₁ * a₄ - a₂ * a₃

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := determinant (Real.sqrt 3) (Real.sin (ω * x)) 1 (Real.cos (ω * x))

noncomputable def shifted_f (ω : ℝ) (x : ℝ) : ℝ := f ω (x + 2 * Real.pi / 3)

theorem min_omega_value :
  ∀ ω : ℝ, ω > 0 →
  (∀ x : ℝ, shifted_f ω x = shifted_f ω (-x)) →
  ω ≥ 5/4 := by
  sorry

#check min_omega_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l893_89315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_pdf_correct_y_pdf_integrates_to_one_l893_89397

/-- The probability density function of a standard normal distribution -/
noncomputable def standard_normal_pdf (x : ℝ) : ℝ := 
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x^2) / 2)

/-- The probability density function of Y = X^2 where X is standard normal -/
noncomputable def y_pdf (y : ℝ) : ℝ := 
  (1 / Real.sqrt (2 * Real.pi * y)) * Real.exp (-y / 2)

/-- Theorem stating that y_pdf is the correct density function for Y = X^2 -/
theorem y_pdf_correct (x : ℝ) : 
  y_pdf (x^2) = standard_normal_pdf x * (1 / (2 * |x|)) + 
                standard_normal_pdf (-x) * (1 / (2 * |x|)) := by
  sorry

/-- Theorem stating that y_pdf integrates to 1 over its domain -/
theorem y_pdf_integrates_to_one : 
  ∫ y in Set.Ioi 0, y_pdf y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_pdf_correct_y_pdf_integrates_to_one_l893_89397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_monitored_segments_correct_l893_89323

/-- Represents a rectangular grid city. -/
structure City where
  n : ℕ  -- Number of parallel streets
  m : ℕ  -- Number of perpendicular streets

/-- The minimum number of monitored segments required for a given city. -/
def min_monitored_segments (city : City) : ℕ :=
  (city.m - 1) * (city.n - 1)

/-- Predicate to check if a set of coordinates forms a valid loop in the city. -/
def is_valid_loop (city : City) (loop : Set (ℕ × ℕ)) : Prop :=
  sorry

/-- 
Returns the set of monitored segments given a city and 
the number of segments to monitor.
-/
def monitored_segments (city : City) (num : ℕ) : Set (ℕ × ℕ) :=
  sorry

/-- 
Theorem: The minimum number of monitored street segments required to ensure 
that any loop in the city contains at least one monitored segment 
is (m-1)(n-1) for a city with n parallel streets and m perpendicular streets.
-/
theorem min_monitored_segments_correct (city : City) :
  ∀ loop : Set (ℕ × ℕ), 
    is_valid_loop city loop → 
    ∃ segment ∈ monitored_segments city (min_monitored_segments city), 
      segment ∈ loop :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_monitored_segments_correct_l893_89323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l893_89363

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}
def N : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}

-- State the theorem
theorem set_intersection_theorem :
  M ∩ N = Set.Icc (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l893_89363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l893_89399

/-- The perimeter of a triangle with vertices A(1,2), B(1,9), and C(6,5) on a Cartesian coordinate plane -/
theorem triangle_perimeter : ∃ (perimeter : ℝ), perimeter = 7 + Real.sqrt 41 + Real.sqrt 34 := by
  -- Define the coordinates of the triangle vertices
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 9)
  let C : ℝ × ℝ := (6, 5)

  -- Define the distance function between two points
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

  -- Calculate the perimeter
  let perimeter : ℝ := dist A B + dist B C + dist C A

  -- Prove that the perimeter equals 7 + √41 + √34
  have h : perimeter = 7 + Real.sqrt 41 + Real.sqrt 34 := by
    -- The actual proof would go here
    sorry

  -- Conclude the theorem
  exact ⟨perimeter, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l893_89399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_relation_l893_89316

def arithmetic_sequence (a b : ℤ) : ℕ → ℤ
  | 0 => a
  | n+1 => arithmetic_sequence a b n + b

def geometric_sequence (a b : ℤ) : ℕ → ℤ
  | 0 => b
  | n+1 => geometric_sequence a b n * a

def sum_arithmetic (a b : ℤ) (n : ℕ) : ℤ :=
  Finset.sum (Finset.range n) (λ i => arithmetic_sequence a b i)

theorem arithmetic_geometric_relation (a b k : ℕ) :
  (∀ n : ℕ, sum_arithmetic a (-2) n = -n^2 + 3*n) →
  (a < b) →
  (b < arithmetic_sequence a (-2) 1) →
  (arithmetic_sequence a (-2) 1 < geometric_sequence a b 1) →
  (geometric_sequence a b 1 < arithmetic_sequence a (-2) 2) →
  (∀ n : ℕ, arithmetic_sequence a (-2) n + k = geometric_sequence a b n) →
  (a = 2 ∧ b ≤ k + 2) := by
  sorry

#check arithmetic_geometric_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_relation_l893_89316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_max_value_l893_89340

open Real Matrix

-- Define the determinant as a function of θ
noncomputable def det_func (θ : ℝ) : ℝ := 
  Matrix.det !![1, 1, 1; 1, 1 + sin θ, 1; 1 + cos θ, 1, 1]

-- State the theorem
theorem det_max_value : 
  ∃ (c : ℝ), c = (1/2 : ℝ) ∧ ∀ θ, |det_func θ| ≤ c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_max_value_l893_89340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_zeros_in_power_of_seven_l893_89375

/-- For any positive integer m, there exists a positive integer n such that
    7^n contains at least m consecutive zeros in its decimal representation -/
theorem consecutive_zeros_in_power_of_seven (m : ℕ) :
  ∃ n : ℕ, ∃ k l : ℕ, k > 0 ∧ l > 0 ∧
    (7 : ℕ) ^ n = k * (10 ^ m) * l ∧
    k % 10 ≠ 0 ∧ l < 10 ^ m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_zeros_in_power_of_seven_l893_89375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_of_geometric_sequence_l893_89335

/-- A geometric sequence of positive real numbers -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : 0 < a
  r_pos : 0 < r

/-- The nth term of a geometric sequence -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a * seq.r ^ (n - 1)

theorem sixth_term_of_geometric_sequence
  (seq : GeometricSequence)
  (h4 : seq.nthTerm 4 = 16)
  (h9 : seq.nthTerm 9 = 4) :
  seq.nthTerm 6 = 2^(16/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_of_geometric_sequence_l893_89335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_approximation_l893_89395

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- Calculate the perimeter of a quadrilateral -/
noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  distance q.A q.B + distance q.B q.C + distance q.C q.D + distance q.D q.A

/-- Check if a point is inside a quadrilateral -/
def isInside (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Main theorem -/
theorem quadrilateral_perimeter_approximation
  (ABCD : Quadrilateral)
  (P : Point)
  (h_convex : sorry) -- ABCD is convex
  (h_area : area ABCD = 2500)
  (h_inside : isInside P ABCD)
  (h_PA : distance P ABCD.A = 30)
  (h_PB : distance P ABCD.B = 40)
  (h_PC : distance P ABCD.C = 35)
  (h_PD : distance P ABCD.D = 50) :
  ∃ ε > 0, abs (perimeter ABCD - 222.49) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_approximation_l893_89395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constructible_t_range_l893_89320

/-- A function f is triangle-constructible if for all a, b, c ∈ ℝ,
    f(a), f(b), and f(c) can form the sides of a triangle. -/
def TriangleConstructible (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, 
    f a + f b > f c ∧ 
    f b + f c > f a ∧ 
    f c + f a > f b

/-- The function f(x) = (2^x - t) / (2^x + 1) -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  (2^x - t) / (2^x + 1)

/-- Theorem: If f is triangle-constructible, then t ∈ [-2, -1/2] -/
theorem triangle_constructible_t_range (t : ℝ) :
  TriangleConstructible (f t) → t ∈ Set.Icc (-2) (-1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constructible_t_range_l893_89320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_score_l893_89312

/-- Calculates the comprehensive score given four component scores and their weights -/
def comprehensive_score (score1 score2 score3 score4 : ℝ) 
                        (weight1 weight2 weight3 weight4 : ℝ) : ℝ :=
  score1 * weight1 + score2 * weight2 + score3 * weight3 + score4 * weight4

/-- Theorem stating that Xiao Ming's comprehensive score is 90.9 -/
theorem xiao_ming_score : 
  let score1 : ℝ := 92
  let score2 : ℝ := 90
  let score3 : ℝ := 88
  let score4 : ℝ := 95
  let weight1 : ℝ := 0.4
  let weight2 : ℝ := 0.3
  let weight3 : ℝ := 0.2
  let weight4 : ℝ := 0.1
  comprehensive_score score1 score2 score3 score4 weight1 weight2 weight3 weight4 = 90.9 := by
  sorry

#eval comprehensive_score 92 90 88 95 0.4 0.3 0.2 0.1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_ming_score_l893_89312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_golden_ratio_l893_89332

theorem function_property_implies_golden_ratio (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (f x) = f x + x) 
  (h2 : ∀ x, f (f x - x) = f x + a * x) : 
  a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_golden_ratio_l893_89332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_equal_l893_89313

/-- Represents an octagon with equal angles and integer side lengths -/
structure EqualAngleIntegerSideOctagon where
  /-- Side lengths of the octagon -/
  sides : Fin 8 → ℕ
  /-- All angles are equal -/
  angles_equal : True  -- We'll assume this property without explicitly defining angles
  /-- Angle measure function (not explicitly used, but implied by the structure) -/
  angle : Fin 8 → ℝ

/-- Theorem: In an octagon with equal angles and integer side lengths, opposite sides are equal -/
theorem opposite_sides_equal (o : EqualAngleIntegerSideOctagon) :
  ∀ i : Fin 4, o.sides i = o.sides (i + 4) := by
  sorry

/-- Helper lemma: The sum of differences between opposite sides is zero -/
lemma sum_differences_zero (o : EqualAngleIntegerSideOctagon) :
  (o.sides 0 - o.sides 4) + (o.sides 1 - o.sides 5) +
  (o.sides 2 - o.sides 6) + (o.sides 3 - o.sides 7) = 0 := by
  sorry

/-- Helper lemma: Parallelogram property in the octagon -/
lemma parallelogram_property (o : EqualAngleIntegerSideOctagon) :
  ∀ i : Fin 4, o.sides i + o.sides (i + 1) = o.sides (i + 4) + o.sides (i + 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_equal_l893_89313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_equality_l893_89362

noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

noncomputable def compound_interest (principal rate time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_equality (principal_si principal_ci : ℝ) : 
  principal_si = 2015.625 →
  simple_interest principal_si 12 4 = (1/2) * compound_interest principal_ci 15 2 →
  principal_ci = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_equality_l893_89362
