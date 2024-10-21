import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_unit_sphere_with_cap_removed_l867_86775

/-- The volume of a sphere with a spherical cap removed -/
noncomputable def sphere_with_cap_removed_volume (R : ℝ) (h : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3 - (Real.pi * h^2 * (3 * R - h)) / 3

/-- Theorem: Volume of a unit sphere with a specific cap removed -/
theorem volume_unit_sphere_with_cap_removed :
  sphere_with_cap_removed_volume 1 (1/2) = 59 * Real.pi / 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_unit_sphere_with_cap_removed_l867_86775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l867_86780

-- Define the quadratic function f
noncomputable def f (x : ℝ) : ℝ := x^2 + x + 1

-- Define g in terms of f
noncomputable def g (x : ℝ) : ℝ := 2^(f x)

-- Theorem statement
theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = 2 * x + 2) ∧
  f 0 = 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, g x ∈ Set.Icc (2^(3/4)) 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l867_86780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_leap_years_in_180_years_l867_86701

/-- A function that determines if a year is a leap year based on the given rule -/
def isLeapYear (year : ℕ) : Bool :=
  year % 4 = 0 && (year % 100 ≠ 0 || year % 400 = 0)

/-- The maximum number of leap years in any 180-year period -/
def maxLeapYears : ℕ := 43

/-- Theorem stating that the maximum number of leap years in any 180-year period is 43 -/
theorem max_leap_years_in_180_years :
  ∀ start : ℕ, (Finset.filter (fun y => isLeapYear (start + y)) (Finset.range 180)).card ≤ maxLeapYears :=
by
  intro start
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_leap_years_in_180_years_l867_86701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l867_86713

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3^(x + 1)

-- Define the domain of the original function
def domain_f : Set ℝ := {x | -1 ≤ x ∧ x < 0}

-- Define the inverse function
noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 3 - 1

-- Define the domain of the inverse function
def domain_f_inv : Set ℝ := {x | 1 ≤ x ∧ x < 3}

-- Theorem stating that f_inv is the inverse of f on their respective domains
theorem f_inverse_correct :
  (∀ x ∈ domain_f, f_inv (f x) = x) ∧
  (∀ y ∈ domain_f_inv, f (f_inv y) = y) := by
  sorry

#check f_inverse_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l867_86713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_discount_theorem_l867_86788

/-- The discount percentage for concert tickets -/
noncomputable def discount_percentage : ℝ → ℝ → ℝ := fun original_price discounted_price =>
  (1 - discounted_price / original_price) * 100

/-- Theorem: The discount percentage is 25% when 4 discounted tickets cost the same as 3 non-discounted tickets -/
theorem concert_discount_theorem (P : ℝ) (h : P > 0) :
  discount_percentage (3 * P) (4 * P * (1 - 25 / 100)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_discount_theorem_l867_86788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_elements_l867_86755

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the set A
def A : Set ℂ := {x | ∃ n : ℕ, n > 0 ∧ x = i^n + (-i)^n}

-- Statement to prove
theorem set_A_elements : A = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_elements_l867_86755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l867_86773

/-- The area of a triangle with vertices at (3, 1), (3, 6), and (8, 6) is 12.5 square units. -/
theorem triangle_area : ∃ (A : ℝ), A = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l867_86773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sums_theorem_l867_86748

/-- Definition of binomial coefficient -/
def binomial (n m : ℕ) : ℕ :=
  if m > n then 0 else Nat.choose n m

/-- Definition of Sn -/
noncomputable def Sn (n : ℕ) : ℝ := ∑' k, (binomial n (3*k) : ℝ)

/-- Definition of Tn -/
noncomputable def Tn (n : ℕ) : ℝ := ∑' k, (binomial n (3*k + 1) : ℝ)

/-- Definition of Hn -/
noncomputable def Hn (n : ℕ) : ℝ := ∑' k, (binomial n (3*k + 2) : ℝ)

/-- Main theorem -/
theorem sums_theorem (n : ℕ) (hn : n > 0) :
  Sn n = (1/3) * (2^n + 2 * Real.cos (n * π / 3)) ∧
  Tn n = (1/3) * (2^n + 2 * Real.cos ((n - 2) * π / 3)) ∧
  Hn n = (1/3) * (2^n - 2 * Real.cos ((n - 1) * π / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sums_theorem_l867_86748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solutions_l867_86740

theorem quadratic_equation_solutions (a b : ℝ) :
  ((∃! x, x^2 + a*x + b = 0) ↔ a^2 = 4*b) ∧
  ((∃ x y, x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0 ∧ x = 1 ∧ y = 3) ↔ a = -4 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solutions_l867_86740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_one_one_min_radius_for_intersection_l867_86738

-- Define the line equation
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x - k + 1

-- Define the circle equation
def circle_eq (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2

-- Theorem 1: The line always passes through (1, 1)
theorem line_passes_through_point_one_one (k : ℝ) : line k 1 1 := by
  -- Proof
  sorry

-- Theorem 2: Minimum radius for intersection
theorem min_radius_for_intersection (r : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, line k x y ∧ circle_eq r x y) → r ≥ Real.sqrt 2 := by
  -- Proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_one_one_min_radius_for_intersection_l867_86738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_on_ellipse_l867_86752

def A : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (0, -3)

def on_ellipse (p : ℝ × ℝ) : Prop :=
  (p.1^2 / 16) + (p.2^2 / 25) = 1

theorem triangle_ratio_on_ellipse (B : ℝ × ℝ) 
  (h : on_ellipse B) : 
  ∃ (a b c : ℝ), 
    (Real.sin (Real.arctan ((B.2 - 3) / B.1) + Real.arctan ((B.2 + 3) / B.1))) / 
    (Real.sin (Real.arctan ((B.2 - 3) / B.1)) + Real.sin (Real.arctan ((B.2 + 3) / B.1))) = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_on_ellipse_l867_86752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_frustum_volume_l867_86763

/-- A model consisting of a hemisphere and a frustum -/
structure HemisphereFrustumModel where
  hemisphere_volume : ℝ
  frustum_large_radius : ℝ
  frustum_small_radius : ℝ
  frustum_height : ℝ

/-- The total volume of the hemisphere-frustum model -/
noncomputable def total_volume (model : HemisphereFrustumModel) : ℝ :=
  model.hemisphere_volume + 
  (1/3) * Real.pi * model.frustum_height * 
  (model.frustum_small_radius^2 + model.frustum_small_radius * model.frustum_large_radius + model.frustum_large_radius^2)

/-- Theorem stating the total volume of the model under given conditions -/
theorem hemisphere_frustum_volume : 
  ∀ (model : HemisphereFrustumModel),
  model.hemisphere_volume = 144 * Real.pi ∧
  model.frustum_small_radius = (1/2) * model.frustum_large_radius ∧
  model.frustum_height = (1/2) * model.frustum_large_radius →
  total_volume model = 648 * Real.pi := by
  sorry

#check hemisphere_frustum_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_frustum_volume_l867_86763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_root_inequality_l867_86781

theorem nth_root_inequality (a : ℝ) (n : ℕ) (hn : n > 0) :
  (a > 1 → (a ^ (1 / n : ℝ)) > 1) ∧ (a < 1 → (a ^ (1 / n : ℝ)) < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_root_inequality_l867_86781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l867_86717

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (7*θ) = 49/2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_theta_l867_86717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_copy_cost_l867_86791

/-- The cost of Sandy's copies --/
theorem sandys_copy_cost 
  (colored_cost : ℝ)
  (white_cost : ℝ)
  (total_copies : ℕ)
  (colored_copies : ℕ)
  (h1 : colored_cost = 0.10)
  (h2 : white_cost = 0.05)
  (h3 : total_copies = 400)
  (h4 : colored_copies = 50) :
  colored_cost * (colored_copies : ℝ) + white_cost * ((total_copies - colored_copies) : ℝ) = 22.50 := by
  sorry

#check sandys_copy_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_copy_cost_l867_86791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_is_six_l867_86776

/-- The set S containing elements from 1 to 10 -/
def S : Finset ℕ := Finset.range 10

/-- A family of subsets of S -/
def A : ℕ → Finset ℕ := sorry

/-- The number of subsets in the family A -/
def k : ℕ := sorry

/-- Each subset in A has exactly 5 elements -/
axiom subset_size : ∀ i : ℕ, i ≤ k → Finset.card (A i) = 5

/-- Each subset in A is a subset of S -/
axiom subset_of_S : ∀ i : ℕ, i ≤ k → A i ⊆ S

/-- The intersection of any two distinct subsets in A has at most 2 elements -/
axiom intersection_size : ∀ i j : ℕ, i < j → j ≤ k → Finset.card (A i ∩ A j) ≤ 2

/-- The maximum value of k is 6 -/
theorem max_k_is_six : k ≤ 6 ∧ ∃ A : ℕ → Finset ℕ, ∃ k : ℕ, k = 6 ∧
  (∀ i : ℕ, i ≤ k → Finset.card (A i) = 5) ∧
  (∀ i : ℕ, i ≤ k → A i ⊆ S) ∧
  (∀ i j : ℕ, i < j → j ≤ k → Finset.card (A i ∩ A j) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_is_six_l867_86776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_range_l867_86736

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2

-- Define the theorem
theorem triangle_area_and_function_range :
  ∀ (A B C : ℝ) (a b c : ℝ),
  -- Conditions
  (∀ x, -3 ≤ f x ∧ f x ≤ 1) →  -- Range of f
  f A = -2 →                   -- Given condition
  a = Real.sqrt 3 →            -- Given condition
  -- Conclusions
  (∃ (S : ℝ), S ≤ Real.sqrt 3 / 4 ∧ 
    (S = Real.sqrt 3 / 4 → 
      S = 1/2 * b * c * Real.sin A ∧ 
      a^2 = b^2 + c^2 - 2*b*c*Real.cos A)) :=
by
  sorry

#check triangle_area_and_function_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_range_l867_86736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_proof_l867_86719

theorem cosine_value_proof (θ : Real) (a b : Fin 2 → ℝ) :
  θ ∈ Set.Ioo π (2 * π) →
  a = ![1, 2] →
  b = ![Real.cos θ, Real.sin θ] →
  ∃ k : ℝ, k ≠ 0 ∧ a = k • b →
  Real.cos θ = -Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_proof_l867_86719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_angle_product_l867_86704

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if the angle between three points is a right angle -/
def isRightAngle (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y) = 0

/-- Theorem: For any point P on the ellipse x²/49 + y²/24 = 1, 
    if the angle between PF₁ and PF₂ is a right angle, 
    then |PF₁| · |PF₂| = 48, where F₁ and F₂ are the foci of the ellipse -/
theorem ellipse_right_angle_product (e : Ellipse) (p f1 f2 : Point) :
  e.a = 7 ∧ e.b = 2 * Real.sqrt 6 ∧
  isOnEllipse p e ∧
  isRightAngle f1 p f2 →
  distance p f1 * distance p f2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_angle_product_l867_86704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_set_B_l867_86782

def U : Set ℕ := {1,2,3,4,5,6,7}

def A : Set ℕ := sorry

def B : Set ℕ := sorry

theorem determine_set_B (h1 : U = A ∪ B) (h2 : A ∩ (U \ B) = {2,4,6}) :
  B = {1,3,5,7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_set_B_l867_86782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l867_86790

-- Define the function f(x) = √x + √(6 - 2x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x + Real.sqrt (6 - 2 * x)

-- Define the domain of x
def valid_x (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Define the theorem
theorem solution_count (a : ℝ) :
  (∃! x, valid_x x ∧ f x = a ∧ (a = 3 ∨ (Real.sqrt 3 ≤ a ∧ a < Real.sqrt 6))) ∨
  (∃ x y, x ≠ y ∧ valid_x x ∧ valid_x y ∧ f x = a ∧ f y = a ∧ Real.sqrt 6 ≤ a ∧ a < 3) ∨
  (∀ x, valid_x x → f x ≠ a ∧ (a < Real.sqrt 3 ∨ 3 < a)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l867_86790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_decreasing_sine_l867_86766

theorem omega_range_for_decreasing_sine (ω : ℝ) (h_pos : ω > 0) :
  (∀ x₁ x₂ : ℝ, π / 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < π → 
    Real.sin (ω * x₁ + π / 4) ≥ Real.sin (ω * x₂ + π / 4)) →
  ω ∈ Set.Icc (1 / 2 : ℝ) (5 / 4 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_decreasing_sine_l867_86766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l867_86797

theorem tan_alpha_plus_pi_fourth (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 3 / 5) :
  Real.tan (α + π / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l867_86797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l867_86772

-- Define the domains A and B
def A : Set ℝ := {x : ℝ | 4 - x^2 ≥ 0}
def B : Set ℝ := {x : ℝ | 1 - x > 0}

-- Theorem statement
theorem domain_intersection :
  A ∩ B = Set.Icc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l867_86772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l867_86764

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin (π - α) = 1/3) : 
  Real.cos α = -2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l867_86764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_combinations_l867_86795

def num_cone_types : ℕ := 3
def num_flavors : ℕ := 6
def max_scoops : ℕ := 3
def num_topping_types : ℕ := 6
def max_toppings : ℕ := 3

def scoop_permutations (n : ℕ) (k : ℕ) : ℕ := n ^ k

def topping_combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

def total_scoop_permutations : ℕ :=
  Finset.sum (Finset.range (max_scoops + 1)) (fun k => scoop_permutations num_flavors k)

def total_topping_combinations : ℕ :=
  Finset.sum (Finset.range (max_toppings + 1)) (fun k => topping_combinations num_topping_types k)

theorem ice_cream_combinations : 
  num_cone_types * total_scoop_permutations * total_topping_combinations = 65016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_combinations_l867_86795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_solution_l867_86779

theorem sequence_solution (a b : ℝ) : 
  (∃ (s : ℕ → ℝ), s 0 = Real.sqrt 5 / 3 ∧ 
                   s 1 = Real.sqrt 10 / 8 ∧ 
                   s 2 = Real.sqrt 17 / (a + b) ∧ 
                   s 3 = Real.sqrt (a - b) / 24 ∧ 
                   s 4 = Real.sqrt 37 / 35) →
  a = 41 / 2 ∧ b = 11 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_solution_l867_86779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_l867_86756

/-- The measure of an exterior angle of a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ := by
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  have h : exterior_angle = 45 := by sorry
  exact exterior_angle


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_l867_86756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l867_86789

/-- The profit function y(x) where x is the advertising cost in ten thousand yuan -/
noncomputable def y (x : ℝ) : ℝ := 21 - x - 18 / (2 * x + 1)

/-- The domain of the profit function -/
def Domain : Set ℝ := {x : ℝ | x > 0}

theorem profit_maximization :
  ∃ (x_max : ℝ), x_max ∈ Domain ∧
  (∀ (x : ℝ), x ∈ Domain → y x ≤ y x_max) ∧
  x_max = 5/2 ∧ y x_max = 31/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l867_86789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l867_86751

noncomputable def a (n : ℕ) : ℝ := (1 + 1 / (n : ℝ)) ^ n
noncomputable def b (n : ℕ) : ℝ := (1 + 1 / (n : ℝ)) ^ (n + 1)
noncomputable def f (x : ℝ) : ℝ := (1 + 1 / x) ^ x

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) ∧
  (∀ n : ℕ, n > 0 → b (n + 1) < b n) ∧
  (∀ n : ℕ, n > 0 → b n > a n) ∧
  (∀ x y : ℝ, x > 0 → y > x → f y > f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l867_86751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ckb_l867_86792

/-- Given a right triangle ABC with leg BC = a and leg AC = b, and a circle
    constructed on BC as diameter intersecting hypotenuse AB at point K,
    the area of triangle CKB is (a³b) / (2(a² + b²)). -/
theorem area_triangle_ckb (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let k := c / 2
  (a^2 * k) / (2 * c) = (a^3 * b) / (2 * (a^2 + b^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ckb_l867_86792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_place_mat_length_approx_l867_86777

/-- The radius of the round table in feet -/
def table_radius : ℝ := 5

/-- The number of place mats on the table -/
def num_mats : ℕ := 8

/-- The width of each place mat in feet -/
def mat_width : ℝ := 1

/-- The length of each place mat in feet -/
noncomputable def mat_length : ℝ := Real.sqrt 24.75 - 2.5 * Real.sqrt (2 - Real.sqrt 2) + 0.5

/-- Theorem stating the relationship between the table radius, number of mats, 
    mat width, and mat length -/
theorem place_mat_length : 
  mat_length = Real.sqrt 24.75 - 2.5 * Real.sqrt (2 - Real.sqrt 2) + 0.5 := by
  -- The proof is omitted
  sorry

/-- Theorem to show that the calculated length is approximately 3.68 -/
theorem place_mat_length_approx :
  ∃ ε > 0, abs (mat_length - 3.68) < ε := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_place_mat_length_approx_l867_86777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_extended_l867_86750

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

-- State the theorem
theorem odd_decreasing_extended (h_odd : is_odd f) 
  (h_dec : is_decreasing_on f (Set.Ici 0)) :
  is_decreasing_on f (Set.Iic 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_extended_l867_86750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_equals_neg_sqrt_three_l867_86730

theorem tan_sum_equals_neg_sqrt_three (α β : ℝ) 
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : Real.sin α + Real.sin β = Real.sqrt 3 * (Real.cos α + Real.cos β)) :
  Real.tan (α + β) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_equals_neg_sqrt_three_l867_86730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_median_l867_86767

/-- Represents the sequence where each integer n from 1 to 250 appears n times -/
def our_sequence : List ℕ := sorry

/-- The total number of elements in the sequence -/
def total_elements : ℕ := (250 * 251) / 2

/-- The median of the sequence -/
def median (seq : List ℕ) : ℚ := sorry

/-- Theorem stating that the median of our sequence is 177 -/
theorem sequence_median : median our_sequence = 177 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_median_l867_86767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l867_86757

def U : Set ℕ := {x | x ∈ Finset.range 9 ∧ x ≠ 0}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

theorem set_operations :
  (U \ A = {4, 5, 6, 7, 8}) ∧
  (U \ B = {1, 2, 7, 8}) ∧
  (B ∩ (U \ A) = {4, 5, 6}) ∧
  (A ∪ (U \ B) = {1, 2, 3, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l867_86757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_purchase_equation_l867_86731

theorem ticket_purchase_equation : 
  ∀ (x y : ℕ),
  (x + y = 38 ∧ 26 * x + 20 * y = 952) ↔ 
  (∃ (a b : ℕ), 
    a = x ∧ 
    b = y ∧ 
    a + b = 38 ∧ 
    26 * a + 20 * b = 952) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_purchase_equation_l867_86731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadratic_at_five_l867_86774

/-- A quadratic function with a minimum value of 10 at x = -2 and passing through (0, 16) -/
noncomputable def special_quadratic (x : ℝ) : ℝ :=
  let a : ℝ := 3/2  -- Derived from the conditions
  let b : ℝ := -6   -- Derived from the conditions
  let c : ℝ := 16   -- Given by the point (0, 16)
  a * x^2 + b * x + c

/-- Theorem stating the value of the special quadratic function at x = 5 -/
theorem special_quadratic_at_five :
  special_quadratic 5 = 83.5 := by
  sorry

/-- Lemma: The minimum value of the special quadratic function occurs at x = -2 -/
lemma min_at_negative_two :
  ∀ x : ℝ, special_quadratic (-2) ≤ special_quadratic x := by
  sorry

/-- Lemma: The minimum value of the special quadratic function is 10 -/
lemma min_value_is_ten :
  special_quadratic (-2) = 10 := by
  sorry

/-- Lemma: The special quadratic function passes through the point (0, 16) -/
lemma passes_through_zero_sixteen :
  special_quadratic 0 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadratic_at_five_l867_86774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l867_86716

-- Define the set A
def A : Set ℝ := {x : ℝ | (1 - x) * (x + 2) ≤ 0}

-- State the theorem
theorem complement_of_A : 
  (Set.univ \ A) = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l867_86716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l867_86783

-- Define a regular hexadecagon
structure RegularHexadecagon where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the center square within the hexadecagon
noncomputable def center_square (h : RegularHexadecagon) : ℝ :=
  (h.side_length * Real.cos (22.5 * Real.pi / 180)) ^ 2

-- Define the total area of the hexadecagon
noncomputable def total_area (h : RegularHexadecagon) : ℝ :=
  9 * (h.side_length * Real.cos (22.5 * Real.pi / 180)) ^ 2

-- Theorem: The probability of a dart landing in the center square is 1/9
theorem dart_probability (h : RegularHexadecagon) :
  center_square h / total_area h = 1 / 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l867_86783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quadrilaterals_l867_86722

/-- The number of non-similar plane quadrilaterals ABCD with given side lengths and right angle -/
noncomputable def num_quadrilaterals (a b : ℝ) : ℕ :=
  if a > b * Real.sqrt 3 then 0
  else if a = b * Real.sqrt 3 then 1
  else 2

/-- Theorem stating the number of non-similar plane quadrilaterals ABCD
    with AB = a, BC = CD = DA = b, and ∠B = 90° -/
theorem count_quadrilaterals (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  num_quadrilaterals a b =
    if a > b * Real.sqrt 3 then 0
    else if a = b * Real.sqrt 3 then 1
    else 2 := by
  -- Unfold the definition of num_quadrilaterals
  unfold num_quadrilaterals
  -- The definition matches the right-hand side exactly, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quadrilaterals_l867_86722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l867_86765

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : -π/2 < φ ∧ φ < π/2) 
  (h2 : ∀ x, f φ (3*π/8 - x) = f φ (3*π/8 + x)) :
  (∀ x, f φ (x + π/8) = -f φ (-x + π/8)) ∧
  (∀ x₁ x₂, |f φ x₁ - f φ x₂| = 2 → |x₁ - x₂| ≥ π/2) ∧
  (∃ M, M = (4/9) * Real.sqrt 3 ∧ 
   ∀ x, f φ (x - 3*π/8) * (-Real.cos x) ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l867_86765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l867_86747

/-- Proves that a train crossing a bridge has a speed of approximately 36 km/h given specific conditions -/
theorem train_speed_proof (train_length bridge_length time_to_cross : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 180)
  (h3 : time_to_cross = 27.997760179185665) : 
  ∃ (speed : ℝ), (abs (speed - 36) < 0.1) ∧ speed = (train_length + bridge_length) / time_to_cross * 3.6 := by
  sorry

#check train_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_proof_l867_86747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l867_86798

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (Real.cos C) / (Real.cos B) = (2 * a - c) / b →
  Real.tan (A + π/4) = 7 →
  B = π/3 ∧ Real.cos C = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l867_86798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_correct_l867_86711

/-- The smallest possible value of b satisfying the given conditions -/
noncomputable def smallest_b : ℝ := (5 + Real.sqrt 21) / 2

theorem smallest_b_is_correct (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2)) →
  (¬ (1/b + 1/a > 2)) →
  b ≥ smallest_b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_correct_l867_86711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l867_86771

/-- Given two lines l₁ and l₂ in the real plane, prove that if they are perpendicular
    and have the equations ax + 2y + 6 = 0 and x + (a-1)y + a² - 1 = 0 respectively,
    then a = 2/3. -/
theorem perpendicular_lines_a_value (a : ℝ) :
  let l₁ := λ (x y : ℝ) ↦ a * x + 2 * y + 6 = 0
  let l₂ := λ (x y : ℝ) ↦ x + (a - 1) * y + a^2 - 1 = 0
  (∀ x y z w, l₁ x y → l₂ z w → (x - z) * (x - z) + (y - w) * (y - w) ≠ 0 →
    (a * 1 + 2 * (a - 1)) * (1 * 1 + (a - 1) * (a - 1)) = -1) →
  a = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l867_86771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_points_distance_l867_86727

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The sphere with radius 4 centered at the origin -/
def Sphere : Set Point3D :=
  {p : Point3D | p.x^2 + p.y^2 + p.z^2 = 16}

/-- Euclidean distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Color assignment function -/
def color : Point3D → Fin 4 := sorry

theorem colored_points_distance :
  ∃ p q : Point3D, p ∈ Sphere ∧ q ∈ Sphere ∧ p ≠ q ∧ color p = color q ∧
    (distance p q = 4 * Real.sqrt 3 ∨ distance p q = 2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_points_distance_l867_86727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_permutations_l867_86793

theorem four_digit_permutations (digits : Finset ℕ) :
  digits.card = 5 →
  (∀ d ∈ digits, 1 ≤ d ∧ d ≤ 5) →
  (Finset.filter (λ n : ℕ ↦ 1000 ≤ n ∧ n < 10000 ∧
    (∀ d ∈ digits, (n.digits 10).count d ≤ 1)) (Finset.range 10000)).card = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_permutations_l867_86793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_ellipse_l867_86702

theorem equilateral_triangle_in_ellipse :
  ∀ (p q : ℕ) (triangle : Set (ℝ × ℝ)),
    -- The triangle is equilateral
    (∀ (a b : ℝ × ℝ), a ∈ triangle → b ∈ triangle → a ≠ b → 
      ∃ (side_length : ℝ), (a.1 - b.1)^2 + (a.2 - b.2)^2 = side_length^2) →
    -- The triangle is inscribed in the ellipse x^2 + 9y^2 = 9
    (∀ (point : ℝ × ℝ), point ∈ triangle → point.1^2 + 9 * point.2^2 = 9) →
    -- One vertex is at (0,1)
    ((0, 1) ∈ triangle) →
    -- One altitude is contained in the y-axis
    (∃ (a b : ℝ × ℝ), a ∈ triangle ∧ b ∈ triangle ∧ a ≠ b ∧ a.1 = 0 ∧ b.1 = 0) →
    -- The square of the length of each side is p/q
    (∃ (side_length : ℝ), ∀ (a b : ℝ × ℝ), a ∈ triangle → b ∈ triangle → a ≠ b → 
      (a.1 - b.1)^2 + (a.2 - b.2)^2 = side_length^2 ∧ side_length^2 = p / q) →
    -- p and q are relatively prime positive integers
    (p > 0 ∧ q > 0 ∧ Nat.Coprime p q) →
    p + q = 292 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_ellipse_l867_86702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_five_twentysevenths_l867_86753

/-- Represents a regular tetrahedron with associated spheres -/
structure TetrahedronWithSpheres where
  /-- Side length of the regular tetrahedron -/
  a : ℝ

/-- Radius of the inscribed sphere -/
noncomputable def R₁ (t : TetrahedronWithSpheres) : ℝ := (Real.sqrt 6 / 12) * t.a

/-- Radius of the circumscribed sphere -/
noncomputable def R₂ (t : TetrahedronWithSpheres) : ℝ := (Real.sqrt 6 / 4) * t.a

/-- Radius of the smaller spheres on each face -/
noncomputable def r (t : TetrahedronWithSpheres) : ℝ := (Real.sqrt 6 / 12) * t.a

/-- The probability that a random point within the circumscribed sphere
    of a regular tetrahedron lies within one of the five smaller spheres
    (including the inscribed sphere) -/
noncomputable def probabilityInSmallerSpheres (t : TetrahedronWithSpheres) : ℝ :=
  (5 * (r t) ^ 3) / (R₂ t) ^ 3

/-- Theorem stating that the probability is equal to 5/27 -/
theorem probability_is_five_twentysevenths (t : TetrahedronWithSpheres) :
    probabilityInSmallerSpheres t = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_five_twentysevenths_l867_86753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_relationships_l867_86734

structure Plane where

structure Line where

def intersect (l1 l2 : Line) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

def parallel_plane_line (p : Plane) (l : Line) : Prop := sorry

def parallel_planes (p1 p2 : Plane) : Prop := sorry

def perpendicular_lines (l1 l2 : Line) : Prop := sorry

def perpendicular_plane_line (p : Plane) (l : Line) : Prop := sorry

def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

def line_in_plane (l : Line) (p : Plane) : Prop := sorry

def plane_intersection (p1 p2 : Plane) : Line := sorry

theorem plane_relationships (α β : Plane) (h_not_coincident : α ≠ β) : 
  (∃ (incorrect : Fin 4), 
    (incorrect.val = 0 ↔ 
      ¬(∀ (l1 l2 l3 l4 : Line), 
        line_in_plane l1 α ∧ line_in_plane l2 α ∧ intersect l1 l2 ∧
        line_in_plane l3 β ∧ line_in_plane l4 β ∧
        parallel_lines l1 l3 ∧ parallel_lines l2 l4 →
        parallel_planes α β)) ∧
    (incorrect.val = 1 ↔ 
      ¬(∀ (l l_in_α : Line),
        ¬line_in_plane l α ∧ line_in_plane l_in_α α ∧ parallel_lines l l_in_α →
        parallel_plane_line α l)) ∧
    (incorrect.val = 2 ↔ 
      ¬(∀ (l l_in_α : Line),
        l = plane_intersection α β ∧ line_in_plane l_in_α α ∧ perpendicular_lines l l_in_α →
        perpendicular_planes α β)) ∧
    (incorrect.val = 3 ↔ 
      ¬(∀ (l l1 l2 : Line),
        line_in_plane l1 α ∧ line_in_plane l2 α ∧
        perpendicular_lines l l1 ∧ perpendicular_lines l l2 →
        perpendicular_plane_line α l)) ∧
    (Finset.filter (λ i : Fin 4 => incorrect = i) Finset.univ).card = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_relationships_l867_86734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_area_equality_l867_86786

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- length of one leg
  b : ℝ  -- length of the other leg
  c : ℝ  -- length of the hypotenuse
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

-- Define the areas of squares on the sides
noncomputable def square_area_a (t : RightTriangle) : ℝ := t.a^2
noncomputable def square_area_b (t : RightTriangle) : ℝ := t.b^2
noncomputable def square_area_c (t : RightTriangle) : ℝ := t.c^2

-- Define the areas of rectangles formed by the altitude
noncomputable def rectangle_area_a (t : RightTriangle) : ℝ := (t.c^2) / 2
noncomputable def rectangle_area_b (t : RightTriangle) : ℝ := (t.c^2) / 2

-- Theorem statement
theorem rectangle_square_area_equality (t : RightTriangle) :
  rectangle_area_a t = square_area_a t ∧ rectangle_area_b t = square_area_b t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_area_equality_l867_86786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_equal_l867_86760

theorem at_least_three_equal (a b c d : ℕ) 
  (h : ∀ (x y z w : ℕ), ({x, y, z, w} : Finset ℕ) = {a, b, c, d} → (x + y)^2 % (z * w) = 0) : 
  ∃ (n : ℕ), (({a, b, c, d} : Finset ℕ).filter (λ x => x = n)).card ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_equal_l867_86760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_monatomic_gas_work_l867_86718

/-- Represents the work done by an ideal monatomic gas in different processes -/
def gas_work (n : ℝ) (R : ℝ) (W_isobaric : ℝ) (W_isothermal : ℝ) : Prop :=
  n > 0 ∧ R > 0 ∧ W_isobaric > 0 →
  let Q_isobaric := W_isobaric + (3/2) * n * R * (W_isobaric / (n * R))
  W_isothermal = Q_isobaric

theorem ideal_monatomic_gas_work :
  gas_work 1 8.314 30 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_monatomic_gas_work_l867_86718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_pot_pie_pieces_l867_86769

/-- Given information about pies sold by Chef Michel, prove the number of pieces each chicken pot pie was cut into. -/
theorem chicken_pot_pie_pieces
  (shepherds_pie_pieces : ℕ)
  (shepherds_pie_customers : ℕ)
  (chicken_pot_pie_customers : ℕ)
  (total_pies : ℕ) :
  shepherds_pie_pieces = 4 →
  shepherds_pie_customers = 52 →
  chicken_pot_pie_customers = 80 →
  total_pies = 29 →
  ∃ (chicken_pot_pie_pieces : ℕ),
    chicken_pot_pie_pieces = 5 ∧
    chicken_pot_pie_pieces * (total_pies - (shepherds_pie_customers / shepherds_pie_pieces)) = chicken_pot_pie_customers :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_pot_pie_pieces_l867_86769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_are_bounded_l867_86725

-- Definition of a bounded function
def IsBounded (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x, |f x| ≤ M

-- Function B
noncomputable def f (x : ℝ) : ℝ := 5 / (2 * x^2 - 4 * x + 3)

-- Function C
noncomputable def g (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Theorem stating that both functions are bounded
theorem f_and_g_are_bounded : IsBounded f ∧ IsBounded g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_are_bounded_l867_86725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l867_86759

/-- Calculates the time (in seconds) for a train to cross an overbridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Proves that a train with given specifications takes 70 seconds to cross an overbridge -/
theorem train_crossing_theorem :
  train_crossing_time 600 100 36 = 70 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l867_86759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l867_86745

theorem sin_pi_plus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin (π / 2 + α) = 3 / 5) : Real.sin (π + α) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l867_86745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_implies_k_range_l867_86770

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - log x

-- Define the property of f being not monotonic in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x)

-- State the theorem
theorem f_not_monotonic_implies_k_range (k : ℝ) :
  (∀ x ∈ Set.Ioo (k - 1) (k + 1), x > 0) →
  not_monotonic f (k - 1) (k + 1) →
  1 ≤ k ∧ k < 3/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_implies_k_range_l867_86770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_l867_86705

/-- A geometric sequence of positive integers with first term 3 and seventh term 2187 -/
def GeometricSequence : ℕ → ℕ := sorry

/-- The first term of the sequence is 3 -/
axiom first_term : GeometricSequence 1 = 3

/-- The seventh term of the sequence is 2187 -/
axiom seventh_term : GeometricSequence 7 = 2187

/-- The sequence is geometric -/
axiom is_geometric : ∀ n : ℕ, n > 0 → ∃ r : ℚ, GeometricSequence (n + 1) = (GeometricSequence n) * r

/-- The eighth term of the sequence is 6561 -/
theorem eighth_term : GeometricSequence 8 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_l867_86705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_line_slope_l867_86741

/-- Given an ellipse and a parabola with specific properties, prove that the slope of a certain line is ±√6/2 -/
theorem ellipse_parabola_line_slope :
  ∀ (a b c p : ℝ) (F A P Q B D : ℝ × ℝ),
    a > b ∧ b > 0 ∧ p > 0 ∧  -- Positive parameters
    c / a = 1 / 2 ∧  -- Eccentricity
    a - c = 1 / 2 ∧  -- A is focus of parabola
    p / 2 = 1 / 2 ∧  -- Distance from F to directrix
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ Set.range (λ t ↦ (a * Real.cos t, b * Real.sin t))) ∧  -- Ellipse equation
    (∀ (x y : ℝ), y^2 = 2 * p * x ↔ (x, y) ∈ Set.range (λ t ↦ (t^2 / (2 * p), t))) ∧  -- Parabola equation
    F.1 = -c ∧ F.2 = 0 ∧  -- Left focus
    A.1 = a ∧ A.2 = 0 ∧  -- Right vertex
    P.1 = -1 ∧ Q.1 = -1 ∧ P.2 = -Q.2 ∧  -- P and Q symmetric about x-axis
    (B.1 - A.1) * (B.2 - A.2) = (P.1 - A.1) * (P.2 - A.2) ∧  -- AP and AB collinear
    B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧  -- B on ellipse
    (D.1 - B.1) * (D.2 - B.2) = (Q.1 - B.1) * (Q.2 - B.2) ∧  -- BQ and BD collinear
    D.2 = 0 ∧  -- D on x-axis
    1/2 * (A.1 - D.1) * (2 / ((P.2 - A.2) / (P.1 - A.1))) = Real.sqrt 6 / 2  -- Area of triangle APD
    →
    ((P.2 - A.2) / (P.1 - A.1))^2 = 6 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_line_slope_l867_86741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_xi_one_greatest_l867_86706

noncomputable def probability_hit_A : ℝ := 1/2

noncomputable def probability_hit_BC (a : ℝ) : ℝ := a

noncomputable def P_xi (a : ℝ) (i : ℕ) : ℝ :=
  match i with
  | 0 => (1 - probability_hit_A) * (1 - probability_hit_BC a)^2
  | 1 => probability_hit_A * (1 - probability_hit_BC a)^2 + 
         (1 - probability_hit_A) * 2 * (probability_hit_BC a) * (1 - probability_hit_BC a)
  | 2 => probability_hit_A * 2 * (probability_hit_BC a) * (1 - probability_hit_BC a) + 
         (1 - probability_hit_A) * (probability_hit_BC a)^2
  | 3 => probability_hit_A * (probability_hit_BC a)^2
  | _ => 0

theorem prob_xi_one_greatest (a : ℝ) :
  0 < a ∧ a < 1 →
  (∀ i : ℕ, P_xi a 1 ≥ P_xi a i) ↔ 0 < a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_xi_one_greatest_l867_86706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_has_local_symmetry_point_exponential_function_local_symmetry_condition_l867_86758

-- Define the concept of a local symmetry point
def is_local_symmetry_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (-x₀) = -f x₀

-- Statement for the first part of the problem
theorem cubic_function_has_local_symmetry_point (a b c : ℝ) :
  ∃ x₀ : ℝ, is_local_symmetry_point (fun x ↦ a * x^3 + b * x^2 + c * x - b) x₀ := by
  sorry

-- Statement for the second part of the problem
theorem exponential_function_local_symmetry_condition :
  ∃ m : ℝ, (∃ x₀ : ℝ, is_local_symmetry_point (fun x ↦ 4^x - m * 2^(x+1) + m^2 - 3) x₀) ↔ 
  -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_has_local_symmetry_point_exponential_function_local_symmetry_condition_l867_86758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investor_return_is_25_percent_l867_86700

-- Define the given values
noncomputable def dividend_rate : ℚ := 185/10
noncomputable def face_value : ℚ := 50
noncomputable def purchase_price : ℚ := 37

-- Define the function to calculate percentage return on investment
noncomputable def percentage_return_on_investment (dividend_rate face_value purchase_price : ℚ) : ℚ :=
  (dividend_rate / 100 * face_value) / purchase_price * 100

-- Theorem statement
theorem investor_return_is_25_percent :
  percentage_return_on_investment dividend_rate face_value purchase_price = 25 := by
  -- Unfold the definition and simplify
  unfold percentage_return_on_investment
  simp [dividend_rate, face_value, purchase_price]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investor_return_is_25_percent_l867_86700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_value_l867_86726

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  -- Base lengths
  AB : ℝ
  CD : ℝ
  -- Height between bases
  height : ℝ
  -- Conditions
  AB_twice_CD : AB = 2 * CD
  CD_length : CD = 6
  height_value : height = 5

/-- Calculate the perimeter of the isosceles trapezoid -/
noncomputable def perimeter (t : IsoscelesTrapezoid) : ℝ :=
  t.AB + t.CD + 2 * Real.sqrt (((t.AB - t.CD) / 2) ^ 2 + t.height ^ 2)

/-- Theorem stating that the perimeter of the specific isosceles trapezoid is 18 + 2√34 -/
theorem perimeter_value (t : IsoscelesTrapezoid) : perimeter t = 18 + 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_value_l867_86726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_absolute_value_l867_86754

theorem integral_absolute_value :
  (∫ x in (-1 : ℝ)..1, |x^2 - x|) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_absolute_value_l867_86754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_razorback_shop_revenue_difference_l867_86714

/-- The revenue difference between t-shirts and jerseys sold at a Razorback shop event -/
theorem razorback_shop_revenue_difference :
  let jersey_revenue : ℕ := 210
  let tshirt_revenue : ℕ := 240
  let jerseys_sold : ℕ := 23
  let tshirts_sold : ℕ := 177
  (tshirt_revenue * tshirts_sold) - (jersey_revenue * jerseys_sold) = 37650 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_razorback_shop_revenue_difference_l867_86714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l867_86799

theorem no_real_solutions : ¬∃ x : ℝ, (3*x - 4)^2 + 3 = -2*|x-1| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l867_86799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l867_86715

theorem complex_number_properties (z : ℂ) (h : z * (1 - 3*Complex.I) = 10) : 
  Complex.abs z = Real.sqrt 10 ∧ z - 3 * (Complex.exp (Complex.I * Real.pi / 4))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l867_86715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l867_86743

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.S = t.b * t.c * Real.cos t.A)
  (h2 : t.C = Real.pi / 4) :
  (Real.cos t.B = Real.sqrt 10 / 10) ∧
  (t.c = Real.sqrt 5 → t.S = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l867_86743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_square_inequality_l867_86749

noncomputable def f (x : ℝ) : ℝ := 2^x * x - x / 2^x

theorem f_inequality_implies_square_inequality (m n : ℝ) :
  f m < f n → m^2 < n^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_square_inequality_l867_86749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_equals_two_sqrt_two_l867_86794

theorem power_difference_equals_two_sqrt_two
  (a b : ℝ)
  (ha : a > 1)
  (hb : b > 0)
  (h_sum : a^b + a^(-b) = 2 * Real.sqrt 3) :
  a^b - a^(-b) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_equals_two_sqrt_two_l867_86794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_test_scores_l867_86746

theorem johns_test_scores :
  let first_three : List Nat := [74, 65, 79]
  let all_five : List Nat := [92, 79, 75, 74, 65]
  (∀ x, x ∈ all_five → x < 95) ∧
  (∀ x y, x ∈ all_five → y ∈ all_five → x ≠ y → x ≠ y) ∧
  (List.sum all_five = 5 * 77) ∧
  (List.take 3 all_five = first_three) ∧
  (List.length all_five = 5) →
  all_five = [92, 79, 75, 74, 65] :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_test_scores_l867_86746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_amount_l867_86739

/-- The amount of money (in Rs.) that A lends to B -/
def principal : ℝ := sorry

/-- The interest rate (as a decimal) at which A lends to B -/
def rate_A_to_B : ℝ := 0.10

/-- The interest rate (as a decimal) at which B lends to C -/
def rate_B_to_C : ℝ := 0.115

/-- The time period in years -/
def time : ℝ := 3

/-- B's gain (in Rs.) over the time period -/
def gain : ℝ := 157.5

theorem loan_amount : 
  principal * rate_B_to_C * time - principal * rate_A_to_B * time = gain →
  principal = 3500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_amount_l867_86739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_theorem_l867_86712

-- Define a type for regions (representing mobots and their mowed areas)
def Region : Type := ℕ × ℕ

-- Define a distance function between regions
noncomputable def distance (r1 r2 : Region) : ℝ :=
  Real.sqrt (((r1.1 : ℝ) - (r2.1 : ℝ))^2 + ((r1.2 : ℝ) - (r2.2 : ℝ))^2)

-- Define adjacency of regions
def adjacent (r1 r2 : Region) : Prop :=
  distance r1 r2 = 1

-- Define a color type
inductive Color
| white
| black
| blue

-- State the theorem
theorem three_color_theorem (lawn : Set Region) :
  ∃ (coloring : Region → Color),
    ∀ (r1 r2 : Region),
      r1 ∈ lawn → r2 ∈ lawn → adjacent r1 r2 → coloring r1 ≠ coloring r2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_theorem_l867_86712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l867_86710

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c < 0}) :
  {x : ℝ | b * x^2 + a * x - c ≤ 0} = Set.Icc (-1 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l867_86710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trigonometric_expression_l867_86742

theorem max_value_trigonometric_expression :
  ∀ x y z : ℝ,
  (Real.sin x + Real.sin (2 * y) + Real.sin (3 * z)) * (Real.cos x + Real.cos (2 * y) + Real.cos (3 * z)) ≤ 4.5 ∧
  ∃ a b c : ℝ,
  (Real.sin a + Real.sin (2 * b) + Real.sin (3 * c)) * (Real.cos a + Real.cos (2 * b) + Real.cos (3 * c)) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trigonometric_expression_l867_86742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_one_l867_86708

noncomputable def ArithmeticSequence (x y : ℝ) : ℕ → ℝ
  | 0 => x^2 + y^2
  | 1 => x^2 - y^2
  | 2 => x^2 * y^2
  | 3 => x^2 / y^2
  | n + 4 => ArithmeticSequence x y 3 - 2 * y^2

theorem fifth_term_is_one :
  ∀ x y : ℝ, x^2 = 3 → y^2 = 1 →
  ArithmeticSequence x y 4 = 1 := by
  sorry

#check fifth_term_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_one_l867_86708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l867_86721

/-- A right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  height : ℝ

/-- The volume of a right pyramid -/
noncomputable def volume (p : RightPyramid) : ℝ := (1/3) * p.base_side^2 * p.height

/-- The surface area of a right pyramid -/
noncomputable def surface_area (p : RightPyramid) : ℝ :=
  p.base_side^2 + 2 * p.base_side * Real.sqrt (p.height^2 + (p.base_side/2)^2)

/-- The area of a triangular face of a right pyramid -/
noncomputable def triangular_face_area (p : RightPyramid) : ℝ :=
  (1/2) * p.base_side * Real.sqrt (p.height^2 + (p.base_side/2)^2)

theorem pyramid_volume_theorem (p : RightPyramid) 
  (h1 : surface_area p = 600)
  (h2 : triangular_face_area p = (1/3) * p.base_side^2) :
  volume p = 120 * Real.sqrt 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l867_86721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_sum_l867_86720

theorem max_value_and_sum :
  ∃ (x_M y_M z_M v_M w_M : ℝ),
    (∀ x y z v w : ℝ, x > 0 → y > 0 → z > 0 → v > 0 → w > 0 →
      x^2 + y^2 + z^2 + v^2 + w^2 = 1024 →
      x*z + 3*y*z + 4*z*v + 6*z*w + 2*x*w ≤ x_M*z_M + 3*y_M*z_M + 4*z_M*v_M + 6*z_M*w_M + 2*x_M*w_M) ∧
    x_M*z_M + 3*y_M*z_M + 4*z_M*v_M + 6*z_M*w_M + 2*x_M*w_M = 262144 ∧
    x_M*z_M + 3*y_M*z_M + 4*z_M*v_M + 6*z_M*w_M + 2*x_M*w_M + x_M + y_M + z_M + v_M + w_M = 262144 + 44 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_sum_l867_86720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_range_l867_86732

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (1+a)^x

theorem monotone_f_range (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x > 0, Monotone (f a)) → a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_range_l867_86732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sons_yearly_profit_l867_86735

/-- Calculates the profit for each son after one year of cultivation --/
noncomputable def profit_per_son (total_land : ℝ) (num_sons : ℕ) (profit_per_unit : ℝ) (unit_area : ℝ) : ℝ :=
  let hectare_to_sqm : ℝ := 10000
  let months_in_year : ℕ := 12
  let son_share := total_land / (num_sons : ℝ)
  let son_share_sqm := son_share * hectare_to_sqm
  let num_units := son_share_sqm / unit_area
  let profit_per_quarter := num_units * profit_per_unit
  profit_per_quarter * ((months_in_year / 3) : ℝ)

/-- Theorem stating that each son's profit for one year is $10,000 --/
theorem sons_yearly_profit :
  profit_per_son 3 8 500 750 = 10000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sons_yearly_profit_l867_86735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_is_222_l867_86768

/-- Represents a pizza delivery with number of pizzas and distance from pizzeria -/
structure Delivery where
  pizzas : ℕ
  distance : ℚ

def pizza_cost : ℚ := 12
def delivery_charge : ℚ := 2
def delivery_charge_threshold : ℚ := 1

def calculate_cost (d : Delivery) : ℚ :=
  d.pizzas * pizza_cost + if d.distance > delivery_charge_threshold then delivery_charge else 0

def deliveries : List Delivery := [
  ⟨3, 1/10⟩,
  ⟨2, 2⟩,
  ⟨4, 4/5⟩,
  ⟨5, 3/2⟩,
  ⟨1, 3/10⟩,
  ⟨3, 6/5⟩
]

theorem total_payment_is_222 :
  (deliveries.map calculate_cost).sum = 222 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_is_222_l867_86768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_g_properties_l867_86796

open Real

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 - Real.sin x ^ 4 + Real.sin (2 * x - π / 6)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x - φ)

theorem f_monotone_and_g_properties :
  (∃ (a b : ℝ), a = 0 ∧ b = π / 6 ∧ 
    (∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, x ≤ y → f x ≤ f y) ∧
    (∀ x ∈ Set.Icc 0 (π / 2), x < a ∨ x > b → ¬(∀ y ∈ Set.Icc 0 (π / 2), x ≤ y → f x ≤ f y))) ∧
  (∃ (φ : ℝ), 0 < φ ∧ φ < π / 4 ∧
    (∀ x : ℝ, g φ (π / 3 + x) = g φ (π / 3 - x)) ∧
    (∃ (α : ℝ), π / 12 ≤ α ∧ α ≤ 5 * π / 12 ∧
      Set.range (fun x => g φ x) ∩ Set.Icc (-π / 4) α = Set.Icc (-1 / 2) 1) ∧
    (∀ β : ℝ, β < π / 12 ∨ β > 5 * π / 12 →
      Set.range (fun x => g φ x) ∩ Set.Icc (-π / 4) β ≠ Set.Icc (-1 / 2) 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_g_properties_l867_86796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_catches_mary_l867_86737

/-- The time it takes for Paul to catch up with Mary after passing the gas station -/
noncomputable def catchUpTime (marySpeed paulSpeed : ℝ) (headStart : ℝ) : ℝ :=
  let distanceAhead := marySpeed * (headStart / 60)
  let catchUpSpeed := paulSpeed - marySpeed
  (distanceAhead / catchUpSpeed) * 60

theorem paul_catches_mary :
  let marySpeed : ℝ := 50
  let paulSpeed : ℝ := 80
  let headStart : ℝ := 15
  catchUpTime marySpeed paulSpeed headStart = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_catches_mary_l867_86737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_approx_l867_86703

/-- Calculates the depth of a cylindrical well given its diameter, digging cost per cubic meter, and total cost. -/
noncomputable def well_depth (diameter : ℝ) (cost_per_cubic_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := total_cost / cost_per_cubic_meter
  volume / (Real.pi * radius ^ 2)

/-- Theorem stating that the depth of the well with given parameters is approximately 14.01 meters. -/
theorem well_depth_approx :
  let d := 3  -- diameter in meters
  let c := 18 -- cost per cubic meter in Rs.
  let t := 1781.28 -- total cost in Rs.
  abs (well_depth d c t - 14.01) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_approx_l867_86703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_document_delivery_speed_relation_l867_86724

/-- The specified time for document delivery in days -/
def x : ℝ := sorry

/-- The distance of the document delivery in miles -/
def distance : ℝ := 900

/-- The time taken by the slow horse in days -/
def slow_horse_time (x : ℝ) : ℝ := x + 1

/-- The time taken by the fast horse in days -/
def fast_horse_time (x : ℝ) : ℝ := x - 3

/-- The theorem stating the relationship between the speeds of the fast and slow horses -/
theorem document_delivery_speed_relation (x : ℝ) :
  distance / fast_horse_time x = 2 * (distance / slow_horse_time x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_document_delivery_speed_relation_l867_86724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l867_86707

noncomputable def v1 : ℝ × ℝ := (3, -2)
noncomputable def v2 : ℝ × ℝ := (2, 5)
noncomputable def q : ℝ × ℝ := (133/50, 39/50)

def perpendicular (v : ℝ × ℝ) : ℝ × ℝ := (-v.2, v.1)

theorem projection_equality (u : ℝ × ℝ) :
  (∃ (k1 k2 : ℝ), q = k1 • u ∧ v1 - q = k2 • (perpendicular u)) ∧
  (∃ (k3 k4 : ℝ), q = k3 • u ∧ v2 - q = k4 • (perpendicular u)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l867_86707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_solutions_l867_86728

theorem trapezoid_bases_solutions :
  ∃ (S : List (ℕ × ℕ)), 
    (∀ (b₁ b₂ : ℕ), (b₁, b₂) ∈ S ↔ 
      (6 ∣ b₁) ∧ 
      (6 ∣ b₂) ∧ 
      b₁ + b₂ = 60 ∧ 
      (60 * (b₁ + b₂)) / 2 = 1800) ∧
    S.length > 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bases_solutions_l867_86728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_intervals_l867_86787

-- Define the custom operation
noncomputable def customOp (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := customOp (2 * Real.sin x) (Real.cos x)

-- Define the intervals of monotonic increase
def monotonicIntervals : Set (Set ℝ) := {
  Set.Ioo 0 (Real.pi / 4),
  Set.Ioo Real.pi (5 * Real.pi / 4),
  Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)
}

-- Theorem statement
theorem f_monotonic_intervals :
  ∀ I ∈ monotonicIntervals,
    ∀ x y, x ∈ Set.Icc 0 (2 * Real.pi) → y ∈ Set.Icc 0 (2 * Real.pi) →
      x ∈ I → y ∈ I → x < y → f x < f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_intervals_l867_86787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l867_86761

theorem sine_cosine_identity (α : ℝ) : 
  Real.sin α * Real.cos (α + π/6) - Real.cos α * Real.sin (α + π/6) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l867_86761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l867_86784

def f (a b x : ℝ) := a * x^2 + b * x + 1

noncomputable def F (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f a b x else -f a b x

def g (a b k x : ℝ) := f a b x - k * x

theorem function_properties
  (a b : ℝ)
  (h1 : f a b (-1) = 0)
  (h2 : ∀ x, f a b x ≥ 0)
  (h3 : ∃ M, ∀ x, f a b x ≤ M) :
  (∀ x > 0, F a b x = x^2 + 2*x + 1) ∧
  (∀ x < 0, F a b x = -x^2 - 2*x - 1) ∧
  (∀ k, (∀ x ∈ Set.Icc (-2) 2, Monotone (fun x ↦ g a b k x)) → (k ≥ 6 ∨ k ≤ -2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l867_86784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_radii_l867_86723

/-- Circle C1 with equation x^2 + y^2 - 2ax + a^2 - 9 = 0 -/
def C1 (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 + y^2 - 2*a*x + a^2 - 9 = 0

/-- Circle C2 with equation x^2 + y^2 + 2by + b^2 - 1 = 0 -/
def C2 (b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 + y^2 + 2*b*y + b^2 - 1 = 0

/-- Two circles are tangent if they intersect at exactly one point -/
def are_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃! (x y : ℝ), C1 x y ∧ C2 x y

theorem max_sum_of_radii (a b : ℝ) :
  are_tangent (C1 a) (C2 b) → (a + b ≤ 2 * Real.sqrt 2) ∧ 
  (∃ a₀ b₀ : ℝ, are_tangent (C1 a₀) (C2 b₀) ∧ a₀ + b₀ = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_radii_l867_86723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exterior_angles_sum_of_variables_l867_86733

/-- Represents a convex polygon --/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- Represents an exterior angle of a convex polygon --/
def ExteriorAngle (polygon : ConvexPolygon) := ℝ

/-- The sum of all exterior angles (both sets) of a convex polygon is 720° --/
theorem sum_of_exterior_angles (polygon : ConvexPolygon) :
  ∃ (angles : List ℝ),
    angles.length > 0 ∧ angles.sum = 720 := by
  sorry

/-- The sum of variables p+q+r+s+t+u+v+w+x+y in the diagram is 720° --/
theorem sum_of_variables :
  ∃ (p q r s t u v w x y : ℝ),
    p + q + r + s + t + u + v + w + x + y = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_exterior_angles_sum_of_variables_l867_86733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_terms_geometric_sequence_problem_l867_86709

/-- Sum of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: Sum of first 8 terms of a geometric sequence -/
theorem sum_of_eight_terms (a : ℝ) :
  (geometric_sum a 2 4 = 30) →
  (geometric_sum a 2 8 = 510) := by
  sorry

/-- Main theorem connecting the problem to the solution -/
theorem geometric_sequence_problem :
  ∃ a : ℝ, (geometric_sum a 2 4 = 30) ∧ (geometric_sum a 2 8 = 510) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_terms_geometric_sequence_problem_l867_86709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_probabilities_l867_86744

/-- Represents the color of a ball -/
inductive Color
  | Yellow
  | Blue
  | Red

/-- Represents the box of balls -/
def Box := Multiset Color

/-- The box contains 3 yellow, 2 blue, and 1 red ball -/
def initialBox : Box :=
  Multiset.replicate 3 Color.Yellow +
  Multiset.replicate 2 Color.Blue +
  Multiset.replicate 1 Color.Red

/-- The number of balls to draw -/
def drawCount : Nat := 3

/-- Calculates the probability of drawing exactly two yellow balls -/
noncomputable def probExactlyTwoYellow (box : Box) (draw : Nat) : ℚ :=
  sorry

/-- Calculates the probability of drawing at least one blue ball -/
noncomputable def probAtLeastOneBlue (box : Box) (draw : Nat) : ℚ :=
  sorry

theorem ball_drawing_probabilities :
  probExactlyTwoYellow initialBox drawCount = 9/20 ∧
  probAtLeastOneBlue initialBox drawCount = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_probabilities_l867_86744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_l867_86762

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop := sorry

-- Define the length of a side
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at a vertex
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem right_triangle (t : Triangle) :
  isValidTriangle t →
  length t.A t.B = 2 * length t.B t.C →
  angle t.A t.B t.C = 2 * angle t.B t.A t.C →
  angle t.A t.C t.B = Real.pi / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_l867_86762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_product_eight_less_sum_roots_l867_86778

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem product_eight_less_sum_roots :
  let f : ℝ → ℝ := λ x => x^2 - 8*x + 7
  let r₁ := (-(-8) + Real.sqrt ((-8)^2 - 4*1*7)) / (2*1)
  let r₂ := (-(-8) - Real.sqrt ((-8)^2 - 4*1*7)) / (2*1)
  r₁ + r₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_product_eight_less_sum_roots_l867_86778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coordinate_l867_86729

theorem equidistant_point_x_coordinate :
  ∃ (x y : ℝ),
    (abs x = abs (x - 2)) ∧
    (abs x = abs (2*x + 4 - y) / Real.sqrt 5) ∧
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coordinate_l867_86729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_additional_season_l867_86785

/-- Represents the TV show seasons and episodes -/
structure TVShow where
  regularSeasonEpisodes : ℕ
  lastSeasonEpisodes : ℕ
  initialSeasons : ℕ
  episodeDuration : ℚ
  totalWatchTime : ℚ

/-- Calculates the number of additional seasons for the TV show -/
def additionalSeasons (s : TVShow) : ℕ :=
  let initialWatchTime := s.initialSeasons * s.regularSeasonEpisodes * s.episodeDuration
  let additionalWatchTime := s.totalWatchTime - initialWatchTime
  let additionalEpisodes := additionalWatchTime / s.episodeDuration
  if additionalEpisodes = s.lastSeasonEpisodes then 1 else 0

/-- Theorem stating that the number of additional seasons is 1 for the given TV show -/
theorem one_additional_season (s : TVShow)
  (h1 : s.regularSeasonEpisodes = 22)
  (h2 : s.lastSeasonEpisodes = 26)
  (h3 : s.initialSeasons = 9)
  (h4 : s.episodeDuration = 1/2)
  (h5 : s.totalWatchTime = 112) :
  additionalSeasons s = 1 := by
  sorry

#eval additionalSeasons ⟨22, 26, 9, 1/2, 112⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_additional_season_l867_86785
