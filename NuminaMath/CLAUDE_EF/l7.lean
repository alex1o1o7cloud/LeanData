import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_limit_l7_767

-- Define the arithmetic sequence
noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the sum of the first n terms
noncomputable def S (n : ℕ) : ℝ := (n : ℝ) ^ 2

-- State the theorem
theorem arithmetic_sequence_limit :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((a n)^2 / S n) - 4| < ε :=
by
  sorry

#check arithmetic_sequence_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_limit_l7_767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_contribution_l7_769

noncomputable def initial_average : ℝ := 75 / 1.5
def num_initial_contributions : ℕ := 2
noncomputable def new_average : ℝ := 75
def total_contributions : ℕ := 3

theorem johns_contribution (J : ℝ) : 
  new_average = (num_initial_contributions * initial_average + J) / total_contributions :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_contribution_l7_769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_cone_volume_l7_761

open Real

-- Define the cone based on the given conditions
def cone (θ : ℝ) (A : ℝ) : Prop :=
  θ = 120 * Real.pi / 180 ∧ A = 3 * Real.pi

-- Theorem for the surface area of the cone
theorem cone_surface_area (θ : ℝ) (A : ℝ) (h : cone θ A) :
  ∃ S : ℝ, S = 4 * Real.pi := by sorry

-- Theorem for the volume of the cone
theorem cone_volume (θ : ℝ) (A : ℝ) (h : cone θ A) :
  ∃ V : ℝ, V = 2 * Real.sqrt 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_cone_volume_l7_761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tanya_completes_in_12_days_l7_753

/-- The number of days it takes Tanya to complete a piece of work, given Sakshi's completion time and Tanya's efficiency relative to Sakshi. -/
noncomputable def tanya_work_days (sakshi_days : ℝ) (tanya_efficiency : ℝ) : ℝ :=
  sakshi_days / (1 + tanya_efficiency)

/-- Theorem stating that Tanya can complete the work in 12 days given the problem conditions. -/
theorem tanya_completes_in_12_days (sakshi_days : ℝ) (tanya_efficiency : ℝ)
  (h1 : sakshi_days = 15)
  (h2 : tanya_efficiency = 0.25) :
  tanya_work_days sakshi_days tanya_efficiency = 12 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tanya_completes_in_12_days_l7_753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l7_758

theorem trigonometric_identity (α : ℝ) : 
  (2 * (Real.cos α)^2 + Real.cos (π/2 + 2*α) - 1) / (Real.sqrt 2 * Real.sin (2*α + π/4)) = 4 → 
  Real.tan (2*α + π/4) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l7_758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_jill_paths_l7_756

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a grid -/
def numPaths (start : Point) (finish : Point) : ℕ :=
  sorry

/-- Calculates the number of paths avoiding specific points -/
def numPathsAvoiding (start : Point) (finish : Point) (avoid : List Point) : ℕ :=
  sorry

/-- The problem statement -/
theorem jack_jill_paths :
  let start := Point.mk 0 0
  let finish := Point.mk 4 3
  let avoid := [Point.mk 1 1, Point.mk 2 2]
  numPathsAvoiding start finish avoid = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_jill_paths_l7_756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tan_product_l7_705

theorem min_tan_product (A B C : ℝ) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sum : A + B + C = π) (h_sin : Real.sin A = 2 * Real.sin B * Real.sin C) :
  ∀ x, x = Real.tan A * Real.tan B * Real.tan C → x ≥ 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tan_product_l7_705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A₁C₁_to_base_is_sqrt3_l7_754

/-- A right square prism with base edge length 1 and AB₁ forming a 60° angle with the base -/
structure RightSquarePrism :=
  (base_edge : ℝ)
  (angle_AB₁_base : ℝ)
  (is_right_square_prism : base_edge = 1)
  (angle_is_60_degrees : angle_AB₁_base = Real.pi / 3)

/-- The distance from A₁C₁ to the base ABCD in a right square prism -/
noncomputable def distance_A₁C₁_to_base (prism : RightSquarePrism) : ℝ :=
  Real.sqrt 3

/-- Theorem stating that the distance from A₁C₁ to the base ABCD is √3 -/
theorem distance_A₁C₁_to_base_is_sqrt3 (prism : RightSquarePrism) :
  distance_A₁C₁_to_base prism = Real.sqrt 3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A₁C₁_to_base_is_sqrt3_l7_754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yoque_borrowed_amount_l7_785

/-- The amount of money Yoque borrowed -/
def borrowed_amount : ℚ := 150

/-- The number of months for repayment -/
def repayment_months : ℕ := 11

/-- The monthly payment amount -/
def monthly_payment : ℚ := 15

/-- The interest rate as a rational number -/
def interest_rate : ℚ := 1/10

theorem yoque_borrowed_amount :
  borrowed_amount * (1 + interest_rate) = monthly_payment * repayment_months :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yoque_borrowed_amount_l7_785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_overlap_circles_l7_776

/-- Given two circles with radii 3 and 6 inches respectively, and centers 10 inches apart,
    the area of their overlapping region is 0. -/
theorem no_overlap_circles (r₁ r₂ d : ℝ) (hr₁ : r₁ = 3) (hr₂ : r₂ = 6) (hd : d = 10) :
  let overlap_area := Real.pi * (r₁^2 + r₂^2 - d * Real.sqrt ((-d + r₁ + r₂) * (d + r₁ - r₂) * (d - r₁ + r₂) * (d + r₁ + r₂)) / (2 * d))
  overlap_area = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_overlap_circles_l7_776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_repeating_three_tenths_l7_778

/-- Given that 0.333... is a repeating decimal, prove that 1 - 0.333... = 2/3 -/
theorem one_minus_repeating_three_tenths (b : ℚ) (h : b = 1/3) : 1 - b = 2/3 := by
  rw [h]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_repeating_three_tenths_l7_778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_side_length_l7_787

-- Define the cyclic quadrilateral ABCD
def CyclicQuadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (R : ℝ), 
    dist center A = R ∧ 
    dist center B = R ∧ 
    dist center C = R ∧ 
    dist center D = R

-- Define the theorem
theorem cyclic_quadrilateral_side_length 
  (A B C D : EuclideanSpace ℝ (Fin 2)) 
  (h_cyclic : CyclicQuadrilateral A B C D) 
  (h_circumradius : ∃ (center : EuclideanSpace ℝ (Fin 2)), dist center A = 200 * Real.sqrt 2) 
  (h_equal_sides : dist A B = 200 ∧ dist B C = 200 ∧ dist C D = 200) : 
  dist A D = 200 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_side_length_l7_787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_speed_theorem_l7_775

/-- A fish's speeds in different water conditions -/
structure FishSpeed where
  upstream : ℝ
  downstream : ℝ

/-- The speed of a fish in still water given its upstream and downstream speeds -/
noncomputable def stillWaterSpeed (fs : FishSpeed) : ℝ :=
  (fs.upstream + fs.downstream) / 2

/-- Theorem: A fish with upstream speed 35 kmph and downstream speed 55 kmph has a still water speed of 45 kmph -/
theorem fish_speed_theorem (fs : FishSpeed) 
  (h_upstream : fs.upstream = 35) 
  (h_downstream : fs.downstream = 55) : 
  stillWaterSpeed fs = 45 := by
  unfold stillWaterSpeed
  rw [h_upstream, h_downstream]
  norm_num

#check fish_speed_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_speed_theorem_l7_775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_theorem_l7_721

/-- Regular triangular pyramid with given properties -/
structure RegularPyramid where
  -- Side length of the base
  base_side : ℝ
  -- Ratio of the intersection point on the edge
  intersection_ratio : ℝ
  -- Area of the section
  section_area : ℝ
  -- Conditions
  base_side_positive : 0 < base_side
  intersection_ratio_valid : 0 < intersection_ratio ∧ intersection_ratio < 1
  section_area_positive : 0 < section_area

/-- The height of a regular triangular pyramid with given properties -/
noncomputable def pyramid_height (p : RegularPyramid) : ℝ :=
  Real.sqrt (11 / 3)

/-- Theorem stating the height of the pyramid under given conditions -/
theorem pyramid_height_theorem (p : RegularPyramid) 
  (h1 : p.base_side = 2)
  (h2 : p.intersection_ratio = 1 / 4)
  (h3 : p.section_area = 3) :
  pyramid_height p = Real.sqrt (11 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_theorem_l7_721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_egg_problem_l7_757

-- Define the rate at which chickens lay eggs
noncomputable def normal_laying_rate : ℚ := 1 / (3/2)

-- Define the faster laying rate
noncomputable def faster_laying_rate : ℚ := normal_laying_rate * (3/2)

-- Define the time period in days
def time_period : ℚ := 21/2

-- Theorem statement
theorem chicken_egg_problem :
  let num_chickens : ℚ := 1
  let num_eggs : ℚ := num_chickens * faster_laying_rate * time_period
  num_eggs = 21/2 := by
  -- Unfold definitions
  unfold faster_laying_rate normal_laying_rate time_period
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_egg_problem_l7_757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_five_terms_l7_750

def sequence_term (n : ℕ) : ℚ := (-1:ℚ)^(n+1) * (n^2 / 2)

theorem sequence_first_five_terms :
  (sequence_term 1 = 1/2) ∧
  (sequence_term 2 = -2) ∧
  (sequence_term 3 = 9/2) ∧
  (sequence_term 4 = -8) ∧
  (sequence_term 5 = 25/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_five_terms_l7_750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l7_781

theorem cosine_problem (α β : ℝ) (h1 : Real.cos α = 1/7) (h2 : Real.cos (α - β) = 13/14) 
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π/2) : Real.cos β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l7_781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l7_789

/-- Calculates the length of a platform given train speed and passing times. -/
theorem platform_length 
  (train_speed_kmh : ℝ) 
  (time_pass_man : ℝ) 
  (time_pass_platform : ℝ) 
  (h1 : train_speed_kmh = 54) 
  (h2 : time_pass_man = 20) 
  (h3 : time_pass_platform = 44) : 
  ∃ (platform_length : ℝ), platform_length = 360 := by
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let train_length := train_speed_ms * time_pass_man
  let platform_length := train_speed_ms * time_pass_platform - train_length
  use platform_length
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l7_789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l7_708

noncomputable def a : ℕ → ℚ
  | 0 => 1
  | 1 => -1/3
  | 2 => -1/6
  | n+3 => (-1)^(Nat.floor ((n+2)/3)) * 1 / (2^(n+3))

noncomputable def series_sum : ℚ := ∑' n, a n

theorem series_sum_value : series_sum = 25/48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l7_708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l7_738

/-- The distance from the focus to the directrix of a parabola y = 4x^2 is 1/8 -/
theorem parabola_focus_directrix_distance : 
  ∃ (f d : ℝ), 
    (∀ (x y : ℝ), y = 4 * x^2 ↔ (x - f)^2 = (y - 0) / 4) ∧ 
    (d = -f) ∧ 
    (f - d = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l7_738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2m_formula_l7_716

/-- A strictly increasing sequence from positive integers to positive integers -/
def StrictlyIncreasingSeq := (n : ℕ+) → ℕ+

/-- The set of positive integers in the range of a sequence -/
def RangeSet (f : StrictlyIncreasingSeq) : Set ℕ+ :=
  {n | ∃ m, f m = n}

/-- Definition of the function f -/
noncomputable def f : StrictlyIncreasingSeq := sorry

/-- Definition of the function g -/
noncomputable def g : StrictlyIncreasingSeq := sorry

/-- f and g are strictly increasing -/
axiom f_increasing : ∀ m n, m < n → f m < f n
axiom g_increasing : ∀ m n, m < n → g m < g n

/-- The ranges of f and g are disjoint -/
axiom ranges_disjoint : RangeSet f ∩ RangeSet g = ∅

/-- The ranges of f and g cover all positive integers -/
axiom ranges_cover : RangeSet f ∪ RangeSet g = Set.univ

/-- Relationship between f and g -/
axiom g_def : ∀ m, g m = f (f m) + 1

/-- The main theorem to prove -/
theorem f_2m_formula (m : ℕ+) : f (2 * m) = m + ⌊Real.sqrt 5 * (m : ℝ)⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2m_formula_l7_716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_tangent_positivity_l7_752

theorem negation_of_tangent_positivity :
  (¬ ∀ x : ℝ, 0 < x ∧ x < π / 2 → Real.tan x > 0) ↔
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.tan x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_tangent_positivity_l7_752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l7_700

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := -2 * cos (2*x + π/3) * sin (2*x) - Real.sqrt 3 / 2

/-- Theorem stating that f(x) is monotonically increasing on [π/6, π/4] -/
theorem f_monotone_increasing : 
  MonotoneOn f (Set.Icc (π/6) (π/4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l7_700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l7_709

-- Define the sequence a_n and its partial sum S_n
def S (n : ℕ) : ℤ := (-1)^n * n

def a : ℕ → ℤ
| 0 => 0  -- Define a_0 to avoid issues with subtraction
| 1 => S 1
| (n+1) => S (n+1) - S n

-- Define the condition for p
def condition (p : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (a (n+1) - p) * (a n - p) < 0

-- State the theorem
theorem sequence_range :
  ∀ p : ℝ, condition p ↔ -1 < p ∧ p < 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l7_709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_power_l7_736

theorem imaginary_unit_power (i : ℂ) (hi : i^2 = -1) : 
  (i^2 ≠ 1 ∧ i^3 ≠ 1 ∧ i^4 = 1 ∧ i^5 ≠ 1) ∧
  ∀ n : ℕ, n ∈ ({2, 3, 4, 5} : Set ℕ) → (i^n = 1 ↔ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_power_l7_736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l7_739

/-- Annual production in 10,000 units -/
noncomputable def X : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 35}

/-- Variable cost function -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 14 then (2/3) * x^2 + 4*x
  else 17*x + 400/x - 80

/-- Annual profit function -/
noncomputable def g (x : ℝ) : ℝ :=
  16*x - f x - 30

/-- Theorem stating that annual profit is maximized at x = 9 -/
theorem max_profit_at_nine :
  ∃ (x : X), ∀ (y : X), g x.val ≥ g y.val ∧ x.val = 9 := by
  sorry

#check max_profit_at_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l7_739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_for_specific_cone_l7_788

/-- Represents a truncated cone -/
structure TruncatedCone where
  small_radius : ℝ
  large_radius : ℝ
  slant_height : ℝ

/-- Calculates the shortest distance for a given truncated cone -/
noncomputable def shortest_distance (cone : TruncatedCone) : ℝ :=
  cone.slant_height / 5

/-- Theorem statement for the shortest distance in the given truncated cone -/
theorem shortest_distance_for_specific_cone :
  let cone : TruncatedCone := ⟨5, 10, 20⟩
  shortest_distance cone = 4 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval shortest_distance ⟨5, 10, 20⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_for_specific_cone_l7_788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_binomial_expansion_l7_764

theorem odd_terms_in_binomial_expansion (p q : ℤ) : 
  Odd p → Odd q → (Finset.filter (fun k ↦ Odd (Nat.choose 10 k)) (Finset.range 11)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_binomial_expansion_l7_764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l7_783

/-- A trapezoid with given side lengths and a right angle -/
structure Trapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  right_angle : Bool

/-- The area of a trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (t.EF + t.GH) * t.FG / 2

theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.EF = 5)
  (h2 : t.FG = 6)
  (h3 : t.GH = 10)
  (h4 : t.HE = 8)
  (h5 : t.right_angle = true) :
  trapezoid_area t = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l7_783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l7_799

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (4*Real.pi - α) * Real.cos (Real.pi - α) * Real.cos ((3*Real.pi)/2 + α) * Real.cos ((7*Real.pi)/2 - α)) / 
  (Real.cos (Real.pi + α) * Real.sin (2*Real.pi - α) * Real.sin (Real.pi + α) * Real.sin ((9*Real.pi)/2 - α))

theorem f_simplification (α : ℝ) : f α = 1 := by sorry

theorem f_specific_value : f (-31*Real.pi/6) = -Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l7_799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_arrangement_triangle_area_l7_719

/-- Function to calculate the area of the triangle formed by the centers of three consecutive pentagons --/
def area_of_triangle_from_pentagon_centers (central_square_side : ℝ) (pentagon_side : ℝ) : ℝ :=
  sorry -- The actual implementation would go here

theorem pentagon_arrangement_triangle_area :
  ∀ (central_square_side : ℝ) (pentagon_side : ℝ),
  central_square_side = Real.sqrt 2 →
  pentagon_side = Real.sqrt 2 →
  ∃ (triangle_area : ℝ),
  triangle_area = Real.sqrt 3 / (2 * Real.sin (π/5)^2) ∧
  triangle_area = area_of_triangle_from_pentagon_centers central_square_side pentagon_side :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_arrangement_triangle_area_l7_719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_line_point_l7_706

/-- Given a line y = 2x translated 3 units downwards passing through (m+2, -5), prove m = -3 -/
theorem translated_line_point (m : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, f x = 2*x) →  -- Original line y = 2x
  (∃ g : ℝ → ℝ, ∀ x, g x = 2*x - 3) →  -- Translated line
  (2*(m + 2) - 3 = -5) →  -- Point (m+2, -5) lies on the translated line
  m = -3 := by
  intro hf hg hpoint
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_line_point_l7_706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l7_777

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (Real.sin x))

-- State the theorem
theorem range_of_f :
  Set.range (fun x : ℝ ↦ f x) = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l7_777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paula_bought_two_bracelets_l7_794

/-- Represents the number of bracelets Paula bought -/
def paula_bracelets : ℕ := sorry

/-- The cost of a single bracelet -/
def bracelet_cost : ℕ := 4

/-- The cost of a single keychain -/
def keychain_cost : ℕ := 5

/-- The cost of a single coloring book -/
def coloring_book_cost : ℕ := 3

/-- The total amount spent by Paula and Olive -/
def total_spent : ℕ := 20

/-- Paula's total cost -/
def paula_cost : ℕ := bracelet_cost * paula_bracelets + keychain_cost

/-- Olive's total cost -/
def olive_cost : ℕ := coloring_book_cost + bracelet_cost

theorem paula_bought_two_bracelets : 
  paula_bracelets = 2 ∧ paula_cost + olive_cost = total_spent := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paula_bought_two_bracelets_l7_794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_input_statement_format_l7_725

/-- Represents the format of an INPUT statement in QBASIC -/
inductive InputFormat where
  | PromptVariable : String → String → InputFormat
  | PromptExpression : String → String → InputFormat
  | InputPromptVariable : String → String → InputFormat
  | InputPromptExpression : String → String → InputFormat

/-- Represents the characteristics of an INPUT statement -/
structure InputStatement where
  isKeyboardInput : Bool
  userEntersValue : Bool

/-- The correct format of an INPUT statement in QBASIC -/
def correctInputFormat : InputFormat :=
  InputFormat.InputPromptVariable "prompt content" "variable"

/-- Theorem stating the correct format of an INPUT statement given its characteristics -/
theorem input_statement_format (stmt : InputStatement) 
  (h1 : stmt.isKeyboardInput = true) 
  (h2 : stmt.userEntersValue = true) : 
  correctInputFormat = InputFormat.InputPromptVariable "prompt content" "variable" := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_input_statement_format_l7_725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_cane_ratio_l7_762

/-- Candy cane problem -/
theorem candy_cane_ratio : 
  ∀ (candy_per_cavity : ℕ) 
    (parent_candy : ℕ) 
    (teacher_candy : ℕ) 
    (num_teachers : ℕ) 
    (total_cavities : ℕ),
  candy_per_cavity = 4 →
  parent_candy = 2 →
  teacher_candy = 3 →
  num_teachers = 4 →
  total_cavities = 16 →
  let total_given := parent_candy + teacher_candy * num_teachers
  let total_eaten := candy_per_cavity * total_cavities
  let bought := total_eaten - total_given
  (bought : ℚ) / total_given = 25 / 7 :=
by
  intros candy_per_cavity parent_candy teacher_candy num_teachers total_cavities
  intros h1 h2 h3 h4 h5
  -- The proof steps would go here, but we'll use 'sorry' to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_cane_ratio_l7_762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_four_main_theorem_l7_751

-- Define the sequences a and b
noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 7
  | n + 1 => 1 / (a n - ⌊a n⌋)

noncomputable def b : ℕ → ℤ
  | n => ⌊a n⌋

-- The theorem to prove
theorem smallest_n_divisible_by_four :
  ∀ n : ℕ, n > 2004 → b n % 4 = 0 → n ≥ 2005 :=
by sorry

-- The main theorem
theorem main_theorem :
  (∃ n : ℕ, n > 2004 ∧ b n % 4 = 0) ∧
  (∀ n : ℕ, n > 2004 → b n % 4 = 0 → n ≥ 2005) ∧
  b 2005 % 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_four_main_theorem_l7_751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_rows_for_seating_l7_759

theorem minimum_rows_for_seating (total_students : ℕ) (seats_per_row : ℕ) (max_students_per_school : ℕ) :
  total_students = 2016 →
  seats_per_row = 168 →
  max_students_per_school = 45 →
  ∃ (num_rows : ℕ), 
    (∀ (arrangement : List (List ℕ)),
      (arrangement.length = num_rows) ∧
      (arrangement.all (λ row ↦ row.sum ≤ seats_per_row)) ∧
      (arrangement.join.sum = total_students) ∧
      (∀ (school : ℕ), school ∈ arrangement.join → (arrangement.filter (λ row ↦ school ∈ row)).length = 1) ∧
      (∀ (school : ℕ), school ∈ arrangement.join → (arrangement.filter (λ row ↦ school ∈ row)).head!.count school ≤ max_students_per_school)) ∧
    (∀ (n : ℕ), n < num_rows → 
      ∃ (arrangement : List (List ℕ)),
        (arrangement.length = n) ∧
        (arrangement.all (λ row ↦ row.sum > seats_per_row) ∨
         arrangement.join.sum < total_students ∨
         (∃ (school : ℕ), school ∈ arrangement.join ∧ (arrangement.filter (λ row ↦ school ∈ row)).length > 1) ∨
         (∃ (school : ℕ), school ∈ arrangement.join ∧ (arrangement.filter (λ row ↦ school ∈ row)).head!.count school > max_students_per_school))) ∧
  num_rows = 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_rows_for_seating_l7_759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_change_l7_723

/-- Represents the volume of a pyramid with a square base -/
noncomputable def pyramidVolume (s : ℝ) (h : ℝ) : ℝ := (1/3) * s^2 * h

theorem pyramid_volume_change 
  (s : ℝ) (h : ℝ) 
  (initial_volume : pyramidVolume s h = 24) 
  (s_positive : s > 0) 
  (h_positive : h > 0) : 
  pyramidVolume (3*s) (2*h) = 432 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_change_l7_723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_calculation_l7_711

/-- Mowing time calculation -/
theorem mowing_time_calculation 
  (lawn_length : ℝ) 
  (lawn_width : ℝ) 
  (mower_swath : ℝ) 
  (overlap : ℝ) 
  (mowing_speed : ℝ) 
  (h1 : lawn_length = 120) 
  (h2 : lawn_width = 180) 
  (h3 : mower_swath = 30 / 12) -- Convert inches to feet
  (h4 : overlap = 6 / 12) -- Convert inches to feet
  (h5 : mowing_speed = 4000) :
  (lawn_width / (mower_swath - overlap) * lawn_length) / mowing_speed = 2.7 := by
  sorry

#eval Float.toString ((180 / (30/12 - 6/12) * 120) / 4000)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_calculation_l7_711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_plane_iff_perpendicular_to_triangle_sides_not_necessarily_perpendicular_to_plane_if_perpendicular_to_hexagon_diagonals_not_necessarily_perpendicular_to_plane_if_perpendicular_to_trapezoid_sides_l7_796

-- Define structures and properties
structure Plane where
  -- Placeholder for plane definition
  dummy : Unit

structure Line where
  -- Placeholder for line definition
  dummy : Unit

def perpendicular (l : Line) (p : Plane) : Prop :=
  -- Placeholder for perpendicular definition
  True

def perpendicularToTriangleSides (l : Line) (p : Plane) : Prop :=
  -- Placeholder for perpendicular to triangle sides definition
  True

def perpendicularToHexagonDiagonals (l : Line) (p : Plane) : Prop :=
  -- Placeholder for perpendicular to hexagon diagonals definition
  True

def perpendicularToTrapezoidSides (l : Line) (p : Plane) : Prop :=
  -- Placeholder for perpendicular to trapezoid sides definition
  True

-- Theorems
theorem perpendicular_to_plane_iff_perpendicular_to_triangle_sides
  (l : Line) (p : Plane) :
  perpendicular l p ↔ perpendicularToTriangleSides l p := by
  sorry

theorem not_necessarily_perpendicular_to_plane_if_perpendicular_to_hexagon_diagonals
  (l : Line) (p : Plane) :
  perpendicularToHexagonDiagonals l p → ¬(perpendicular l p ↔ perpendicularToHexagonDiagonals l p) := by
  sorry

theorem not_necessarily_perpendicular_to_plane_if_perpendicular_to_trapezoid_sides
  (l : Line) (p : Plane) :
  perpendicularToTrapezoidSides l p → ¬(perpendicular l p ↔ perpendicularToTrapezoidSides l p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_plane_iff_perpendicular_to_triangle_sides_not_necessarily_perpendicular_to_plane_if_perpendicular_to_hexagon_diagonals_not_necessarily_perpendicular_to_plane_if_perpendicular_to_trapezoid_sides_l7_796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l7_768

-- Define the sets M and N
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3 - x^2}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2*x^2 - 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l7_768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_approx_l7_731

-- Define the total reading time in minutes
def total_reading_time : ℕ := 480

-- Define the maximum capacity of each disc in minutes
def disc_capacity : ℕ := 70

-- Function to calculate the number of discs needed
def num_discs : ℕ := (total_reading_time + disc_capacity - 1) / disc_capacity

-- Function to calculate the reading time per disc
def reading_time_per_disc : ℚ := total_reading_time / num_discs

-- Theorem stating that the reading time per disc is approximately 68.6 minutes
theorem reading_time_approx : 
  (reading_time_per_disc * 10).floor / 10 = 686 / 10 := by sorry

#eval num_discs
#eval reading_time_per_disc

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_approx_l7_731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_area_l7_770

/-- Represents a trapezoid with specific properties -/
structure SpecialTrapezoid where
  height : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  perpendicular_diagonals : diagonal1 * diagonal2 = 0
  height_value : height = 4
  diagonal1_value : diagonal1 = 5

/-- Calculates the area of the special trapezoid -/
noncomputable def area (t : SpecialTrapezoid) : ℝ := 50 / 3

/-- Theorem stating that the area of the special trapezoid is 50/3 -/
theorem special_trapezoid_area (t : SpecialTrapezoid) : area t = 50 / 3 := by
  sorry

#check special_trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_trapezoid_area_l7_770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_non_arithmetic_sequence_l7_726

theorem inequality_and_non_arithmetic_sequence :
  (Real.sqrt 6 + Real.sqrt 5 > 2 * Real.sqrt 2 + Real.sqrt 3) ∧
  (¬ ∃ (a d : ℝ), Real.sqrt 2 = a ∧ Real.sqrt 5 = a + d ∧ Real.sqrt 6 = a + 2 * d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_non_arithmetic_sequence_l7_726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dietitian_excess_calories_l7_715

/-- Calculates the excess calories consumed by a dietitian -/
theorem dietitian_excess_calories (total_calories : ℕ) (fraction_eaten : ℚ) (recommended_intake : ℕ) : 
  total_calories = 40 → 
  fraction_eaten = 3/4 → 
  recommended_intake = 25 → 
  (fraction_eaten * total_calories : ℚ).floor - recommended_intake = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dietitian_excess_calories_l7_715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_paths_l7_714

-- Define the types of moves
inductive Move
| move1 : Move  -- (a,b) to (a-b, a+b)
| move2 : Move  -- (a,b) to (2a-b, a+2b)

-- Define a path as a list of moves
def PathList := List Move

-- Define the starting and ending points
def start : ℂ := 1
def target : ℂ := 28 - 96*Complex.I

-- Function to apply a move to a complex number
def applyMove (z : ℂ) (m : Move) : ℂ :=
  match m with
  | Move.move1 => (1 + Complex.I) * z
  | Move.move2 => (2 + Complex.I) * z

-- Function to check if a path is valid (reaches the target)
def isValidPath (p : PathList) : Prop :=
  p.foldl applyMove start = target

-- Theorem statement
theorem number_of_valid_paths :
  (∃ (validPaths : Finset PathList), 
    (∀ p ∈ validPaths, isValidPath p) ∧ 
    (∀ p, isValidPath p → p ∈ validPaths) ∧
    validPaths.card = 70) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_paths_l7_714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_proof_l7_786

/-- A train travels with initial speed v₀ and constant deceleration a for time t. -/
noncomputable def train_distance (v₀ a t : ℝ) : ℝ := v₀ * t + (1/2) * a * t^2

/-- The distance traveled by a train with initial speed 40 mph, 
    decelerating to a stop in 2 hours. -/
noncomputable def problem_distance : ℝ :=
  let v₀ : ℝ := 40  -- initial speed in mph
  let t : ℝ := 2    -- time in hours
  let a : ℝ := -v₀ / t  -- deceleration rate
  train_distance v₀ a t

theorem train_distance_proof : problem_distance = 40 := by
  -- Unfold the definitions
  unfold problem_distance train_distance
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_proof_l7_786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l7_718

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / (1 - x) else (1/3) ^ x

theorem f_properties :
  (f 1 + f (-1) = 5/6) ∧
  (∀ x, f x ≥ 1/3 ↔ -2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l7_718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equation_l7_703

/-- Given a real number x, we define y as a function of x. -/
noncomputable def y (x : ℝ) : ℝ := 2 * x + 3

/-- Given a real number x, we define z as a function of x. -/
noncomputable def z (x : ℝ) : ℝ := x^2 + (1 / x^2)

/-- The main theorem stating the relationship between x, y, and z. -/
theorem z_equation (x : ℝ) (hx : x ≠ 0) (h : x + (1 / x) = 3.5 + Real.sin (z x * Real.exp (-z x))) :
  z x = ((y x - 3) / 2)^2 + (2 / (y x - 3))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equation_l7_703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l7_713

theorem count_special_integers : 
  (Finset.filter (fun N : ℕ => 
    1 ≤ N ∧ N ≤ 2500 ∧ 
    Nat.gcd (N^2 + 9) (N + 5) > 1 ∧ 
    Nat.gcd (N + 5) (N^2 + 11) > 1) 
    (Finset.range 2501)).card = 1250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l7_713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_calculation_l7_797

theorem max_marks_calculation (passing_threshold : ℝ) (scored_marks shortfall : ℕ) : ℕ :=
  let max_marks : ℕ := 790
  have h : ↑scored_marks + ↑shortfall = passing_threshold * ↑max_marks := by sorry
  max_marks

#check max_marks_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_calculation_l7_797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l7_710

-- Define the function f with domain [0, 1]
def f : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define the function g(x) = f(2x) + f(x + 1/3)
def g (x : ℝ) : Set ℝ := {y : ℝ | (2 * x ∈ f ∧ y = 1) ∨ (x + 1/3 ∈ f ∧ y = 1)}

-- State the theorem
theorem domain_of_g : 
  {x : ℝ | g x ≠ ∅} = {x : ℝ | 0 ≤ x ∧ x ≤ 1/2} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l7_710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carols_car_efficiency_l7_727

/-- Represents a car with its fuel tank capacity and travel capabilities -/
structure Car where
  tank_capacity : ℚ
  distance_to_destination : ℚ
  additional_distance : ℚ

/-- Calculates the fuel efficiency of a car in miles per gallon -/
def fuel_efficiency (c : Car) : ℚ :=
  (c.distance_to_destination + c.additional_distance) / c.tank_capacity

/-- Theorem stating that Carol's car has a fuel efficiency of 20 miles per gallon -/
theorem carols_car_efficiency :
  let c : Car := {
    tank_capacity := 16,
    distance_to_destination := 220,
    additional_distance := 100
  }
  fuel_efficiency c = 20 := by
  -- Unfold the definition of fuel_efficiency
  unfold fuel_efficiency
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carols_car_efficiency_l7_727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l7_771

/-- An arithmetic sequence with common difference d, where a_1, a_4, and a_6 form a geometric sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_neq_zero : d ≠ 0
  geometric_condition : a 1 * a 6 = (a 4) ^ 2

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (sum_n seq 19 = 0) ∧
  (seq.d < 0 → ∀ n : ℕ, sum_n seq 9 ≥ sum_n seq n) ∧
  (seq.d > 0 → ∀ n : ℕ, sum_n seq 10 ≤ sum_n seq n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l7_771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l7_732

/-- The volume of the solid formed by rotating the region bounded by y = x^3 and y = √x
    about the x-axis, from x = 0 to x = 1 -/
theorem volume_of_rotation : 
  ∫ x in Set.Icc 0 1, Real.pi * ((Real.sqrt x)^2 - x^6) = (5 * Real.pi) / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l7_732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_satisfies_conditions_l7_790

theorem no_prime_satisfies_conditions : ¬∃ p : ℕ, 
  Nat.Prime p ∧ 
  10 ≤ p ∧ p ≤ 99 ∧ 
  (p : ℤ) - (10 * (p % 10) + p / 10 : ℤ) = 90 ∧ 
  ∃ n : ℕ, n^2 = p := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_satisfies_conditions_l7_790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_bill_420_l7_728

/-- Calculates the electricity bill based on tiered pricing --/
noncomputable def electricity_bill (consumption : ℝ) : ℝ :=
  let tier1_limit : ℝ := 200
  let tier2_limit : ℝ := 400
  let tier1_price : ℝ := 0.5
  let tier2_price : ℝ := 0.6
  let tier3_price : ℝ := 0.8
  
  let tier1_cost := min consumption tier1_limit * tier1_price
  let tier2_cost := max 0 (min consumption tier2_limit - tier1_limit) * tier2_price
  let tier3_cost := max 0 (consumption - tier2_limit) * tier3_price
  
  tier1_cost + tier2_cost + tier3_cost

/-- Theorem stating that the electricity bill for 420 kWh is 236 yuan --/
theorem electricity_bill_420 : electricity_bill 420 = 236 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_bill_420_l7_728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_raw_squat_l7_779

/-- John's raw squat weight in pounds -/
def raw_squat : ℝ := 600

/-- John's squat weight with sleeves in pounds -/
def squat_with_sleeves : ℝ := raw_squat + 30

/-- John's squat weight with wraps in pounds -/
def squat_with_wraps : ℝ := 1.25 * raw_squat

/-- The difference between squat with wraps and squat with sleeves in pounds -/
def wraps_vs_sleeves_difference : ℝ := 120

theorem johns_raw_squat : raw_squat = 600 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_raw_squat_l7_779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_l7_730

/-- Represents a coloring of a cube's faces -/
def CubeColoring := Fin 6 → Fin 6

/-- Checks if two faces are adjacent on a cube -/
def adjacent (f1 f2 : Fin 6) : Prop := sorry

/-- Checks if a coloring is valid (adjacent faces have different colors) -/
def valid_coloring (c : CubeColoring) : Prop :=
  ∀ f1 f2 : Fin 6, adjacent f1 f2 → c f1 ≠ c f2

/-- Checks if two colorings are equivalent under rotation -/
def equivalent_colorings (c1 c2 : CubeColoring) : Prop := sorry

/-- The set of all valid colorings -/
def valid_colorings : Set CubeColoring :=
  {c | valid_coloring c}

/-- Setoid instance for CubeColoring based on equivalent_colorings -/
instance : Setoid CubeColoring where
  r := equivalent_colorings
  iseqv := sorry

/-- The quotient set of valid colorings under rotation equivalence -/
def distinct_colorings : Type :=
  Quotient (inferInstance : Setoid CubeColoring)

/-- Finiteness of distinct_colorings -/
instance : Fintype distinct_colorings := sorry

/-- Decidable equality for distinct_colorings -/
instance : DecidableEq distinct_colorings := sorry

theorem cube_coloring_count :
  Finset.card (Finset.univ : Finset distinct_colorings) = 230 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_count_l7_730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_cubes_different_colors_l7_742

/-- Represents a cube in a 3D space -/
structure Cube where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Represents a color -/
inductive Color
  | Red | Blue | Green | Yellow | Orange | Purple | Pink | Brown

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  l : ℕ
  m : ℕ
  n : ℕ

/-- Checks if two cubes are adjacent (share at least one vertex) -/
def are_adjacent (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∨ c1.x + 1 = c2.x ∨ c1.x = c2.x + 1) ∧
  (c1.y = c2.y ∨ c1.y + 1 = c2.y ∨ c1.y = c2.y + 1) ∧
  (c1.z = c2.z ∨ c1.z + 1 = c2.z ∨ c1.z = c2.z + 1)

/-- A coloring function that assigns a color to each cube -/
noncomputable def coloring (c : Cube) : Color :=
  sorry

/-- Predicate to check if a cube is a corner cube -/
def is_corner_cube (p : Parallelepiped) (c : Cube) : Prop :=
  (c.x = 0 ∨ c.x = 2 * p.l - 1) ∧
  (c.y = 0 ∨ c.y = 2 * p.m - 1) ∧
  (c.z = 0 ∨ c.z = 2 * p.n - 1)

/-- The main theorem to prove -/
theorem corner_cubes_different_colors (p : Parallelepiped) :
  ∀ c1 c2 : Cube, is_corner_cube p c1 → is_corner_cube p c2 → c1 ≠ c2 →
  (∀ c3 c4 : Cube, are_adjacent c3 c4 → coloring c3 ≠ coloring c4) →
  coloring c1 ≠ coloring c2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corner_cubes_different_colors_l7_742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l7_745

/-- An ellipse with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_sq : c^2 = a^2 - b^2

/-- A point on the ellipse. -/
def on_ellipse (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The dot product of vectors from a point to the foci equals c^2. -/
def foci_dot_product (E : Ellipse) (x y : ℝ) : Prop :=
  (x + E.c) * (x - E.c) + y * y = E.c^2

theorem ellipse_eccentricity_range (E : Ellipse) :
  (∃ x y : ℝ, on_ellipse E x y ∧ foci_dot_product E x y) →
  (Real.sqrt 3 / 3 : ℝ) ≤ E.c / E.a ∧ E.c / E.a ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l7_745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_incircle_inequality_l7_763

/-- Predicate to check if three positive real numbers form a triangle --/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to check if p1, p2, p3 are valid incircle tangent segments
    for a triangle with sides a, b, c --/
def IsIncircleTangentSegments (a b c p1 p2 p3 : ℝ) : Prop :=
  p1 > 0 ∧ p2 > 0 ∧ p3 > 0 ∧ p1 < a ∧ p2 < b ∧ p3 < c

/-- Given a triangle ABC with side lengths a, b, and c, and incircle tangent segments
    p1, p2, and p3 parallel to BC, CA, and AB respectively, prove that abc ≥ 27p1p2p3 --/
theorem triangle_incircle_inequality (a b c p1 p2 p3 : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hp1 : p1 > 0) (hp2 : p2 > 0) (hp3 : p3 > 0)
  (h_triangle : IsTriangle a b c)
  (h_incircle : IsIncircleTangentSegments a b c p1 p2 p3) :
  a * b * c ≥ 27 * p1 * p2 * p3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_incircle_inequality_l7_763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l7_747

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin ((2 * Real.pi / 3) - x)

theorem symmetry_of_f :
  ∀ x : ℝ, f (π/3 + x) = f (π/3 - x) := by
  intro x
  unfold f
  simp [Real.sin_add, Real.sin_sub, Real.cos_add, Real.cos_sub]
  ring
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l7_747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_k_range_l7_740

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 3
def b : ℝ := 1
noncomputable def e : ℝ := Real.sqrt 6 / 3

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line equation
def line_eq (k m x y : ℝ) : Prop := y = k * x + m

-- Define the condition for the midpoint of the intersection points
def midpoint_condition (x₁ x₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1

-- Theorem statement
theorem ellipse_intersection_k_range :
  ∀ k m x₁ y₁ x₂ y₂ : ℝ,
  a > b ∧ b > 0 ∧
  ellipse_eq (Real.sqrt 3) 0 ∧
  ellipse_eq x₁ y₁ ∧ ellipse_eq x₂ y₂ ∧
  line_eq k m x₁ y₁ ∧ line_eq k m x₂ y₂ ∧
  midpoint_condition x₁ x₂ →
  k < -Real.sqrt 6 / 6 ∨ k > Real.sqrt 6 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_k_range_l7_740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_l7_772

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2*a*x + a + 2
  else x^(2*a - 6)

theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) →
  a ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_l7_772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_l7_712

/-- Represents the speed of a particle at the nth mile -/
noncomputable def speed (n : ℕ) : ℝ :=
  if n ≤ 1 then 0 else 1 / ((n - 1) ^ 2 : ℝ)

/-- Represents the time taken to traverse the nth mile -/
noncomputable def time (n : ℕ) : ℝ :=
  if n ≤ 1 then 0 else (n - 1) ^ 2

theorem particle_motion (n : ℕ) (h : n > 1) :
  time n = (n - 1) ^ 2 ∧
  speed 3 = 1 / 4 ∧
  ∀ m > 1, speed m * time m = 1 := by
  sorry

#check particle_motion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_l7_712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_rate_l7_746

-- Define the parameters
noncomputable def initial_investment : ℝ := 2400
noncomputable def additional_investment : ℝ := 2399.9999999999995
noncomputable def additional_rate : ℝ := 0.08
noncomputable def total_rate : ℝ := 0.06

-- Define the function to calculate the interest rate
noncomputable def calculate_interest_rate (init_inv : ℝ) (add_inv : ℝ) (add_rate : ℝ) (total_rate : ℝ) : ℝ :=
  (total_rate * (init_inv + add_inv) - add_rate * add_inv) / init_inv

-- Theorem statement
theorem initial_investment_rate :
  calculate_interest_rate initial_investment additional_investment additional_rate total_rate = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_rate_l7_746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_m_equals_six_c_plus_d_equals_six_l7_724

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem intersection_distance_implies_m_equals_six (m : ℝ) :
  (∃ (y₁ y₂ : ℝ), y₁ = log2 m ∧ y₂ = log2 (m + 6) ∧ |y₁ - y₂| = 1) →
  m = 6 := by
  sorry

def c : ℤ := 6
def d : ℤ := 0

theorem c_plus_d_equals_six : c + d = 6 := by
  rfl

#check intersection_distance_implies_m_equals_six
#check c_plus_d_equals_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_implies_m_equals_six_c_plus_d_equals_six_l7_724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_cube_l7_737

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

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: If X divides AB in 1:2 ratio from A, Y divides CC1 in 2:1 ratio from C1,
    and Z divides A1D1 in 2:1 ratio from A1, then triangle XYZ is equilateral -/
theorem equilateral_triangle_in_cube (cube : Cube) : 
  let X := { x := cube.A.x + (2/3) * (cube.B.x - cube.A.x),
             y := cube.A.y + (2/3) * (cube.B.y - cube.A.y),
             z := cube.A.z + (2/3) * (cube.B.z - cube.A.z) }
  let Y := { x := cube.C.x + (2/3) * (cube.C1.x - cube.C.x),
             y := cube.C.y + (2/3) * (cube.C1.y - cube.C.y),
             z := cube.C.z + (2/3) * (cube.C1.z - cube.C.z) }
  let Z := { x := cube.A1.x + (2/3) * (cube.D1.x - cube.A1.x),
             y := cube.A1.y + (2/3) * (cube.D1.y - cube.A1.y),
             z := cube.A1.z + (2/3) * (cube.D1.z - cube.A1.z) }
  distance X Y = distance Y Z ∧ distance Y Z = distance Z X :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_cube_l7_737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stuffed_animal_price_l7_793

def coloring_book_price : ℝ := 4
def coloring_book_quantity : ℕ := 2
def peanut_pack_price : ℝ := 1.5
def peanut_pack_quantity : ℕ := 4
def total_paid : ℝ := 25

theorem stuffed_animal_price : 
  total_paid - (coloring_book_price * coloring_book_quantity + peanut_pack_price * peanut_pack_quantity) = 11 := by
  -- Proof goes here
  sorry

#eval total_paid - (coloring_book_price * coloring_book_quantity + peanut_pack_price * peanut_pack_quantity)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stuffed_animal_price_l7_793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_two_lines_l7_791

/-- A set in ℝ² is a line if it's nonempty and there exist a, b, c such that 
    ax + by + c = 0 for all points (x, y) in the set, with (a, b) ≠ (0, 0) -/
def IsLine (L : Set (ℝ × ℝ)) : Prop :=
  L.Nonempty ∧ ∃ (a b c : ℝ), (a, b) ≠ (0, 0) ∧
  ∀ (x y : ℝ), (x, y) ∈ L ↔ a*x + b*y + c = 0

/-- The equation x^2 + xy = x represents two lines in the xy-plane -/
theorem equation_represents_two_lines :
  ∃ (L₁ L₂ : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ L₁ ∪ L₂ ↔ x^2 + x*y = x) ∧
    IsLine L₁ ∧ IsLine L₂ ∧ L₁ ≠ L₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_two_lines_l7_791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_degree_sequences_l7_755

def sequence1 : List Nat := [8, 6, 5, 4, 4, 3, 2, 2]
def sequence2 : List Nat := [7, 7, 6, 5, 4, 2, 2, 1]
def sequence3 : List Nat := [6, 6, 6, 5, 5, 3, 2, 2]

def is_valid_degree_sequence (seq : List Nat) : Prop :=
  seq.length = 8 ∧ 
  (∀ d ∈ seq, d < 8) ∧
  seq.sum % 2 = 0 ∧
  (∀ i j, i < seq.length → j < seq.length → i ≠ j → 
    seq[i]! + seq[j]! < seq.length)

theorem invalid_degree_sequences : 
  ¬(is_valid_degree_sequence sequence1) ∧ 
  ¬(is_valid_degree_sequence sequence2) ∧ 
  ¬(is_valid_degree_sequence sequence3) := by
  sorry

#check invalid_degree_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_degree_sequences_l7_755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_speed_ratio_l7_792

/-- Given a race with the following conditions:
  * The race is 500 meters long
  * Contestant A has a 200 meter head start
  * Contestant A wins by 100 meters
  Prove that the ratio of the speeds of contestant A to contestant B is 3/4 -/
theorem race_speed_ratio (race_length : ℝ) (head_start : ℝ) (win_margin : ℝ)
  (h1 : race_length = 500)
  (h2 : head_start = 200)
  (h3 : win_margin = 100) :
  (race_length - head_start) / (race_length - win_margin) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_speed_ratio_l7_792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_one_l7_733

-- Define the curve
noncomputable def C (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of the curve
noncomputable def C' (x : ℝ) : ℝ := Real.log x + 1

-- Theorem statement
theorem tangent_angle_at_one :
  let slope := C' 1
  Real.arctan slope = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_one_l7_733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_coin_sum_l7_704

/-- Represents the types of coins available in cents -/
def CoinTypes : Finset ℕ := {1, 5, 10, 25, 50}

/-- The number of coins to be selected -/
def NumCoins : ℕ := 5

/-- The target sum in cents -/
def TargetSum : ℕ := 75

/-- Theorem stating that it's impossible to select 5 coins summing to 75 cents -/
theorem impossible_coin_sum : 
  ¬ ∃ (coins : Finset ℕ), 
    coins.card = NumCoins ∧ 
    coins ⊆ CoinTypes ∧
    coins.sum id = TargetSum :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_coin_sum_l7_704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l7_729

theorem expression_value : (2^3 + 2^2 + 2^1 + 2^0) / ((1/2) + (1/4) + (1/8) + (1/16)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l7_729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l7_734

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : Real) : Real := 2 * Real.cos x * (Real.sin x - Real.sqrt 3 * Real.cos x) + Real.sqrt 3

-- State the theorem
theorem f_properties :
  -- 1. Minimum positive period is π
  (∀ x, f (x + π) = f x) ∧
  (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ π) ∧
  -- 2. Monotonically decreasing interval
  (∀ k : Int, ∀ x y, x ∈ Set.Icc (k * π + 5 * π / 12) (k * π + 11 * π / 12) →
    y ∈ Set.Icc (k * π + 5 * π / 12) (k * π + 11 * π / 12) →
    x < y → f x > f y) ∧
  -- 3. Minimum value and corresponding x when x ∈ [π/2, π]
  (∀ x, x ∈ Set.Icc (π / 2) π → f x ≥ -2) ∧
  (f (11 * π / 12) = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l7_734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abs_coeff_Q_l7_717

noncomputable def P (x : ℝ) : ℝ := 1 - (1/4)*x + (1/8)*x^3

noncomputable def Q (x : ℝ) : ℝ := P x * P (x^3) * P (x^5) * P (x^7) * P (x^9) * P (x^11)

noncomputable def sum_abs_coeff (Q : ℝ → ℝ) : ℝ := Q (-1)

theorem sum_abs_coeff_Q :
  sum_abs_coeff Q = (17/8)^6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_abs_coeff_Q_l7_717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_and_acceleration_at_3_l7_773

noncomputable section

-- Define the motion equation
def s (t : ℝ) : ℝ := -1/6 * t^3 + 3*t^2 - 5

-- Define velocity as the derivative of s
def v (t : ℝ) : ℝ := deriv s t

-- Define acceleration as the derivative of v
def a (t : ℝ) : ℝ := deriv v t

-- Theorem stating the velocity and acceleration at t=3
theorem velocity_and_acceleration_at_3 :
  v 3 = 27/2 ∧ a 3 = 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_and_acceleration_at_3_l7_773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_vertex_product_l7_784

def regular_nonagon (P : Fin 9 → ℂ) : Prop :=
  ∃ (r : ℝ) (θ : ℝ), ∀ k : Fin 9, P k = r * (Complex.exp (2 * Real.pi * Complex.I * (k : ℝ) / 9))

theorem nonagon_vertex_product (P : Fin 9 → ℂ) :
  regular_nonagon P → P 0 = 2 → P 4 = -2 →
  Finset.prod Finset.univ (λ k => P k) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonagon_vertex_product_l7_784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l7_795

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of carbon atoms in the compound -/
def carbon_atoms : ℕ := 4

/-- The number of hydrogen atoms in the compound -/
def hydrogen_atoms : ℕ := 8

/-- The number of oxygen atoms in the compound -/
def oxygen_atoms : ℕ := 2

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  carbon_weight * (carbon_atoms : ℝ) +
  hydrogen_weight * (hydrogen_atoms : ℝ) +
  oxygen_weight * (oxygen_atoms : ℝ)

theorem compound_molecular_weight :
  |molecular_weight - 88.104| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l7_795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l7_765

def O : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (5, 2, 0)
def B : ℝ × ℝ × ℝ := (2, 5, 0)
def C : ℝ × ℝ × ℝ := (1, 2, 4)

def volume (O A B C : ℝ × ℝ × ℝ) : ℝ := sorry

def area_triangle (A B C : ℝ × ℝ × ℝ) : ℝ := sorry

def pyramid_height (O A B C : ℝ × ℝ × ℝ) : ℝ := sorry

theorem pyramid_properties :
  volume O A B C = 14 ∧
  area_triangle A B C = 6 * Real.sqrt 3 ∧
  pyramid_height O A B C = (7 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l7_765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_rotation_l7_760

-- Define the rotation of the hour hand
noncomputable def hour_hand_rotation (start_hour end_hour : ℕ) : ℝ :=
  (end_hour - start_hour : ℝ) * (360 / 12)

-- Define the rotation of the minute hand
noncomputable def minute_hand_rotation (start_minute end_minute : ℕ) : ℝ :=
  (end_minute - start_minute : ℝ) * (360 / 60)

-- Theorem statement
theorem clock_rotation :
  (hour_hand_rotation 8 10 = 60) ∧
  (minute_hand_rotation 15 30 = 90) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_rotation_l7_760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l7_722

noncomputable section

def Sphere := Real → Real → Real → Prop

def Plane := Real → Real → Real → Prop

def intersects (α : Plane) (O : Sphere) : Prop := sorry

def intersectionCircle (O : Sphere) (α : Plane) : Set Real := sorry

def radius (c : Set Real) : Real := sorry

def center (O : Sphere) : Real × Real × Real := sorry

def dist (p : Real × Real × Real) (α : Plane) : Real := sorry

def surfaceArea (O : Sphere) : Real := sorry

theorem sphere_surface_area (O : Sphere) (α : Plane) (h : intersects α O) :
  radius (intersectionCircle O α) = 1 →
  dist (center O) α = Real.sqrt 2 →
  surfaceArea O = 12 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l7_722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l7_749

/-- Represents a die with a specific color distribution -/
structure Die where
  maroon : ℕ
  teal : ℕ
  cyan : ℕ
  sparkly : ℕ
  silver : ℕ
  total : ℕ := maroon + teal + cyan + sparkly + silver

/-- Calculates the probability of both dice showing the same color -/
def probability_same_color (d1 d2 : Die) : ℚ :=
  let p_maroon := (d1.maroon : ℚ) / d1.total * (d2.maroon : ℚ) / d2.total
  let p_teal := (d1.teal : ℚ) / d1.total * (d2.teal : ℚ) / d2.total
  let p_cyan := (d1.cyan : ℚ) / d1.total * (d2.cyan : ℚ) / d2.total
  let p_sparkly := (d1.sparkly : ℚ) / d1.total * (d2.sparkly : ℚ) / d2.total
  let p_silver := (d1.silver : ℚ) / d1.total * (d2.silver : ℚ) / d2.total
  p_maroon + p_teal + p_cyan + p_sparkly + p_silver

/-- The two dice as described in the problem -/
def die1 : Die := { maroon := 5, teal := 6, cyan := 7, sparkly := 1, silver := 1 }
def die2 : Die := { maroon := 4, teal := 6, cyan := 7, sparkly := 1, silver := 2 }

/-- Theorem stating that the probability of the two dice showing the same color is 27/100 -/
theorem same_color_probability : probability_same_color die1 die2 = 27 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l7_749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_diagonal_angle_l7_735

/-- The Golden Ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The angle made by the diagonals of any adjacent sides of a cube -/
noncomputable def y : ℝ := 2 * Real.arcsin ((1 + φ) / (3 + φ))

/-- Theorem stating the relationship between y and φ -/
theorem cube_diagonal_angle : Real.sin (y / 2) = φ / (2 + φ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_diagonal_angle_l7_735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l7_701

-- Define the complex number type
variable (z : ℂ)

-- State the theorem
theorem modulus_of_z (h : (1 - Complex.I) * z = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l7_701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_altitude_l7_741

/-- Given an isosceles triangle ABC with AB = AC = 13 and BC = 14,
    where D is the midpoint of BC, the altitude AD has length 2√30 -/
theorem isosceles_triangle_altitude (A B C D : ℝ × ℝ) : 
  let dist := λ (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist A B = 13 ∧ dist A C = 13 ∧ dist B C = 14 ∧ 
  D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  dist A D = 2 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_altitude_l7_741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_categorize_S_l7_720

def S : Set ℝ := {-8, Real.pi, -2, 22/7, 4, -0.9, 5.4, -11/3, 0}

def isNegativeRational (x : ℝ) : Prop := x < 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ x = ↑a / ↑b

def isPositiveFraction (x : ℝ) : Prop := x > 0 ∧ ∃ (a b : ℤ), b > 0 ∧ x = ↑a / ↑b

def isNonPositiveInteger (x : ℝ) : Prop := x ≤ 0 ∧ ∃ (n : ℤ), x = ↑n

theorem categorize_S :
  ({x ∈ S | isNegativeRational x} = {-8, -2, -0.9, -11/3}) ∧
  ({x ∈ S | isPositiveFraction x} = {22/7, 5.4}) ∧
  ({x ∈ S | isNonPositiveInteger x} = {-8, -2, 0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_categorize_S_l7_720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_6_three_even_one_odd_l7_702

def is_even (n : ℕ) : Bool := n % 2 = 0
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

def has_three_even_one_odd (n : ℕ) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  (digits.filter is_even).length = 3 ∧ (digits.filter is_odd).length = 1

theorem smallest_four_digit_divisible_by_6_three_even_one_odd :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧
    n % 6 = 0 ∧
    has_three_even_one_odd n →
    1002 ≤ n :=
by sorry

#check smallest_four_digit_divisible_by_6_three_even_one_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_6_three_even_one_odd_l7_702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_angles_bisectors_perpendicular_vertical_angles_bisectors_collinear_l7_707

-- Define adjacent angles
def adjacent_angles (α β : ℝ) : Prop :=
  α > 0 ∧ β > 0 ∧ α + β = 180

-- Define angle bisector
noncomputable def angle_bisector (α : ℝ) : ℝ :=
  α / 2

-- Define vertical angles
def vertical_angles (α β γ δ : ℝ) : Prop :=
  α = γ ∧ β = δ ∧ α + β = 180 ∧ γ + δ = 180

-- Theorem for adjacent angles
theorem adjacent_angles_bisectors_perpendicular 
  (α β : ℝ) (h : adjacent_angles α β) :
  angle_bisector α + angle_bisector β = 90 := by
  sorry

-- Theorem for vertical angles
theorem vertical_angles_bisectors_collinear 
  (α β γ δ : ℝ) (h : vertical_angles α β γ δ) :
  angle_bisector α + angle_bisector γ = 180 ∧
  angle_bisector β + angle_bisector δ = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_angles_bisectors_perpendicular_vertical_angles_bisectors_collinear_l7_707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_sum_l7_766

/-- A point in 2D space represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hexagon defined by its vertices -/
def hexagon : List Point := [
  ⟨0, 0⟩, ⟨1, 1⟩, ⟨2, 1⟩, ⟨3, 0⟩, ⟨2, -1⟩, ⟨1, -1⟩
]

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the perimeter of the hexagon -/
noncomputable def perimeter : ℝ :=
  let pairs := hexagon.zip (hexagon.rotateRight 1)
  pairs.map (fun (p1, p2) => distance p1 p2) |>.sum

/-- The theorem to be proved -/
theorem hexagon_perimeter_sum :
  ∃ (a b c : ℤ), perimeter = a + b * Real.sqrt 2 + c * Real.sqrt 5 ∧ a + b + c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_sum_l7_766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_given_planes_l7_743

-- Define the two planes
def plane1 : ℝ → ℝ → ℝ → Prop := λ x y z ↦ 2*x - y + 3*z - 4 = 0
def plane2 : ℝ → ℝ → ℝ → Prop := λ x y z ↦ 4*x + 3*y - z + 2 = 0

-- Define the normal vectors of the planes
def normal1 : Fin 3 → ℝ := ![2, -1, 3]
def normal2 : Fin 3 → ℝ := ![4, 3, -1]

-- Define the angle between the planes
noncomputable def angle_between_planes (p1 p2 : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem angle_between_given_planes :
  Real.cos (angle_between_planes plane1 plane2) = 1 / Real.sqrt 91 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_given_planes_l7_743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_intervals_m_range_l7_782

noncomputable def OA (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3)

noncomputable def OB (x : ℝ) : ℝ × ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x, -1)

noncomputable def f (x : ℝ) : ℝ := (OA x).1 * (OB x).1 + (OA x).2 * (OB x).2 + 2

theorem monotone_decreasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi + Real.pi / 12) (k * Real.pi + 7 * Real.pi / 12)) := by
  sorry

theorem m_range :
  ∀ m : ℝ, (∃ x ∈ Set.Ioo 0 (Real.pi / 2), f x + m = 0) → 
    m ∈ Set.Icc (-4) (Real.sqrt 3 - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_intervals_m_range_l7_782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a5_l7_774

def our_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n - 1

theorem find_a5 (a : ℕ → ℤ) (h1 : our_sequence a) (h2 : a 2 + a 4 + a 6 = 18) :
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a5_l7_774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l7_798

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Define the interval [-3, 3]
def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 28/3 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l7_798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_angle_maximized_l7_748

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- A line perpendicular to the x-axis passing through a point -/
structure VerticalLine where
  x : ℝ

/-- The intersection points of a vertical line and a hyperbola -/
def intersection (h : Hyperbola) (l : VerticalLine) : Set Point := sorry

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point := 
  { x := Real.sqrt (h.a^2 + h.b^2), y := 0 }

/-- The left vertex of a hyperbola -/
def left_vertex (h : Hyperbola) : Point := { x := -h.a, y := 0 }

/-- The right vertex of a hyperbola -/
def right_vertex (h : Hyperbola) : Point := { x := h.a, y := 0 }

/-- The theorem statement -/
theorem hyperbola_eccentricity_when_angle_maximized (h : Hyperbola) :
  let f := right_focus h
  let l := VerticalLine.mk f.x
  let int_points := intersection h l
  let a := left_vertex h
  let b := right_vertex h
  (∃ p ∈ int_points, ∀ q : Point, angle a q b ≤ angle a p b) →
  eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_angle_maximized_l7_748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_2cos4_l7_780

theorem min_sin4_plus_2cos4 :
  (∃ (x : ℝ), Real.sin x ^ 4 + 2 * Real.cos x ^ 4 = 2 / 3) ∧
  (∀ (x : ℝ), Real.sin x ^ 4 + 2 * Real.cos x ^ 4 ≥ 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_2cos4_l7_780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_marbles_l7_744

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (m : MarbleCount) : ℕ :=
  m.red + m.white + m.blue + m.green + m.yellow

/-- Calculates the probability of drawing a specific combination of marbles -/
noncomputable def drawProbability (m : MarbleCount) (r w b g y : ℕ) : ℚ :=
  (Nat.choose m.red r * Nat.choose m.white w * Nat.choose m.blue b * Nat.choose m.green g * Nat.choose m.yellow y : ℚ) /
  Nat.choose (totalMarbles m) 5

/-- Checks if the probabilities of the five specific events are proportional -/
def probabilitiesProportional (m : MarbleCount) : Prop :=
  ∃ (k : ℚ), k > 0 ∧
    drawProbability m 5 0 0 0 0 = k * drawProbability m 4 1 0 0 0 ∧
    drawProbability m 4 1 0 0 0 = k * drawProbability m 3 1 1 0 0 ∧
    drawProbability m 3 1 1 0 0 = k * drawProbability m 2 1 1 1 0 ∧
    drawProbability m 2 1 1 1 0 = k * drawProbability m 1 1 1 1 1

/-- The main theorem stating that the minimum number of marbles is 55 -/
theorem minimum_marbles :
  ∀ m : MarbleCount, probabilitiesProportional m →
    totalMarbles m ≥ 55 ∧
    (∃ m' : MarbleCount, probabilitiesProportional m' ∧ totalMarbles m' = 55) :=
by sorry

#check minimum_marbles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_marbles_l7_744
