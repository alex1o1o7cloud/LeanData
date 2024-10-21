import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l785_78548

theorem monic_quadratic_with_complex_root :
  ∃! p : Polynomial ℝ,
    Polynomial.Monic p ∧
    (Polynomial.degree p = 2) ∧
    (∀ x : ℂ, Polynomial.eval₂ Complex.ofReal x p = 0 ↔ x = Complex.mk 2 (-1)) ∧
    (∀ a b : ℝ, p = Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b →
      a = -4 ∧ b = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l785_78548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_probability_l785_78576

/-- The probability that the sum of dice rolls equals n at some point. -/
noncomputable def P (n : ℕ) : ℝ := sorry

/-- 
For n ≥ 7, the probability P_n that the sum of points from all dice rolls 
is equal to n satisfies the equation:
P_n = (1/6) * (P_(n-1) + P_(n-2) + P_(n-3) + P_(n-4) + P_(n-5) + P_(n-6))
-/
theorem dice_sum_probability (n : ℕ) (h : n ≥ 7) : 
  P n = (1 / 6) * (P (n - 1) + P (n - 2) + P (n - 3) + P (n - 4) + P (n - 5) + P (n - 6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_probability_l785_78576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_equation_solution_l785_78514

theorem combination_equation_solution (x : ℕ) 
  (h : Nat.choose 9 x = Nat.choose 9 (2 * x + 3)) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_equation_solution_l785_78514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_a_n_l785_78599

def a : ℕ → ℤ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | n + 2 => 2 * (a (n + 1))^2 - 1

theorem coprime_a_n (n : ℕ) (h : n > 0) : Nat.gcd n (a n).natAbs = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_a_n_l785_78599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l785_78507

/-- A hyperbola with equation x² - y²/3 = 1 has asymptotes y = ±√3x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - y^2 / 3 = 1) →
  ∃ (ε : ℝ → ℝ), (Filter.Tendsto ε Filter.atTop (nhds 0)) ∧
    (y = Real.sqrt 3 * x + ε x ∨ y = -Real.sqrt 3 * x + ε x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l785_78507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_in_parallel_planes_infinitely_many_perpendicular_lines_l785_78549

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (in_plane : Line → Plane → Prop)

-- Theorem 1
theorem parallel_lines_in_parallel_planes 
  (α β γ : Plane) (m n : Line) :
  parallel_plane α β →
  intersection α γ = m →
  intersection β γ = n →
  parallel m n :=
sorry

-- Theorem 2
theorem infinitely_many_perpendicular_lines
  (α : Plane) (m : Line) :
  ¬ perpendicular_plane m α →
  ∃ (S : Set Line), (∀ l ∈ S, in_plane l α ∧ perpendicular m l) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_in_parallel_planes_infinitely_many_perpendicular_lines_l785_78549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_24_l785_78555

theorem sum_of_divisors_24 : (Finset.sum (Nat.divisors 24) id) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_24_l785_78555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_is_five_l785_78550

-- Define the simple interest formula
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time / 100

-- Define the given values
def SI : ℝ := 4016.25
def P : ℝ := 8925
def R : ℝ := 9

-- Theorem statement
theorem time_is_five :
  ∃ T : ℝ, simple_interest P R T = SI ∧ T = 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_is_five_l785_78550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicks_speed_l785_78581

/-- Proves that Nick's speed is 5 laps per hour given the conditions of the problem -/
theorem nicks_speed (marla_speed : ℝ) (time : ℝ) (lap_difference : ℕ) 
  (h1 : marla_speed = 10) -- Marla's speed in laps per hour
  (h2 : time = 48 / 60) -- Time in hours (48 minutes)
  (h3 : lap_difference = 4) -- Difference in laps completed after 48 minutes
  : (marla_speed * time - lap_difference) / time = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicks_speed_l785_78581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_working_first_scenario_l785_78546

/- Define the theorem and its components -/
theorem men_working_first_scenario : ∃ (men_first : ℕ), men_first = 4 := by
  -- Define the constants
  let hours_first : ℕ := 10
  let hours_second : ℕ := 6
  let earnings_first : ℚ := 1400
  let earnings_second : ℚ := 1890
  let men_second : ℕ := 9
  let days_per_week : ℕ := 7

  -- Define men_first as a natural number
  let men_first : ℕ := 4

  -- The main proof
  have h1 : (men_first : ℚ) * hours_first * days_per_week / earnings_first = 
            (men_second : ℚ) * hours_second * days_per_week / earnings_second := by
    -- The actual calculation would go here
    sorry

  -- Show that men_first satisfies the equation
  have h2 : (4 : ℚ) * hours_first * days_per_week / earnings_first = 
            (men_second : ℚ) * hours_second * days_per_week / earnings_second := by
    -- The actual calculation would go here
    sorry

  -- Conclude that men_first = 4
  have h3 : men_first = 4 := by
    -- The actual proof would go here
    sorry

  -- Prove the existence
  exact ⟨men_first, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_working_first_scenario_l785_78546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_is_8_l785_78503

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a kite -/
structure Kite where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- Calculates the area of a kite -/
noncomputable def kiteArea (k : Kite) : ℝ :=
  let upperBase := k.v3.x - k.v1.x
  let upperHeight := k.v2.y - k.v1.y
  let lowerBase := k.v3.x - k.v1.x
  let lowerHeight := k.v1.y - k.v4.y
  triangleArea upperBase upperHeight + triangleArea lowerBase lowerHeight

/-- The theorem to be proved -/
theorem kite_area_is_8 (k : Kite) 
  (h1 : k.v1 = ⟨0, 4⟩) 
  (h2 : k.v2 = ⟨2, 6⟩) 
  (h3 : k.v3 = ⟨4, 4⟩) 
  (h4 : k.v4 = ⟨2, 0⟩) : 
  kiteArea k = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_is_8_l785_78503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_even_l785_78553

-- Define the real number a
variable (a : ℝ)

-- Define the function F
variable (F : ℝ → ℝ)

-- Define the function G
noncomputable def G (x : ℝ) : ℝ := F x * (1 / (a^x - 1) + 1 / 2)

-- State the theorem
theorem G_is_even (ha_pos : a > 0) (ha_neq_one : a ≠ 1) (hF_odd : ∀ x, F (-x) = -F x) :
  ∀ x, G a F (-x) = G a F x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_even_l785_78553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_decimal_values_l785_78591

theorem count_decimal_values (s : ℚ) : 
  (s ≥ 261400/1000000 ∧ s ≤ 287900/1000000) →
  (∀ (n d : ℕ), (n = 1 ∨ n = 2) ∧ d > 0 → |s - 3/11| ≤ |s - n/d|) →
  (∃ (w x y z : ℕ), w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧ 
    s = (w * 1000 + x * 100 + y * 10 + z) / 10000) →
  (Finset.filter (λ i : ℕ ↦ 
    i ≥ 2614 ∧ i ≤ 2879) (Finset.range 10000)).card = 266 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_decimal_values_l785_78591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l785_78524

/-- The area of an equilateral triangle with side length s -/
noncomputable def area_equilateral_triangle (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The side length of the large equilateral triangle -/
def large_side : ℝ := 12

/-- The side length of the small equilateral triangle -/
def small_side : ℝ := 6

/-- The area of the large equilateral triangle -/
noncomputable def area_large : ℝ := area_equilateral_triangle large_side

/-- The area of the small equilateral triangle -/
noncomputable def area_small : ℝ := area_equilateral_triangle small_side

/-- The area of the isosceles trapezoid -/
noncomputable def area_trapezoid : ℝ := area_large - area_small

theorem area_ratio_is_one_third :
  area_small / area_trapezoid = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l785_78524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_nine_probability_l785_78536

def s : Finset ℕ := {2, 3, 4, 5}
def b : Finset ℕ := {4, 5, 6, 7, 8}

theorem sum_nine_probability :
  (Finset.card (Finset.filter (λ p : ℕ × ℕ => p.1 ∈ s ∧ p.2 ∈ b ∧ p.1 + p.2 = 9) (s.product b)) : ℚ) /
  ((Finset.card s * Finset.card b) : ℚ) = 3 / 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_nine_probability_l785_78536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_inside_circle_l785_78577

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 1 = 0

def circle_center : ℝ × ℝ := (1, -2)

noncomputable def circle_radius : ℝ := Real.sqrt 6

noncomputable def distance_from_origin (point : ℝ × ℝ) : ℝ :=
  Real.sqrt (point.1^2 + point.2^2)

theorem origin_inside_circle :
  distance_from_origin circle_center < circle_radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_inside_circle_l785_78577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cyclic_product_sum_l785_78510

open Finset

def cyclic_product_sum (p : Fin 5 → Fin 5) : ℕ :=
  (p 0).val * (p 1).val + (p 1).val * (p 2).val + (p 2).val * (p 3).val + (p 3).val * (p 4).val + (p 4).val * (p 0).val

theorem max_cyclic_product_sum :
  (∃ (p : Equiv.Perm (Fin 5)), cyclic_product_sum p = 45) ∧
  (∀ (p : Equiv.Perm (Fin 5)), cyclic_product_sum p ≤ 45) ∧
  (Fintype.card {p : Equiv.Perm (Fin 5) | cyclic_product_sum p = 45} = 10) := by
  sorry

#eval 45 + 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cyclic_product_sum_l785_78510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atMostOneHeads_exactlyTwoHeads_mutually_exclusive_not_complementary_l785_78584

-- Define the sample space for three coin tosses
def SampleSpace := List Bool

-- Define the events
def atMostOneHeads (outcome : SampleSpace) : Prop :=
  (outcome.filter id).length ≤ 1

def exactlyTwoHeads (outcome : SampleSpace) : Prop :=
  (outcome.filter id).length = 2

-- Theorem statement
theorem atMostOneHeads_exactlyTwoHeads_mutually_exclusive_not_complementary :
  (∀ (outcome : SampleSpace), ¬(atMostOneHeads outcome ∧ exactlyTwoHeads outcome)) ∧
  (∃ (outcome : SampleSpace), ¬(atMostOneHeads outcome ∨ exactlyTwoHeads outcome)) := by
  sorry

#check atMostOneHeads_exactlyTwoHeads_mutually_exclusive_not_complementary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_atMostOneHeads_exactlyTwoHeads_mutually_exclusive_not_complementary_l785_78584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_nets_count_l785_78573

/-- A net is a flattened arrangement of the faces of a cube. -/
structure Net where
  faces : Finset (Fin 6)
  is_connected : Bool

/-- Two nets are considered equivalent if they can be transformed into each other through rotation or reflection. -/
def equivalent_nets (n1 n2 : Net) : Prop :=
  ∃ (f : Fin 6 → Fin 6), Function.Bijective f ∧ n1.faces = n2.faces.image f

/-- The set of all possible nets of a cube. -/
noncomputable def all_nets : Finset Net :=
  sorry

/-- The set of unique nets of a cube, considering equivalence under rotation and reflection. -/
noncomputable def unique_nets : Finset Net :=
  sorry

theorem cube_nets_count :
  Finset.card unique_nets = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_nets_count_l785_78573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l785_78560

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem f_monotone_increasing :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l785_78560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l785_78521

-- Define the points A, B, and C
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 8)
def C : ℝ × ℝ := (2, 6)

-- Define a function to check if a point is on a line segment
def is_on_segment (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (t * Q.1 + (1 - t) * P.1, t * Q.2 + (1 - t) * P.2)

-- Define a function to calculate the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem point_C_coordinates : 
  is_on_segment A B C ∧ distance A C = 3 * distance C B → C = (2, 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l785_78521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_sum_l785_78597

/-- Parabola with equation y² = 4x -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ × ℝ

/-- Points A and B are intersections of directrix and parabola -/
noncomputable def directrix_intersections (p : Parabola) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Point P is intersection of parabola and line x = -1 -/
noncomputable def point_P (p : Parabola) : ℝ × ℝ := sorry

/-- Vector from point1 to point2 -/
def vector (point1 point2 : ℝ × ℝ) : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)

/-- Scalar multiple of a vector -/
def scale_vector (s : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (s * v.1, s * v.2)

theorem parabola_vector_sum (p : Parabola) :
  let (A, B) := directrix_intersections p
  let P := point_P p
  let F := p.focus
  ∃ (l m : ℝ), vector P A = scale_vector l (vector A F) ∧
                vector P B = scale_vector m (vector B F) →
  l + m = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_sum_l785_78597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l785_78541

/-- Represents a hiker's journey --/
structure HikerJourney where
  speed : ℚ  -- Speed in minutes per kilometer
  initial_distance : ℚ  -- Initial distance hiked east in kilometers
  total_time : ℚ  -- Total time available in minutes

/-- Calculates the total distance hiked east given a hiker's journey --/
def total_distance_east (journey : HikerJourney) : ℚ :=
  journey.initial_distance + 
  (journey.total_time - journey.speed * journey.initial_distance) / (2 * journey.speed)

/-- Theorem stating the total distance hiked east for the given problem --/
theorem annika_hike_distance :
  let journey : HikerJourney := {
    speed := 12,
    initial_distance := 2.75,
    total_time := 51
  }
  total_distance_east journey = 3.5 := by
  sorry

#eval total_distance_east {speed := 12, initial_distance := 2.75, total_time := 51}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l785_78541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_after_addition_l785_78539

-- Define the initial conditions
def initial_volume : ℚ := 45
def initial_milk_ratio : ℚ := 4
def initial_water_ratio : ℚ := 1
def added_water : ℚ := 9

-- Define the function to calculate the new ratio
def new_milk_water_ratio (
  init_vol : ℚ
  ) (init_milk_ratio : ℚ
  ) (init_water_ratio : ℚ
  ) (added_water : ℚ
  ) : ℚ × ℚ :=
  let total_ratio := init_milk_ratio + init_water_ratio
  let milk_volume := (init_milk_ratio / total_ratio) * init_vol
  let initial_water_volume := (init_water_ratio / total_ratio) * init_vol
  let new_water_volume := initial_water_volume + added_water
  let new_milk_ratio := milk_volume / new_water_volume
  (new_milk_ratio, 1)

-- Theorem statement
theorem milk_water_ratio_after_addition :
  new_milk_water_ratio initial_volume initial_milk_ratio initial_water_ratio added_water = (2, 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_after_addition_l785_78539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_labeling_l785_78590

open Finset

theorem subset_labeling {α : Type*} [DecidableEq α] (S : Finset α) :
  ∃ (A : Fin (2^S.card) → Finset α),
    (A 0 = ∅) ∧
    (∀ n : Fin (2^S.card), n.val > 0 →
      ((A (n-1) ⊂ A n ∧ (A n \ A (n-1)).card = 1) ∨
       (A n ⊂ A (n-1) ∧ (A (n-1) \ A n).card = 1))) ∧
    (∀ T : Finset α, T ⊆ S → ∃ i, A i = T) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_labeling_l785_78590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_positive_iff_l785_78523

noncomputable def a (x : ℝ) : ℝ := x^3 - 100*x
noncomputable def b (x : ℝ) : ℝ := x^4 - 16
noncomputable def c (x : ℝ) : ℝ := x + 20 - x^2

noncomputable def median (x y z : ℝ) : ℝ :=
  max (min x y) (min (max x y) z)

theorem median_positive_iff (x : ℝ) :
  median (a x) (b x) (c x) > 0 ↔ 
  (x > -10 ∧ x < 0) ∨ (x > 2 ∧ x < 5) ∨ x > 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_positive_iff_l785_78523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_f_l785_78551

def f (x : ℝ) : ℝ := 3 * x - 2

theorem inverse_of_inverse_f (x : ℝ) : 
  (Function.invFun f) ((Function.invFun f) 14) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_f_l785_78551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_15_choose_7_l785_78534

theorem binomial_15_choose_7 : Nat.choose 15 7 = 6435 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_15_choose_7_l785_78534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floating_cylinder_specific_gravity_l785_78542

/-- The specific gravity of a floating cylinder -/
theorem floating_cylinder_specific_gravity :
  ∀ (r m : ℝ),
  r > 0 → m > 0 →
  let cylinder_volume := Real.pi * r^2 * m
  let submerged_volume := r^2 * (Real.pi/3 - Real.sqrt 3 / 4) * m
  let specific_gravity := submerged_volume / cylinder_volume
  specific_gravity = (Real.pi/3 - Real.sqrt 3 / 4) / Real.pi :=
by
  intros r m hr hm
  -- Proof goes here
  sorry

#check floating_cylinder_specific_gravity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floating_cylinder_specific_gravity_l785_78542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l785_78562

theorem rationalize_denominator :
  2 / (3^(1/3) + 27^(1/3)) = 1 - 3^(1/3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l785_78562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_L_satisfies_conditions_l785_78538

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos (3 * x)

def tangent_line (x : ℝ) : ℝ := 2 * x + 1

def line_L (m : ℝ) (x : ℝ) : ℝ := 2 * x + m

theorem line_L_satisfies_conditions (m : ℝ) :
  (m = -4 ∨ m = 6) →
  (∀ x, (deriv (line_L m)) x = (deriv tangent_line) x) ∧
  (abs (m - 1) / Real.sqrt (1 + 2^2) = Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_L_satisfies_conditions_l785_78538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_perimeter_l785_78527

/-- A triangle with sides in geometric progression --/
structure GeometricTriangle where
  -- The common ratio of the geometric progression
  r : ℝ
  -- Constraint that r is greater than 1
  r_gt_one : r > 1

/-- The perimeter of a geometric triangle --/
noncomputable def perimeter (t : GeometricTriangle) : ℝ := 16 + 16 * t.r + 16 * t.r^2

/-- The sine ratio for the given trigonometric equation --/
noncomputable def sine_ratio (t : GeometricTriangle) : ℝ :=
  (1 - 2 * t.r + 3 * t.r^2) / (3 * t.r^2 - 2 * t.r + 3)

theorem geometric_triangle_perimeter :
  ∀ t : GeometricTriangle,
    sine_ratio t = 19/9 →
    perimeter t = 76 := by
  intro t h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_perimeter_l785_78527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_location_l785_78528

/-- The milepost of the second exit -/
noncomputable def second_exit : ℝ := 30

/-- The milepost of the eighth exit -/
noncomputable def eighth_exit : ℝ := 110

/-- The milepost of the restaurant -/
noncomputable def restaurant : ℝ := (second_exit + eighth_exit) / 2

/-- Proof that the restaurant is at milepost 70 -/
theorem restaurant_location : restaurant = 70 := by
  -- Unfold the definitions
  unfold restaurant second_exit eighth_exit
  -- Simplify the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_location_l785_78528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_pi_third_l785_78563

noncomputable def f : ℕ → ℝ → ℝ
  | 0, x => 2 * Real.cos x
  | n + 1, x => 4 / (2 - f n x)

theorem f_2023_pi_third : f 2023 (π / 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2023_pi_third_l785_78563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_is_87_5_l785_78530

/-- Calculates the total time Marcus spends with his dog -/
noncomputable def total_time_with_dog (bath_time : ℝ) (fetch_time : ℝ) (training_time : ℝ) 
  (first_mile_speed : ℝ) (second_mile_speed : ℝ) (third_mile_speed : ℝ) : ℝ :=
  let blow_dry_time := bath_time / 2
  let first_mile_time := 60 / first_mile_speed
  let second_mile_time := 60 / second_mile_speed
  let third_mile_time := 60 / third_mile_speed
  bath_time + blow_dry_time + fetch_time + training_time + 
  first_mile_time + second_mile_time + third_mile_time

/-- Theorem stating that the total time Marcus spends with his dog is 87.5 minutes -/
theorem total_time_is_87_5 : 
  total_time_with_dog 20 15 10 6 4 8 = 87.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_time_with_dog 20 15 10 6 4 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_is_87_5_l785_78530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_7_equiv_tail_cuts_l785_78505

/-- Performs one tail cut operation on a positive integer -/
def tailCut (n : Nat) : Nat :=
  (n / 10) - 2 * (n % 10)

/-- Checks if a number is divisible by 7 after a finite number of tail cuts -/
def isDivisibleBy7AfterTailCuts (n : Nat) : Prop :=
  ∃ k : Nat, (Nat.iterate tailCut k n) % 7 = 0

/-- Theorem stating the equivalence of divisibility by 7 and tail cut method -/
theorem divisibility_by_7_equiv_tail_cuts (n : Nat) :
  n % 7 = 0 ↔ isDivisibleBy7AfterTailCuts n := by
  sorry

#check divisibility_by_7_equiv_tail_cuts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_7_equiv_tail_cuts_l785_78505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_point_pairs_count_l785_78583

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else 2 / Real.exp x

-- Define what it means for two points to be a "sister point pair"
def is_sister_point_pair (A B : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  let (x_A, y_A) := A
  let (x_B, y_B) := B
  f x_A = y_A ∧ f x_B = y_B ∧ x_B = -x_A ∧ y_B = -y_A

-- State the theorem
theorem sister_point_pairs_count :
  ∃ (A B C D : ℝ × ℝ),
    is_sister_point_pair A B f ∧
    is_sister_point_pair C D f ∧
    A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧
    (∀ (E F : ℝ × ℝ), is_sister_point_pair E F f →
      (E = A ∧ F = B) ∨ (E = B ∧ F = A) ∨
      (E = C ∧ F = D) ∨ (E = D ∧ F = C)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_point_pairs_count_l785_78583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l785_78522

theorem sin_double_angle_special (α : ℝ) :
  Real.sin (π / 4 + α) = 2 / 5 → Real.sin (2 * α) = -17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l785_78522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l785_78592

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 360 meters, traveling at 45 km/hour,
    will take approximately 41.6 seconds to pass a bridge of length 160 meters -/
theorem train_bridge_passing_time :
  ∃ ε > 0, |time_to_pass_bridge 360 160 45 - 41.6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l785_78592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_determine_order_l785_78582

/-- Represents a question about the order of three weights -/
structure Question where
  a : Fin 5
  b : Fin 5
  c : Fin 5

/-- Represents a strategy for asking questions -/
def Strategy := List Question

/-- The number of possible orderings of 5 weights -/
def numOrderings : Nat := 120

/-- The maximum number of questions allowed -/
def maxQuestions : Nat := 9

/-- Represents the result of asking a question -/
inductive Answer
| Yes
| No

/-- A function that determines the answer to a question given an ordering -/
def answerQuestion (q : Question) (ordering : Fin 5 → Fin 5) : Answer :=
  if ordering q.a < ordering q.b ∧ ordering q.b < ordering q.c then
    Answer.Yes
  else
    Answer.No

/-- The theorem stating that it's impossible to determine the exact order with 9 questions -/
theorem impossible_to_determine_order :
  ∀ (s : List Question),
    s.length ≤ maxQuestions →
    ∃ (ordering1 ordering2 : Fin 5 → Fin 5),
      ordering1 ≠ ordering2 ∧
      (∀ (q : Question), q ∈ s → answerQuestion q ordering1 = answerQuestion q ordering2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_determine_order_l785_78582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goose_eggs_laid_l785_78526

-- Define the number of goose eggs laid at the pond
def total_eggs : ℕ := 2700

-- Define the fraction of eggs that hatched
def hatch_rate : ℚ := 2/3

-- Define the fraction of hatched geese that survived the first month
def first_month_survival_rate : ℚ := 3/4

-- Define the fraction of geese that survived the first month but did not survive the first year
def first_year_death_rate : ℚ := 3/5

-- Define the number of geese that survived the first year
def survived_first_year : ℕ := 180

-- Theorem stating that the number of goose eggs laid at the pond is 2700
theorem goose_eggs_laid : 
  (↑total_eggs * hatch_rate * first_month_survival_rate * (1 - first_year_death_rate) : ℚ) = survived_first_year := by
  sorry

-- Assumption that no more than one goose hatched from each egg
axiom one_goose_per_egg : ∀ egg : ℕ, egg ≤ total_eggs → (↑egg : ℚ) ≤ 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goose_eggs_laid_l785_78526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_value_l785_78543

def increasing_sequence (b : ℕ → ℕ) : Prop :=
  (∀ n, b (n + 1) > b n) ∧
  (∀ n, n ≥ 1 → b (n + 2) = b (n + 1) + b n)

theorem tenth_term_value
  (b : ℕ → ℕ)
  (h_seq : increasing_sequence b)
  (h_ninth : b 9 = 544) :
  b 10 = 883 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_value_l785_78543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_squared_quarter_sum_angles_l785_78587

theorem sine_squared_quarter_sum_angles 
  (α β : Real) 
  (h1 : Real.sin α + Real.sin β = Real.sin α * Real.sin β) 
  (h2 : Real.tan ((α - β) / 2) = 1 / 3) : 
  (Real.sin ((α + β) / 4))^2 = (5 + Real.sqrt 6) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_squared_quarter_sum_angles_l785_78587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_X_l785_78519

-- Define the probability density function
noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if -c < x ∧ x < c then 1 / (Real.pi * Real.sqrt (c^2 - x^2))
  else 0

-- Define the random variable X
def X : Type := ℝ

-- State the theorem
theorem variance_of_X (c : ℝ) (hc : c > 0) :
  ∃ (variance : ℝ), variance = c^3 / 2 ∧
  variance = ∫ x in -c..c, x^2 * f c x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_X_l785_78519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_huawei_mate60_sales_rounding_l785_78501

/-- Rounds a number to the nearest hundred thousand -/
def roundToNearestHundredThousand (n : ℕ) : ℕ :=
  (n + 50000) / 100000 * 100000

/-- Expresses a natural number in scientific notation -/
noncomputable def toScientificNotation (n : ℕ) : ℝ × ℤ :=
  let log10 := Real.log n / Real.log 10
  let exponent := Int.floor log10
  let coefficient := n / (10 : ℝ) ^ exponent
  (coefficient, exponent)

theorem huawei_mate60_sales_rounding :
  let original := 1694000
  let rounded := roundToNearestHundredThousand original
  let (coeff, exp) := toScientificNotation rounded
  coeff = 1.7 ∧ exp = 6 := by
  sorry

#eval roundToNearestHundredThousand 1694000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_huawei_mate60_sales_rounding_l785_78501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_1_to_minus89_l785_78518

/-- An arithmetic sequence with first term a₁, last term aₙ, and common difference d has n terms. -/
def arithmetic_sequence_length (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℕ :=
  Int.natAbs ((aₙ - a₁) / d + 1)

/-- The arithmetic sequence starting with 1, ending with -89, and having a common difference of -2 contains exactly 46 terms. -/
theorem arithmetic_sequence_1_to_minus89 :
  arithmetic_sequence_length 1 (-89) (-2) = 46 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_1_to_minus89_l785_78518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitudes_intersect_l785_78540

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- An altitude of a triangle -/
structure Altitude where
  base : Point2D  -- The vertex from which the altitude is drawn
  foot : Point2D  -- The point where the altitude intersects the opposite side

/-- The orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : Point2D :=
  sorry

/-- A point lies on a line (altitude in this case) -/
def pointOnLine (p : Point2D) (alt : Altitude) : Prop :=
  sorry

/-- Theorem: The altitudes of a triangle intersect at a single point -/
theorem altitudes_intersect (t : Triangle) :
  ∃! H : Point2D,
    (∀ alt : Altitude, pointOnLine H alt) ∧
    (∀ alt1 alt2 : Altitude, alt1 ≠ alt2 → alt1.base ≠ alt2.base) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitudes_intersect_l785_78540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l785_78593

/-- Calculates the length of a bridge given train parameters --/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Theorem stating the bridge length calculation --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ)
  (h1 : train_length = 360)
  (h2 : train_speed_kmh = 50)
  (h3 : time_to_pass = 36) :
  ∃ (ε : ℝ), ε > 0 ∧ |bridge_length train_length train_speed_kmh time_to_pass - 140| < ε :=
by
  sorry

/-- Computes an approximation of the bridge length --/
def bridge_length_approx : ℚ :=
  let train_length : ℚ := 360
  let train_speed_kmh : ℚ := 50
  let time_to_pass : ℚ := 36
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

#eval bridge_length_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l785_78593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_formula_l785_78568

noncomputable def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (1/2) * (a n + 1 / a n)

theorem general_term_formula (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n > 0) →
  (∀ n : ℕ, n > 0 → sequence_sum a n = (1/2) * (a n + 1 / a n)) →
  ∀ n : ℕ, n > 0 → a n = Real.sqrt n - Real.sqrt (n - 1) := by
  sorry

#check general_term_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_formula_l785_78568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_perpendicular_l785_78531

theorem tangents_perpendicular (a : ℝ) (h : a ≠ 0) :
  ∀ x₀ : ℝ, Real.cos x₀ = a * Real.tan x₀ →
  (-Real.sin x₀) * (a * (1 / (Real.cos x₀)^2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_perpendicular_l785_78531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_on_imaginary_axis_unit_circle_sum_with_reciprocal_real_l785_78511

-- Define a point in the complex plane
def Point := ℂ

-- Define the imaginary axis
def ImaginaryAxis := {z : ℂ | z.re = 0}

-- Statement 1
theorem purely_imaginary_on_imaginary_axis (z : ℂ) :
  (∃ b : ℝ, z = Complex.I * b) → (z : Point) ∈ ImaginaryAxis := by
  sorry

-- Statement 2
theorem unit_circle_sum_with_reciprocal_real (z : ℂ) (a b : ℝ) :
  z = Complex.mk a b → a^2 + b^2 = 1 → ∃ r : ℝ, z + z⁻¹ = r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_on_imaginary_axis_unit_circle_sum_with_reciprocal_real_l785_78511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_4_l785_78596

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else x^2

-- State the theorem
theorem f_inverse_of_4 (a : ℝ) : f a = 4 ↔ a = -4 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_4_l785_78596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l785_78578

-- Define the structure of a syllogism
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

-- Define the logarithmic function and its properties
def is_logarithmic_function (f : ℝ → ℝ) : Prop := sorry

def is_increasing_function (f : ℝ → ℝ) : Prop := sorry

-- Define the specific logarithmic functions
noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def log_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem minor_premise_identification :
  ∀ (a : ℝ),
  a > 1 →
  is_logarithmic_function log_2 →
  is_increasing_function (log_a a) →
  is_increasing_function log_2 →
  (Syllogism.mk
    (is_increasing_function (log_a a))
    (is_logarithmic_function log_2)
    (is_increasing_function log_2)).minor_premise =
    (is_logarithmic_function log_2) :=
by sorry

#check minor_premise_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_premise_identification_l785_78578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_seq_theorem_l785_78571

/-- An arithmetic sequence starting with 1 -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- A geometric sequence starting with 1 -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- The combined sequence c_n = a_n + b_n -/
def combined_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem combined_seq_theorem (d r k : ℕ) :
  (∀ n, arithmetic_seq d (n + 1) > arithmetic_seq d n) →
  (∀ n, geometric_seq r (n + 1) > geometric_seq r n) →
  combined_seq d r (k - 1) = 200 →
  combined_seq d r (k + 1) = 900 →
  combined_seq d r k = 928 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_seq_theorem_l785_78571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l785_78504

/-- Given an angle α and a real number a, proves that if the terminal side of α
    passes through (3a-9, a+2), cos α ≤ 0, and sin α > 0, then -2 < a ≤ 3 -/
theorem angle_range (α : ℝ) (a : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos α) = 3*a - 9 ∧ r * (Real.sin α) = a + 2) →
  (Real.cos α ≤ 0) →
  (Real.sin α > 0) →
  (-2 < a ∧ a ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l785_78504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_C_l785_78586

open Real

-- Define the curve C
noncomputable def C (α : ℝ) : ℝ × ℝ := (sin α - cos α, 2 * sin α * cos α)

-- State the theorem
theorem cartesian_equation_of_C :
  ∀ (x y : ℝ), (∃ α : ℝ, C α = (x, y)) ↔ 
  (y = -x^2 + 1 ∧ x ∈ Set.Icc (-sqrt 2) (sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_C_l785_78586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solutions_count_l785_78500

theorem inequality_solutions_count : 
  (Finset.filter (fun x : ℕ => x > 0 ∧ x < 4) (Finset.range 4)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solutions_count_l785_78500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_plane_implies_planes_perp_l785_78556

/-- Two planes are different if they are not equal -/
def DifferentPlanes (α β : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  α ≠ β

/-- Two lines are different if they are not equal -/
def DifferentLines (l m : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  l ≠ m

/-- A line is a subset of a plane -/
def LineInPlane (l : Set (EuclideanSpace ℝ (Fin 3))) (α : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  l ⊆ α

/-- A line is perpendicular to a plane -/
def LinePerpToPlane (l : Set (EuclideanSpace ℝ (Fin 3))) (β : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  sorry

/-- A plane is perpendicular to another plane -/
def PlanePerpToPlane (α β : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  sorry

theorem line_perp_plane_implies_planes_perp
  (α β : Set (EuclideanSpace ℝ (Fin 3))) (l m : Set (EuclideanSpace ℝ (Fin 3))) 
  (h1 : DifferentPlanes α β)
  (h2 : DifferentLines l m)
  (h3 : LineInPlane l α)
  (h4 : LineInPlane m β)
  (h5 : LinePerpToPlane l β) :
  PlanePerpToPlane α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_plane_implies_planes_perp_l785_78556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l785_78595

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x

-- State the theorem
theorem tangent_slope_at_one :
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l785_78595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_setup_I_greater_area_l785_78554

/-- Represents the setup for a rabbit tied to a circular garden --/
structure RabbitSetup where
  garden_radius : ℝ
  leash_length : ℝ
  tie_distance : ℝ  -- Distance from center to tie point

/-- Calculates the area accessible to the rabbit in a given setup --/
noncomputable def accessible_area (setup : RabbitSetup) : ℝ :=
  if setup.tie_distance + setup.leash_length ≤ setup.garden_radius then
    -- Case where rabbit can't reach outside the garden
    (1/2) * Real.pi * setup.leash_length^2
  else
    -- Case where rabbit can reach outside the garden
    let inner_radius := setup.garden_radius - setup.tie_distance
    let outer_arc := Real.arccos (inner_radius / setup.leash_length)
    (1/2) * Real.pi * inner_radius^2 + 
    (1/2) * setup.leash_length^2 * (Real.pi - 2 * outer_arc + Real.sin (2 * outer_arc))

/-- Theorem stating that Setup I provides more area than Setup II --/
theorem setup_I_greater_area :
  let setup_I : RabbitSetup := ⟨10, 6, 0⟩
  let setup_II : RabbitSetup := ⟨10, 6, 7⟩
  accessible_area setup_I - accessible_area setup_II = (9/2) * Real.pi := by
  sorry

#eval "Theorem stated and proof skipped with 'sorry'"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_setup_I_greater_area_l785_78554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_purchase_equation_l785_78565

theorem tshirt_purchase_equation (x : ℝ) (h : x > 0) :
  120000 / x + 5 = 187500 / ((1 + 0.4) * x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_purchase_equation_l785_78565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pumps_to_fill_tires_l785_78579

/-- Represents the state and properties of a tire --/
structure Tire where
  capacity : ℚ
  initial_fill_percentage : ℚ
  injection_rate : ℚ
  deflation_rate : ℚ

/-- Calculates the number of pumps needed to fill a tire --/
def pumps_needed (tire : Tire) : ℚ :=
  let air_needed := tire.capacity * (1 - tire.initial_fill_percentage)
  let effective_rate := tire.injection_rate - tire.deflation_rate
  air_needed / effective_rate

/-- The total number of pumps needed to fill all tires --/
def total_pumps (tires : List Tire) : ℕ :=
  (tires.map pumps_needed).sum.ceil.toNat

/-- Theorem stating that 33 pumps are needed to fill all tires --/
theorem pumps_to_fill_tires : 
  let tires : List Tire := [
    { capacity := 500, initial_fill_percentage := 0, injection_rate := 50, deflation_rate := 0 },
    { capacity := 500, initial_fill_percentage := 0, injection_rate := 50, deflation_rate := 0 },
    { capacity := 500, initial_fill_percentage := 2/5, injection_rate := 40, deflation_rate := 10 },
    { capacity := 500, initial_fill_percentage := 7/10, injection_rate := 60, deflation_rate := 0 }
  ]
  total_pumps tires = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pumps_to_fill_tires_l785_78579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_midpoint_locus_l785_78585

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the locus of midpoint M
def midpoint_locus (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1)^2 = 1/4

theorem circle_line_intersection_and_midpoint_locus :
  ∀ m : ℝ,
  (∃ (x1 y1 x2 y2 : ℝ), 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l m x1 y1 ∧ line_l m x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)) ∧
  (∀ (x1 y1 x2 y2 : ℝ),
    circle_C x1 y1 → circle_C x2 y2 →
    line_l m x1 y1 → line_l m x2 y2 →
    midpoint_locus ((x1 + x2) / 2) ((y1 + y2) / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_midpoint_locus_l785_78585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l785_78520

/-- The molar mass of aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of iodine in g/mol -/
noncomputable def molar_mass_I : ℝ := 126.90

/-- The number of aluminum atoms in AlI3 -/
def num_Al_atoms : ℕ := 1

/-- The number of iodine atoms in AlI3 -/
def num_I_atoms : ℕ := 3

/-- The molar mass of AlI3 in g/mol -/
noncomputable def molar_mass_AlI3 : ℝ := molar_mass_Al * num_Al_atoms + molar_mass_I * num_I_atoms

/-- The mass percentage of Al in AlI3 -/
noncomputable def mass_percentage_Al : ℝ := (molar_mass_Al / molar_mass_AlI3) * 100

theorem mass_percentage_Al_approx :
  |mass_percentage_Al - 6.62| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l785_78520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_prism_path_length_l785_78506

/-- The path length constant for a rolling rectangular prism -/
noncomputable def path_length_constant : ℝ := Real.sqrt 17 - 2

/-- The dimensions of the rectangular prism -/
def prism_dimensions : Fin 3 → ℝ
| 0 => 2
| 1 => 2
| 2 => 1
| _ => 0  -- This case should never occur due to Fin 3

theorem rolling_prism_path_length :
  let prism := prism_dimensions
  let roll_distance := 4  -- Full rotation
  let radius := (Real.sqrt ((prism 0)^2 + (prism 2 / 2)^2) / 2) - (prism 2 / 2)
  path_length_constant = (roll_distance * radius) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_prism_path_length_l785_78506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_omitted_permutation_l785_78516

/-- A permutation of the first 2008 natural numbers with one number omitted. -/
def OmittedPermutation := Fin 2007 → Fin 2008

/-- The differences between adjacent elements in the permutation, including the circular difference. -/
def differences (p : OmittedPermutation) : Fin 2007 → ℕ :=
  fun i => Int.natAbs (p i.val - p ((i.val + 1) % 2007))

/-- Predicate to check if all differences are distinct. -/
def allDifferentDifferences (p : OmittedPermutation) : Prop :=
  ∀ i j : Fin 2007, i ≠ j → differences p i ≠ differences p j

theorem exists_valid_omitted_permutation :
  ∃ (p : OmittedPermutation), allDifferentDifferences p :=
sorry

#check exists_valid_omitted_permutation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_omitted_permutation_l785_78516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_difference_l785_78513

noncomputable def coupon_x (price : ℝ) : ℝ := 0.2 * price

def coupon_y (price : ℝ) : ℝ := 40

noncomputable def coupon_z (price : ℝ) : ℝ := max 0 (0.3 * (price - 150))

def valid_price (price : ℝ) : Prop := price > 100

theorem coupon_difference :
  ∃ (min_price max_price : ℝ),
    valid_price min_price ∧
    valid_price max_price ∧
    (∀ price, valid_price price →
      (coupon_x price ≥ coupon_y price ∧ coupon_x price ≥ coupon_z price) →
      min_price ≤ price ∧ price ≤ max_price) ∧
    (∃ price, valid_price price ∧ 
      coupon_x price ≥ coupon_y price ∧ 
      coupon_x price ≥ coupon_z price ∧
      price = min_price) ∧
    (∃ price, valid_price price ∧ 
      coupon_x price ≥ coupon_y price ∧ 
      coupon_x price ≥ coupon_z price ∧
      price = max_price) ∧
    max_price - min_price = 250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_difference_l785_78513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_difference_l785_78512

/-- Represents a person in the circle, either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- The response a person gives to a question -/
def Response := Fin 3

/-- The circle of people -/
def Circle := Fin 300 → Person

/-- Get the neighbors of a person at a given index -/
def neighbors (c : Circle) (i : Fin 300) : Person × Person :=
  (c <| (i - 1) % 300, c <| (i + 1) % 300)

/-- The response of a person based on their type and their neighbors -/
def personResponse (p : Person) (n : Person × Person) : Response → Response :=
  sorry -- Implementation details omitted for brevity

/-- The sum of responses for all people in the circle -/
def sumResponses (c : Circle) (f : Person → Person × Person → Response) : ℕ :=
  (Finset.range 300).sum fun i => (f (c i) (neighbors c i)).val

/-- The theorem stating the impossibility of the 410 difference -/
theorem impossible_difference (c : Circle) :
  ¬∃ (f g : Person → Person × Person → Response),
    sumResponses c f = sumResponses c g + 410 := by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_difference_l785_78512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_coordinates_l785_78558

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the equation 5 · BANK = 6 · SAD -/
def EquationHolds (B A N K S D : Digit) : Prop :=
  5 * (1000 * B.val + 100 * A.val + 10 * N.val + K.val) = 6 * (100 * S.val + 10 * A.val + D.val)

/-- Represents the GPS coordinates derived from the digits -/
def GPSCoordinates (B A N K S D : Digit) : ℚ :=
  55 + (S.val * 6 + 1) / 10 + (S.val * 6) / 100 + 36 / 1000 + D.val / 10000 + B.val / 100000 + 
  (K.val / 2) / 1000000 + (N.val - 1) / 10000000 + (K.val / 2) / 100000000 + 
  (D.val - 1) / 1000000000 + (N.val - 1) / 10000000000 + (S.val * 9 + D.val) / 100000000000 + 
  A.val / 1000000000000

theorem secret_coordinates :
  ∃ (B A N K S D : Digit), 
    EquationHolds B A N K S D ∧ 
    GPSCoordinates B A N K S D = 55.543065317 := by
  sorry

#eval GPSCoordinates ⟨1, by norm_num⟩ ⟨0, by norm_num⟩ ⟨8, by norm_num⟩ ⟨6, by norm_num⟩ ⟨9, by norm_num⟩ ⟨5, by norm_num⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secret_coordinates_l785_78558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l785_78569

noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 5*x + 6) / (3*x + 1)

noncomputable def g (x : ℝ) : ℝ := x + 2/3

/-- Theorem stating that g is the oblique asymptote of f -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ N, ∀ x > N, |f x - g x| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l785_78569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_identification_l785_78509

-- Define the concept of a linear equation in one variable
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

-- Define the equations
noncomputable def eq_A (x y : ℝ) : ℝ := x - y + 1
noncomputable def eq_B (x : ℝ) : ℝ := x^2 - 4*x + 4
noncomputable def eq_C (x : ℝ) : ℝ := 1/x - 2
noncomputable def eq_D (x : ℝ) : ℝ := Real.pi * x - 2

-- State the theorem
theorem linear_equation_identification :
  ¬(is_linear_equation_one_var (λ x ↦ eq_A x 0)) ∧
  ¬(is_linear_equation_one_var eq_B) ∧
  ¬(is_linear_equation_one_var eq_C) ∧
  (is_linear_equation_one_var eq_D) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_equation_identification_l785_78509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l785_78533

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * (x - 1)
noncomputable def g (x : ℝ) : ℝ := exp x
noncomputable def t (x : ℝ) : ℝ := (1 / x) * g x

theorem function_properties (a : ℝ) :
  (∀ m : ℝ, m ≥ 1 → (∀ x ∈ Set.Icc m (m + 1), t x ≥ t m)) ∧
  (∀ m : ℝ, m > 0 → m < 1 → (∀ x ∈ Set.Icc m (m + 1), t x ≥ t 1)) ∧
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧
    (deriv (f a) x₁) * (deriv g x₂) = 1 →
    (a = 0 ∨ ((exp 1 - 1) / exp 1 < a ∧ a < (exp 2 - 1) / exp 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l785_78533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l785_78564

def functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, ∀ α : ℝ,
  functional_equation f α →
  (∀ x : ℝ, f x = x) ∧ α = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l785_78564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l785_78537

-- Define the ellipse and hyperbola
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def hyperbola (a₁ b₁ x y : ℝ) : Prop := x^2 / a₁^2 - y^2 / b₁^2 = 1

-- Define the theorem
theorem hyperbola_eccentricity 
  (a b a₁ b₁ : ℝ) 
  (h_a : a > 0) (h_b : b > 0) (h_ab : a > b)
  (h_a₁ : a₁ > 0) (h_b₁ : b₁ > 0)
  (h_e : Real.sqrt (a^2 - b^2) / a = 3/4) :
  Real.sqrt (a₁^2 + b₁^2) / a₁ = 3*Real.sqrt 2/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l785_78537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l785_78589

theorem problem_solution (x y : ℕ+) : 
  (20 : ℚ) / 100 * x.val = (15 : ℚ) / 100 * 1500 - 15 →
  (y.val : ℝ) ^ 3 - (y.val : ℝ) ^ 2 = 200 →
  x = 1050 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l785_78589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_path_exists_l785_78574

/-- Represents a square grid -/
structure Grid (n : ℕ) where
  size : n > 0

/-- Represents a cell in the grid -/
structure Cell (n : ℕ) where
  row : Fin n
  col : Fin n

/-- Represents a diagonal in a cell -/
inductive Diagonal (n : ℕ)
  | NW_SE : Cell n → Diagonal n
  | NE_SW : Cell n → Diagonal n

/-- Represents a path of diagonals -/
def DiagonalPath (n : ℕ) := List (Diagonal n)

/-- Predicate to check if a path is closed -/
def is_closed (n : ℕ) (p : DiagonalPath n) : Prop := sorry

/-- Predicate to check if a path visits all cells -/
def visits_all_cells (n : ℕ) (p : DiagonalPath n) (g : Grid n) : Prop := sorry

/-- Predicate to check if a path does not repeat any diagonal -/
def no_repeat_diagonal (n : ℕ) (p : DiagonalPath n) : Prop := sorry

theorem no_valid_path_exists (g : Grid 2019) :
  ¬∃ (p : DiagonalPath 2019), is_closed 2019 p ∧ visits_all_cells 2019 p g ∧ no_repeat_diagonal 2019 p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_path_exists_l785_78574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l785_78515

open Real

-- Define the function
noncomputable def f (x : ℝ) := Real.log (1/x - 1) / Real.log 10

-- Define the domain A
def A : Set ℝ := {x | 0 < x ∧ x < 1}

-- Define the inequality condition
def inequality_holds (m : ℝ) : Prop :=
  ∀ x ∈ A, (9*x)/(2-2*x) - m^2*x - 2*m*x > -2

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (m > 0 ∧ inequality_holds m) ↔ (0 < m ∧ m < (3 * Real.sqrt 6 - 2) / 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l785_78515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_area_l785_78532

/-- A right triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the hypotenuse -/
  s1 : ℝ
  /-- The length of the second segment of the hypotenuse -/
  s2 : ℝ
  /-- The first segment is positive -/
  s1_pos : 0 < s1
  /-- The second segment is positive -/
  s2_pos : 0 < s2
  /-- The radius is positive -/
  r_pos : 0 < r
  /-- The first leg of the triangle -/
  a : ℝ := s1 + r
  /-- The second leg of the triangle -/
  b : ℝ := s2 + r
  /-- The hypotenuse of the triangle -/
  c : ℝ := s1 + s2
  /-- The Pythagorean theorem holds for this triangle -/
  pythagorean : a^2 + b^2 = c^2

/-- The area of the triangle -/
noncomputable def triangleArea (t : InscribedCircleTriangle) : ℝ :=
  (t.a * t.b) / 2

/-- The theorem stating that the area of the specific triangle is 42 -/
theorem inscribed_circle_triangle_area :
  ∃ t : InscribedCircleTriangle, t.s1 = 6 ∧ t.s2 = 7 ∧ triangleArea t = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_area_l785_78532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_and_c_l785_78547

-- Define the function f
noncomputable def f (b c x : ℝ) : ℝ := Real.log ((2 * x^2 + b * x + c) / (x^2 + 1)) / Real.log 3

-- State the theorem
theorem sum_of_b_and_c (b c : ℝ) :
  (∀ x, 0 ≤ f b c x ∧ f b c x ≤ 1) →
  (b + c = 0 ∨ b + c = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_and_c_l785_78547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l785_78559

theorem calculation_proof :
  (((Real.sqrt 2) ^ 2 - Real.sqrt 9 + abs (Real.sqrt 3 - 2)) = (1 - Real.sqrt 3)) ∧
  ((4 * Real.sqrt (1 / 2) - (1 / 2)⁻¹ + 202 * 3 ^ 0) = (2 * Real.sqrt 2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l785_78559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l785_78588

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin (2 * x + 3 * Real.pi / 4) + 2

theorem function_transformation :
  ∀ x : ℝ, transformed_function x = original_function (x + Real.pi / 4) + 2 :=
by
  intro x
  simp [transformed_function, original_function]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l785_78588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_focus_l785_78544

/-- Parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Tangent type -/
structure Tangent where
  point : Point
  slope : ℝ

def on_line (P : Point) (L : Line) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

def on_parabola (P : Point) (C : Parabola) : Prop :=
  P.x^2 = 2 * C.p * P.y

def is_tangent (T : Tangent) (C : Parabola) : Prop :=
  on_parabola T.point C ∧ T.slope = T.point.x / C.p

noncomputable def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem parabola_tangent_focus (C : Parabola) (F : Point) (M : Point) (L : Line) 
    (A B : Point) (MA MB : Tangent) :
  on_line M L →
  L.a = 1 ∧ L.b = -1 ∧ L.c = -2 →
  F.x = 0 ∧ F.y = C.p / 2 →
  is_tangent MA C ∧ is_tangent MB C →
  MA.point = A ∧ MB.point = B →
  (∀ N : Point, on_line N L → distance A F * distance B F ≤ distance N A * distance N B) →
  distance A F * distance B F = 8 →
  C.p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_focus_l785_78544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_function_bound_l785_78557

/-- A function f with two extreme points satisfying certain conditions -/
structure ExtremeFunction where
  f : ℝ → ℝ
  a : ℝ
  x₁ : ℝ
  x₂ : ℝ
  h_extreme : ∀ x, x = x₁ ∨ x = x₂ → (deriv f x = 0)
  h_def : ∀ x, f x = Real.log x + (1/2) * x^2 + a * x
  h_sum : f x₁ + f x₂ ≤ -5

/-- Theorem stating that under given conditions, a ≤ -2√2 -/
theorem extreme_function_bound (ef : ExtremeFunction) :
  ef.a ≤ -2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_function_bound_l785_78557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_projections_slope_l785_78594

/-- Given two vectors OA and OB in a 2D plane, and a line l with an acute angle of inclination,
    prove that the slope of l is 2/5 if the projections of OA and OB on l are equal. -/
theorem equal_projections_slope (OA OB : ℝ × ℝ) (k : ℝ) :
  OA = (1, 4) →
  OB = (-3, 1) →
  (let OC := (1, k)
   OA.1 * OC.1 + OA.2 * OC.2 = OB.1 * OC.1 + OB.2 * OC.2) →
  0 < k →
  k < 1 →
  k = 2/5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_projections_slope_l785_78594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l785_78535

/-- Given two vectors a and b in ℝ², prove that the angle between them is π,
    given the conditions a + 2b = (2, -4) and 3a - b = (-8, 16). -/
theorem angle_between_vectors (a b : ℝ × ℝ) 
    (h1 : a + 2 • b = (2, -4))
    (h2 : 3 • a - b = (-8, 16)) : 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l785_78535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_conditions_l785_78561

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_lines_conditions
  (a b : Line) (α β : Plane) :
  (¬ (contained_in a α ∧ line_parallel_to_plane b β ∧ plane_perpendicular α β → perpendicular a b)) ∧
  (line_perpendicular_to_plane a α ∧ line_perpendicular_to_plane b β ∧ plane_perpendicular α β → perpendicular a b) ∧
  (contained_in a α ∧ line_perpendicular_to_plane b β ∧ parallel_planes α β → perpendicular a b) ∧
  (line_perpendicular_to_plane a α ∧ line_parallel_to_plane b β ∧ parallel_planes α β → perpendicular a b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_conditions_l785_78561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_measurements_l785_78529

/-- Cylinder measurements -/
structure Cylinder where
  d : ℝ
  h : ℝ
  r : ℝ
  hd : d = 7
  hh : h = 40
  hr : r = d / 2

/-- Volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ :=
  Real.pi * c.r^2 * c.h

/-- Curved surface area of a cylinder -/
noncomputable def curvedSurfaceArea (c : Cylinder) : ℝ :=
  2 * Real.pi * c.r * c.h

/-- Total surface area of a cylinder -/
noncomputable def totalSurfaceArea (c : Cylinder) : ℝ :=
  2 * Real.pi * c.r * c.h + 2 * Real.pi * c.r^2

/-- Combined value of volume, curved surface area, and total surface area -/
noncomputable def combinedValue (c : Cylinder) : ℝ :=
  volume c + curvedSurfaceArea c + totalSurfaceArea c

theorem cylinder_measurements (c : Cylinder) :
  combinedValue c = 1074.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_measurements_l785_78529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l785_78525

/-- The sum of the first k terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (k : ℕ) : ℝ :=
  a₁ * (1 - q^k) / (1 - q)

/-- Theorem: The sum of the first k terms of the given geometric sequence is 364 -/
theorem geometric_sequence_sum :
  ∀ (k : ℕ), k > 0 →
  let a₁ : ℝ := 1
  let q : ℝ := 3
  let aₖ : ℝ := 243
  aₖ = a₁ * q^(k - 1) →
  geometric_sum a₁ q k = 364 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l785_78525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l785_78575

/-- Calculates the length of a train given its speed, the length of a platform it crosses, and the time it takes to cross the platform. -/
noncomputable def train_length (speed : ℝ) (platform_length : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time - platform_length

/-- Theorem stating that a train traveling at 72 km/hr, crossing an 80m platform in 26 seconds, has a length of 440 meters. -/
theorem train_length_calculation :
  train_length 72 80 26 = 440 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l785_78575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_area_l785_78580

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadrilateral formed by tangent points and intersection points -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  C : ℝ × ℝ

/-- Define membership for points in a Circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Define membership for points in a Line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  l.a * x + l.b * y + l.c = 0

/-- Area of a quadrilateral (placeholder) -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- The theorem statement -/
theorem tangent_quadrilateral_area (k : ℝ) (C : Circle) (L : Line) (PACB : Quadrilateral) : 
  k > 0 → 
  C.center = (0, 1) → 
  C.radius = 1 → 
  L.a = k ∧ L.b = 1 ∧ L.c = 4 → 
  (∀ (x y : ℝ), x^2 + y^2 - 2*y = 0 ↔ C.contains (x, y)) →
  (∀ (x y : ℝ), k*x + y + 4 = 0 ↔ L.contains (x, y)) →
  (C.contains PACB.A ∧ C.contains PACB.B) →
  (L.contains PACB.P ∧ L.contains PACB.C) →
  (∀ (Q : Quadrilateral), C.contains Q.A ∧ C.contains Q.B ∧ L.contains Q.P ∧ L.contains Q.C → 
    area Q ≥ area PACB) →
  area PACB = 2 →
  k = Real.sqrt 21 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_area_l785_78580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_l785_78570

/-- SlopeAngle function (placeholder) --/
def SlopeAngle (f : ℝ → ℝ) : ℝ :=
  sorry

/-- The slope angle of a line parallel to the x-axis is 0° --/
theorem slope_angle_horizontal_line (y : ℝ → ℝ) :
  (∀ x, y x = 1) →  -- y is a constant function with value 1
  SlopeAngle y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_l785_78570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_iff_a_in_range_l785_78567

/-- The function f parameterized by a real number a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) / Real.sqrt (a * x^2 - 4 * a * x + 2)

/-- The theorem stating that the domain of f is ℝ iff a ∈ [0, 1/2) -/
theorem domain_f_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (0 ≤ a ∧ a < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_iff_a_in_range_l785_78567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_triple_product_sum_triangle_altitudes_intersect_l785_78502

-- Define a structure for a point in a plane
structure Point where
  x : Real
  y : Real

-- Define a vector between two points
def vector (A B : Point) : Point :=
  ⟨B.x - A.x, B.y - A.y⟩

-- Define the scalar triple product of three vectors
def scalarTripleProduct (u v w : Point) : Real :=
  u.x * (v.y * w.x - v.x * w.y) - u.y * (v.x * w.x - v.y * w.x)

-- Theorem for part (a)
theorem scalar_triple_product_sum (A B C D : Point) :
  scalarTripleProduct (vector A B) (vector C D) +
  scalarTripleProduct (vector B C) (vector A D) +
  scalarTripleProduct (vector C A) (vector B D) = 0 :=
by sorry

-- Define a structure for a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define an altitude of a triangle
def altitude (T : Triangle) (vertex : Point) (foot : Point) : Prop :=
  sorry

-- Theorem for part (b)
theorem triangle_altitudes_intersect (T : Triangle) :
  ∃ D : Point, 
    altitude T T.A D ∧ 
    altitude T T.B D ∧ 
    altitude T T.C D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_triple_product_sum_triangle_altitudes_intersect_l785_78502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expanded_grid_ratio_l785_78508

/-- Represents a square grid of tiles -/
structure TileGrid where
  size : Nat
  blackTiles : Nat
  whiteTiles : Nat

/-- Represents the result after expanding the grid -/
structure ExpandedGrid where
  blackTiles : Nat
  whiteTiles : Nat

/-- Expands a TileGrid by adding a surrounding border of alternating tiles -/
def expandGrid (grid : TileGrid) : ExpandedGrid :=
  { blackTiles := grid.blackTiles + (grid.size * 4 + 4) / 2,
    whiteTiles := grid.whiteTiles + (grid.size * 4 + 4) / 2 }

theorem expanded_grid_ratio (grid : TileGrid) 
  (h1 : grid.size = 5)
  (h2 : grid.blackTiles = 10)
  (h3 : grid.whiteTiles = 15) :
  let expanded := expandGrid grid
  (expanded.blackTiles : ℚ) / expanded.whiteTiles = 22 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expanded_grid_ratio_l785_78508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equals_inverse_at_five_l785_78566

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x - 5

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (x + 5) / 2

-- Theorem statement
theorem function_equals_inverse_at_five :
  ∃ x : ℝ, f x = f_inv x ∧ x = 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equals_inverse_at_five_l785_78566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_nephews_count_l785_78552

/-- Proves that Alden and Vihaan have 260 nephews in total given the problem conditions -/
theorem total_nephews_count (aldens_nephews_ten_years_ago aldens_nephews_now vihaans_nephews : ℕ) :
  aldens_nephews_ten_years_ago = 50 →
  aldens_nephews_now = 2 * aldens_nephews_ten_years_ago →
  vihaans_nephews = aldens_nephews_now + 60 →
  aldens_nephews_now + vihaans_nephews = 260 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_nephews_count_l785_78552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_cube_root_of_negative_27_l785_78572

theorem opposite_cube_root_of_negative_27 : -(Real.rpow (-27) (1/3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_cube_root_of_negative_27_l785_78572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l785_78517

/-- The slope angle of the line x - y + 3 = 0 is 45 degrees -/
theorem slope_angle_of_line (x y : ℝ) : x - y + 3 = 0 → ∃ θ : Real, θ * (180 / Real.pi) = 45 ∧ Real.tan θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l785_78517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_annual_rate_approx_l785_78598

/-- The equivalent annual interest rate for a savings account with 4.5% annual interest rate compounding quarterly -/
noncomputable def equivalent_annual_rate : ℝ :=
  ((1 + 0.045 / 4) ^ 4 - 1) * 100

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The equivalent annual rate rounded to the nearest hundredth is 4.56% -/
theorem equivalent_annual_rate_approx :
  round_to_hundredth equivalent_annual_rate = 4.56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalent_annual_rate_approx_l785_78598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l785_78545

theorem angle_relation (α β : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : β ∈ Set.Ioo 0 (π/2)) (h3 : Real.tan α = (1 + Real.sin β) / Real.cos β) : 
  2 * α - β = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l785_78545
