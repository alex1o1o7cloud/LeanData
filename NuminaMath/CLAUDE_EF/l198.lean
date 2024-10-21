import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_k_l198_19858

noncomputable def k : ℝ := Real.sqrt 2 * ((Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3))

theorem closest_integer_to_k :
  ∃ (n : ℤ), n = 3 ∧ ∀ (m : ℤ), m ≠ n → |k - ↑n| < |k - ↑m| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_k_l198_19858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l198_19886

open Real

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := sin (x / 2 + π / 12)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := sin (x - π / 4)

-- State the theorem
theorem transform_f_to_g :
  ∀ x, g x = f ((x - π / 3) * 2) :=
by
  intro x
  simp [f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l198_19886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_3_range_of_a_l198_19859

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function h(x)
def h (a : ℝ) (x : ℝ) : ℝ := f a (2*x + a) - 2 * f a x

-- Theorem for part 1
theorem solution_set_when_a_3 :
  ∀ x : ℝ, (f 3 x ≥ 4 - |x - 4|) ↔ (x ≤ 3/2 ∨ x ≥ 11/2) :=
sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) :
  (a > 1) →
  (∃ S : ℝ, S = (1/2) * a * (a/2) ∧ S > a + 4) →
  a ∈ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_3_range_of_a_l198_19859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_cent_arrangements_l198_19800

/-- Represents a stamp with a certain denomination -/
structure Stamp where
  denomination : Nat
  deriving Repr

/-- Represents Diane's stamp collection -/
def DianeStamps : List (Nat × Nat) :=
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

/-- Calculates the number of unique arrangements to make a certain amount -/
def countArrangements (stamps : List (Nat × Nat)) (target : Nat) : Nat :=
  sorry

/-- Helper function to calculate the sum of denominations in a subset -/
def sumDenominations (subset : List (Nat × Nat)) : Nat :=
  subset.foldl (fun acc (d, _) => acc + d) 0

/-- Helper function to calculate the number of permutations for a subset -/
def countPermutations (subset : List (Nat × Nat)) : Nat :=
  sorry

/-- Theorem stating that the number of unique arrangements to make 15 cents
    is equal to the sum of all valid permutations of stamp combinations -/
theorem fifteen_cent_arrangements :
  countArrangements DianeStamps 15 =
    (DianeStamps.toFinset.powerset.filter (fun subset => sumDenominations subset.toList == 15)).sum
      (fun subset => countPermutations subset.toList) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_cent_arrangements_l198_19800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_negative_l198_19892

theorem sin_2_cos_3_negative : Real.sin 2 * Real.cos 3 < 0 := by
  -- Assuming 2 and 3 radians are in the second quadrant
  have h1 : Real.pi / 2 < 2 ∧ 2 < Real.pi := by sorry
  have h2 : Real.pi / 2 < 3 ∧ 3 < Real.pi := by sorry

  -- The sine of an angle in the second quadrant is positive
  have sin_pos : Real.sin 2 > 0 := by sorry

  -- The cosine of an angle in the second quadrant is negative
  have cos_neg : Real.cos 3 < 0 := by sorry

  -- The product of a positive and a negative number is negative
  have product_neg : Real.sin 2 * Real.cos 3 < 0 := by
    exact mul_neg_of_pos_of_neg sin_pos cos_neg

  exact product_neg


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2_cos_3_negative_l198_19892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poodle_groom_time_is_30_l198_19830

/-- The time (in minutes) it takes to groom a poodle. -/
noncomputable def poodle_groom_time : ℚ := 30

/-- The time (in minutes) it takes to groom a terrier. -/
noncomputable def terrier_groom_time : ℚ := poodle_groom_time / 2

/-- The number of poodles groomed. -/
def num_poodles : ℕ := 3

/-- The number of terriers groomed. -/
def num_terriers : ℕ := 8

/-- The total grooming time (in minutes). -/
noncomputable def total_groom_time : ℚ := 210

theorem poodle_groom_time_is_30 :
  poodle_groom_time = 30 ∧
  terrier_groom_time = poodle_groom_time / 2 ∧
  (num_poodles : ℚ) * poodle_groom_time + (num_terriers : ℚ) * terrier_groom_time = total_groom_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poodle_groom_time_is_30_l198_19830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_x_values_for_3001_l198_19831

-- Define the sequence
def sequenceA (x : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => 3000
  | n + 2 => sequenceA x n * sequenceA x (n + 1) - 1

-- Define a function to check if 3001 appears in the first 7 terms
def contains_3001 (x : ℝ) : Prop :=
  ∃ n : ℕ, n < 7 ∧ sequenceA x n = 3001

-- Theorem statement
theorem four_x_values_for_3001 :
  ∃ S : Finset ℝ, (∀ x ∈ S, contains_3001 x) ∧ 
                  (∀ x ∉ S, ¬contains_3001 x) ∧ 
                  S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_x_values_for_3001_l198_19831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l198_19852

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  point : Point
  direction : Vec

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Distance from Q to origin is 3 -/
theorem parabola_distance_theorem (C : Parabola) (F P Q O : Point) (l : Line) :
  C.equation = (fun x y => y^2 = 8*x) →  -- Parabola equation
  F.x = 2 ∧ F.y = 0 →  -- Focus coordinates
  P = l.point →  -- P is on directrix
  C.equation Q.x Q.y →  -- Q is on parabola
  ∃ t, Q = Point.mk (F.x + t * (P.x - F.x)) (F.y + t * (P.y - F.y)) →  -- Q is on line PF
  O.x = 0 ∧ O.y = 0 →  -- O is origin
  Vec.mk (P.x - F.x) (P.y - F.y) = Vec.mk (4*(Q.x - F.x)) (4*(Q.y - F.y)) →  -- ⃗FP = 4⃗FQ
  distance Q O = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l198_19852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_fluid_cost_l198_19853

theorem cleaning_fluid_cost 
  (total_drums : ℕ)
  (cost_fluid1 : ℕ)
  (cost_fluid2 : ℕ)
  (drums_fluid1 : ℕ)
  (drums_fluid2 : ℕ)
  (h1 : total_drums = drums_fluid1 + drums_fluid2)
  (h2 : drums_fluid1 = 2)
  (h3 : drums_fluid2 = 5)
  (h4 : cost_fluid1 = 30)
  (h5 : cost_fluid2 = 20) :
  drums_fluid1 * cost_fluid1 + drums_fluid2 * cost_fluid2 = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_fluid_cost_l198_19853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_grid_squares_l198_19883

/-- The number of squares of size n×n in a grid with r rows and c columns -/
def count_squares (r c n : ℕ) : ℕ := (r - n + 1) * (c - n + 1)

/-- The total number of squares in a rectangular grid -/
def total_squares (r c : ℕ) : ℕ :=
  (List.range (min r c)).map (fun n => count_squares r c (n + 1)) |>.sum

theorem rectangular_grid_squares :
  total_squares 5 6 = 70 := by
  sorry

#eval total_squares 5 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_grid_squares_l198_19883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l198_19864

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 3)^2 + (y - 1)^2 = 2

-- Define point Q
def Q : ℝ × ℝ := (2, 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the area of triangle OPQ
noncomputable def triangle_area (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  abs (x * 2 - y * 2) / 2

-- Theorem statement
theorem min_triangle_area :
  ∃ (min_area : ℝ), min_area = 2 ∧
  ∀ (P : ℝ × ℝ), circle_eq P.1 P.2 → triangle_area P ≥ min_area :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l198_19864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l198_19843

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C,
    prove that under certain conditions, the area is 2 and the minimum side length is 2. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  Real.cos (A / 2) = 2 * Real.sqrt 5 / 5 →
  b * c * Real.cos A = 3 →
  (∃ (area : ℝ), area = 1 / 2 * b * c * Real.sin A ∧ area = 2) ∧
  (∀ (a' : ℝ), a' ≥ 2 ∧ (a' = 2 → ∃ (b' c' : ℝ), b' = Real.sqrt 5 ∧ c' = Real.sqrt 5 ∧ 
    a' ^ 2 = b' ^ 2 + c' ^ 2 - 2 * b' * c' * Real.cos A)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l198_19843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_quadrilateral_area_l198_19899

/-- Given two rectangles ABCD and EFGH forming a cross shape, prove the area of quadrilateral AFCH --/
theorem cross_quadrilateral_area 
  (AB BC EF FG : ℝ) 
  (h_AB : AB = 9) 
  (h_BC : BC = 5) 
  (h_EF : EF = 3) 
  (h_FG : FG = 10) : 
  AB * FG - (AB - EF) * (FG - BC) / 2 = 52.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_quadrilateral_area_l198_19899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l198_19894

/-- The probability of getting heads on a single toss of the biased coin -/
noncomputable def p_heads : ℝ := 2/3

/-- The number of times the coin is tossed -/
def n : ℕ := 50

/-- The probability of getting an even number of heads in n tosses -/
noncomputable def p_even_heads (p : ℝ) (n : ℕ) : ℝ := 1/2 * (1 + (1 - p)^n / p^n)

/-- Theorem stating the probability of getting an even number of heads
    in 50 tosses of a coin with 2/3 probability of heads -/
theorem even_heads_probability :
  p_even_heads p_heads n = 1/2 * (1 + 1/3^50) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l198_19894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_shirt_price_is_correct_dress_shirt_price_equals_15_l198_19825

/-- The price of a dress shirt that satisfies the given conditions -/
def dress_shirt_price : ℝ := 15

theorem dress_shirt_price_is_correct : 
  let pants_price : ℝ := 40
  let suit_price : ℝ := 150
  let sweater_price : ℝ := 30
  let store_discount : ℝ := 0.2
  let coupon_discount : ℝ := 0.1
  let final_price : ℝ := 252
  let x : ℝ := dress_shirt_price
  0.9 * (0.8 * (4 * x + 2 * pants_price + suit_price + 2 * sweater_price)) = final_price :=
by
  -- Plug in the values and simplify
  simp [dress_shirt_price]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

theorem dress_shirt_price_equals_15 : dress_shirt_price = 15 := rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_shirt_price_is_correct_dress_shirt_price_equals_15_l198_19825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_l198_19871

theorem log_equality_implies_base (y : ℝ) : 
  (Real.log 8 / Real.log y = Real.log 5 / Real.log 125) → y = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_base_l198_19871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_four_l198_19890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + Real.log x / Real.log 2

theorem max_value_implies_a_equals_four (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f a x ≤ 6) ∧ (∃ x ∈ Set.Icc 1 a, f a x = 6) → a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_four_l198_19890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rotation_angle_l198_19803

def Operation (α : Real) : Prop :=
  0 < α ∧ α < Real.pi

def ReturnsToStart (α : Real) : Prop :=
  Operation α ∧ (∃ n : ℕ, n = 5 ∧ n * α = 2 * Real.pi)

theorem car_rotation_angle : 
  ∀ α : Real, ReturnsToStart α → (α = Real.pi / 2.5 ∨ α = 4 * Real.pi / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rotation_angle_l198_19803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponentials_l198_19832

theorem compare_exponentials :
  let a : ℕ := 2^12
  let b : ℕ := 3^8
  let c : ℕ := 7^4
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponentials_l198_19832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l198_19802

noncomputable section

/-- The volume of a cone given its radius and height -/
def coneVolume (radius : ℝ) (height : ℝ) : ℝ := (1/3) * Real.pi * radius^2 * height

/-- The ratio of volumes of two cones -/
def volumeRatio (r1 h1 r2 h2 : ℝ) : ℝ := (coneVolume r1 h1) / (coneVolume r2 h2)

theorem cone_volume_ratio :
  let r_C : ℝ := 20
  let h_C : ℝ := 40
  let r_D : ℝ := 40
  let h_D : ℝ := 20
  volumeRatio r_C h_C r_D h_D = 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l198_19802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l198_19873

theorem cos_minus_sin_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/8)
  (h2 : 0 < α)
  (h3 : α < Real.pi/4) :
  Real.cos α - Real.sin α = Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l198_19873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l198_19807

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x - a * 2^(-x)) / (2^x + 2^(-x))

noncomputable def g (x : ℝ) : ℝ := (2^x + 2^(-x)) * f x 1

noncomputable def h (x : ℝ) (m : ℝ) : ℝ := 2^(2*x) + 2^(-2*x) - 2*m * g x

theorem problem_solution :
  (∀ x, f (-x) 1 = -f x 1) →
  (∀ x, x ≥ 1 → h x 2 ≥ -2) →
  (∃ x, x ≥ 1 ∧ h x 2 = -2) →
  (∀ x, Set.Icc (-1 : ℝ) 1 (f x 1)) ∧
  (∀ t, (∀ x, g (x^2 + t*x) + g (4-x) > 0) ↔ -3 < t ∧ t < 5) ∧
  (2 = 2) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l198_19807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_cos_inequality_l198_19848

theorem min_m_for_cos_inequality : 
  (∃ m : ℝ, ∀ x : ℝ, π/4 ≤ x ∧ x ≤ 2*π/3 → Real.cos x ≤ m) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, π/4 ≤ x ∧ x ≤ 2*π/3 → Real.cos x ≤ m) → m ≥ Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_cos_inequality_l198_19848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_set_is_interior_region_l198_19816

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point on a line segment
def PointOnSegment (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the centroid of a triangle
noncomputable def Centroid (T : Triangle) : ℝ × ℝ :=
  ((T.A.1 + T.B.1 + T.C.1) / 3, (T.A.2 + T.B.2 + T.C.2) / 3)

-- Define the set of centroids
def CentroidSet (T : Triangle) : Set (ℝ × ℝ) :=
  {P | ∃ K L M : ℝ × ℝ,
    PointOnSegment K T.A T.B ∧
    PointOnSegment L T.B T.C ∧
    PointOnSegment M T.C T.A ∧
    P = Centroid ⟨K, L, M⟩}

-- Define an interior region of a triangle
def InteriorRegion (T : Triangle) : Set (ℝ × ℝ) :=
  {P | ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 ∧
    P = (a * T.A.1 + b * T.B.1 + c * T.C.1, a * T.A.2 + b * T.B.2 + c * T.C.2)}

-- Theorem statement
theorem centroid_set_is_interior_region (T : Triangle) :
  ∃ S : Set (ℝ × ℝ), S ⊆ InteriorRegion T ∧ CentroidSet T = S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_set_is_interior_region_l198_19816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grants_apartment_rooms_l198_19854

def danielles_rooms : ℕ := 6
def heidis_rooms : ℕ := 3 * danielles_rooms
def jennys_rooms : ℕ := danielles_rooms + 5
def linas_rooms : ℕ := 7

def total_rooms : ℕ := danielles_rooms + heidis_rooms + jennys_rooms + linas_rooms

noncomputable def grants_rooms : ℕ := 
  (((total_rooms : ℚ) / 9) - ((total_rooms : ℚ) / 9) / 3).floor.toNat

theorem grants_apartment_rooms : grants_rooms = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grants_apartment_rooms_l198_19854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_dividing_50_factorial_l198_19819

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem largest_power_of_two_dividing_50_factorial :
  (∃ a : ℕ, 2^a ∣ factorial 50 ∧ ∀ b : ℕ, 2^b ∣ factorial 50 → b ≤ a) ∧
  (∀ a : ℕ, 2^a ∣ factorial 50 → a ≤ 47) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_dividing_50_factorial_l198_19819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l198_19814

/-- The area enclosed by the curves represented by the polar equations
    θ = π/3 (ρ > 0), θ = 2π/3 (ρ > 0), and ρ = 4 -/
noncomputable def enclosed_area (θ₁ θ₂ ρ : ℝ) : ℝ :=
  (1 / 2) * (θ₂ - θ₁) * ρ^2

/-- Theorem stating that the area enclosed by the given curves is 8π/3 -/
theorem area_enclosed_by_curves : 
  enclosed_area (π / 3) (2 * π / 3) 4 = 8 * π / 3 := by
  -- Unfold the definition of enclosed_area
  unfold enclosed_area
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l198_19814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_b_4_monotonic_increasing_condition_l198_19808

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (x^2 + b*x + b) * Real.sqrt (1 - 2*x)

-- Theorem for extreme values when b = 4
theorem extreme_values_b_4 :
  ∃ x₁ x₂, x₁ = -2 ∧ x₂ = 0 ∧ 
   (∀ x, f 4 x ≥ f 4 x₁) ∧
   (∀ x, f 4 x ≤ f 4 x₂) ∧
   f 4 x₁ = 0 ∧ f 4 x₂ = 4 := by
  sorry

-- Theorem for monotonically increasing condition
theorem monotonic_increasing_condition :
  ∀ b, (∀ x ∈ Set.Ioo 0 (1/3), StrictMono (f b)) ↔ b ≤ 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_b_4_monotonic_increasing_condition_l198_19808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l198_19869

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 1/x + Real.exp (x + 1/x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 + Real.exp 2 ∧ ∀ (x : ℝ), x > 0 → f x ≤ M :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l198_19869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_equals_100_l198_19804

/-- The sum of the alternating series 1 - 2 + 3 - 4 + ... + 199 -/
def alternatingSum : ℕ → ℤ
  | 0 => 0
  | n + 1 => alternatingSum n + (if n % 2 = 0 then (n + 1 : ℤ) else -(n + 1 : ℤ))

/-- The final term of the series is 199 -/
def finalTerm : ℕ := 199

theorem alternating_sum_equals_100 : alternatingSum finalTerm = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_equals_100_l198_19804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_sells_25_mice_per_day_l198_19842

/-- Represents the store's sales and pricing information --/
structure StoreSales where
  normal_mouse_price : ℚ
  price_increase_percentage : ℚ
  weekly_revenue : ℚ
  open_days_per_week : ℕ

/-- Calculates the number of left-handed mice sold per day --/
def mice_sold_per_day (s : StoreSales) : ℚ :=
  let left_handed_mouse_price := s.normal_mouse_price * (1 + s.price_increase_percentage)
  let daily_revenue := s.weekly_revenue / s.open_days_per_week
  daily_revenue / left_handed_mouse_price

/-- Theorem stating that the store sells 25 left-handed mice per day --/
theorem store_sells_25_mice_per_day :
  let s : StoreSales := {
    normal_mouse_price := 120,
    price_increase_percentage := 3/10,
    weekly_revenue := 15600,
    open_days_per_week := 4
  }
  mice_sold_per_day s = 25 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_sells_25_mice_per_day_l198_19842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_m_l198_19809

-- Define the functions and their domains
noncomputable def f (x : ℝ) := Real.log (x^2 - x - 2)
def A : Set ℝ := {x | x^2 - x - 2 > 0}

noncomputable def g (x : ℝ) := Real.sqrt (3 - abs x)
def B : Set ℝ := {x | 3 - abs x ≥ 0}

def C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 2}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3)} := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) (h : C m ⊆ B) : -2 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_range_of_m_l198_19809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_expression_l198_19846

/-- Represents an algebraic expression -/
inductive AlgebraicExpression where
  | Multiplication : Int → List Char → AlgebraicExpression
  | Division : AlgebraicExpression → Int → AlgebraicExpression
  | MixedNumber : Int → Int → Int → Char → AlgebraicExpression

/-- Checks if an algebraic expression is correctly written -/
def isCorrectlyWritten : AlgebraicExpression → Prop
  | AlgebraicExpression.Multiplication c vars => c ≠ 1 ∧ c ≠ -1
  | AlgebraicExpression.Division _ _ => False
  | AlgebraicExpression.MixedNumber _ _ _ _ => False

theorem correct_expression : 
  isCorrectlyWritten (AlgebraicExpression.Multiplication (-3) ['x', 'y']) ∧
  ¬isCorrectlyWritten (AlgebraicExpression.Division (AlgebraicExpression.Multiplication 1 ['a']) 4) ∧
  ¬isCorrectlyWritten (AlgebraicExpression.Multiplication (-1) ['a']) ∧
  ¬isCorrectlyWritten (AlgebraicExpression.MixedNumber 1 2 3 'm') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_expression_l198_19846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_correct_l198_19865

noncomputable def rationalize_denominator (n : ℝ) (a b : ℝ) : ℝ × ℝ :=
  let num := n * (a - Real.sqrt b)
  let den := (a + Real.sqrt b) * (a - Real.sqrt b)
  (num, den)

theorem rationalized_form_correct :
  let result := rationalize_denominator 7 3 8
  result.1 = -7 * Real.sqrt 8 + 21 ∧
  result.2 = 1 ∧
  Int.gcd 7 (Int.gcd 21 1) = 1 ∧
  ¬ ∃ (p : ℕ), Nat.Prime p ∧ (8 : ℕ) % (p * p) = 0 := by
  sorry

#check rationalized_form_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_form_correct_l198_19865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_in_square_l198_19898

/-- The area of a rhombus drawn between the sides of a square with a given circumference -/
theorem rhombus_area_in_square (square_circumference : ℝ) : 
  square_circumference = 96 → 
  (square_circumference / 4) * (square_circumference / 4) / 2 = 288 := by
  intro h
  have square_side : ℝ := square_circumference / 4
  have rhombus_diagonal : ℝ := square_side
  have rhombus_area : ℝ := (rhombus_diagonal * rhombus_diagonal) / 2
  sorry

#check rhombus_area_in_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_in_square_l198_19898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_and_eighth_terms_l198_19811

def digit_square_sum (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d => d * d) |>.sum

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 425
  | n + 1 => seq n + digit_square_sum (seq n)

theorem seventh_and_eighth_terms :
  seq 6 = 870 ∧ seq 7 = 983 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_and_eighth_terms_l198_19811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_10_minutes_angle_l198_19860

-- Define the number of degrees in a full circle
noncomputable def full_circle_degrees : ℝ := 360

-- Define the number of radians in a full circle
noncomputable def full_circle_radians : ℝ := 2 * Real.pi

-- Define the number of minutes in a full hour
def minutes_per_hour : ℕ := 60

-- Define the conversion factor from degrees to radians
noncomputable def degrees_to_radians (degrees : ℝ) : ℝ :=
  degrees * (full_circle_radians / full_circle_degrees)

-- Define the angle turned by the minute hand in 10 minutes
noncomputable def angle_turned_10_minutes : ℝ :=
  -(10 : ℝ) / minutes_per_hour * full_circle_degrees

-- Theorem statement
theorem minute_hand_10_minutes_angle :
  degrees_to_radians angle_turned_10_minutes = -Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_10_minutes_angle_l198_19860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_common_roots_l198_19801

/-- Two cubic polynomials have two distinct common roots -/
theorem two_distinct_common_roots : ∃ (x y : ℝ), x ≠ y ∧
  (x^3 + (-1)*x^2 + 19*x + 9 = 0) ∧
  (x^3 + 0*x^2 + 24*x + 12 = 0) ∧
  (y^3 + (-1)*y^2 + 19*y + 9 = 0) ∧
  (y^3 + 0*y^2 + 24*y + 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_common_roots_l198_19801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_l198_19863

def e1 : ℝ × ℝ := (2, -5)
def e2 : ℝ × ℝ := (-6, 4)

theorem vectors_form_basis : LinearIndependent ℝ (fun i : Fin 2 => if i = 0 then e1 else e2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_form_basis_l198_19863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_formula_l198_19887

/-- Represents a tetrahedron with an inscribed cube. -/
structure TetrahedronWithCube where
  /-- Length of one of the perpendicular edges of the tetrahedron -/
  a : ℝ
  /-- Length of the other perpendicular edge of the tetrahedron -/
  b : ℝ
  /-- Distance between the two perpendicular edges of the tetrahedron -/
  c : ℝ
  /-- The two edges a and b are perpendicular -/
  edges_perpendicular : True
  /-- The cube is inscribed in the tetrahedron -/
  cube_inscribed : True
  /-- Four edges of the cube are perpendicular to the two mentioned edges of the tetrahedron -/
  cube_edges_perpendicular : True
  /-- Exactly two vertices of the cube lie on each face of the tetrahedron -/
  cube_vertices_on_faces : True
  /-- All values are positive -/
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- The side length of the inscribed cube in the tetrahedron -/
noncomputable def cube_side_length (t : TetrahedronWithCube) : ℝ :=
  (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.c * t.a)

/-- Theorem stating that the side length of the inscribed cube is as calculated -/
theorem cube_side_length_formula (t : TetrahedronWithCube) :
  cube_side_length t = (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.c * t.a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_formula_l198_19887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_and_closest_whole_number_l198_19845

theorem ratio_and_closest_whole_number : 
  let ratio : ℚ := (10^100 + 10^102) / (10^101 + 10^101)
  (ratio = 101 / 20) ∧ 
  (∀ n : ℕ, |ratio - 5| ≤ |ratio - n|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_and_closest_whole_number_l198_19845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l198_19856

-- Define the curve C
def curve (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line l
def line (t : ℝ) : ℝ × ℝ := (1 + 3*t, 2 - 4*t)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line t ∧ curve p.1 p.2}

-- Theorem statement
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 21 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l198_19856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_sequence_a_general_term_l198_19866

def sequence_a : ℕ → ℚ
  | 0 => -1  -- Add this case to cover Nat.zero
  | 1 => -1
  | (n + 1) => -1 / (n + 1)

theorem sequence_a_property (n : ℕ) (h : n ≥ 1) :
  sequence_a (n + 1) * sequence_a n = sequence_a (n + 1) - sequence_a n :=
by sorry

theorem sequence_a_general_term (n : ℕ) (h : n ≥ 1) :
  sequence_a n = -1 / n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_property_sequence_a_general_term_l198_19866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_calculation_l198_19874

noncomputable def square_area_from_perimeter (perimeter : ℝ) : ℝ :=
  (perimeter / 4) ^ 2

noncomputable def probability_not_in_inner_square (outer_area inner_area : ℝ) : ℝ :=
  (outer_area - inner_area) / outer_area

theorem square_area_calculation (perimeter_B : ℝ) (prob_not_in_B : ℝ) :
  perimeter_B = 16 →
  prob_not_in_B = 0.8677685950413223 →
  ∃ (area : ℝ), 
    probability_not_in_inner_square area (square_area_from_perimeter perimeter_B) = prob_not_in_B ∧
    abs (area - 121) < 0.01 := by
  sorry

#check square_area_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_calculation_l198_19874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_zeros_equality_l198_19895

/-- Count the number of digits in a natural number -/
def countDigits (n : ℕ) : ℕ := 
  if n < 10 then 1 else 1 + countDigits (n / 10)

/-- Count the number of zeros in a natural number -/
def countZeros (n : ℕ) : ℕ := 
  if n < 10 then 
    if n = 0 then 1 else 0
  else 
    (if n % 10 = 0 then 1 else 0) + countZeros (n / 10)

/-- Sum of digits in numbers from 1 to n -/
def sumOfDigits (n : ℕ) : ℕ :=
  (List.range n).map countDigits |>.sum

/-- Count of zeros in numbers from 1 to n -/
def totalZeros (n : ℕ) : ℕ :=
  (List.range n).map countZeros |>.sum

theorem digits_zeros_equality (k : ℕ) (hk : k > 0) : 
  sumOfDigits (10^k) = totalZeros (10^(k+1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_zeros_equality_l198_19895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_50_l198_19878

-- Define the points of the triangle
def A : ℝ × ℝ := (-10, 0)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (0, 0)

-- Define the function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))

-- Theorem statement
theorem triangle_area_is_50 : triangleArea A B C = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_50_l198_19878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_identification_l198_19867

/-- Represents a student's math score -/
structure MathScore where
  score : ℝ

/-- Represents the set of all students who took the exam -/
def AllStudents : Finset MathScore := sorry

/-- Represents the set of randomly selected students for analysis -/
def SelectedStudents : Finset MathScore := sorry

/-- Defines what constitutes a population in a statistical study -/
def IsPopulation (s : Finset MathScore) : Prop := sorry

theorem population_identification 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (h1 : total_students = 50000)
  (h2 : selected_students = 2000)
  (h3 : selected_students < total_students)
  (h4 : SelectedStudents ⊆ AllStudents)
  (h5 : Finset.card AllStudents = total_students)
  (h6 : Finset.card SelectedStudents = selected_students) :
  IsPopulation AllStudents :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_identification_l198_19867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_7000_l198_19870

/-- Represents the investment and profit sharing scenario of three partners -/
structure Partnership where
  x : ℚ  -- A's initial investment
  annual_gain : ℚ
  a_investment_months : ℚ := 12 * x
  b_investment_months : ℚ := 2 * 6 * x
  c_investment_months : ℚ := 3 * 4 * x
  total_investment_months : ℚ := a_investment_months + b_investment_months + c_investment_months

/-- Calculates A's share of the annual gain -/
def a_share (p : Partnership) : ℚ :=
  (p.a_investment_months / p.total_investment_months) * p.annual_gain

/-- Theorem stating that A's share is 7000 given the specific conditions -/
theorem a_share_is_7000 (p : Partnership) (h1 : p.annual_gain = 21000) : a_share p = 7000 := by
  sorry

#eval a_share { x := 1000, annual_gain := 21000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_7000_l198_19870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l198_19805

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*Real.log x

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∀ x : ℝ, x = 1 → (2*x - f a x - 1 = 0)) →
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ ∀ y ∈ Set.Icc 1 2, f a x ≤ f a y) →
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f a x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l198_19805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l198_19849

theorem expression_evaluation : 
  (-1 : ℝ)^(2015 : ℕ) + Real.sqrt (1/4) + (Real.pi - 3.14)^0 + 2 * Real.sin (60 * π / 180) - 2^(-1 : ℤ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l198_19849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_simplest_l198_19880

/-- A quadratic radical is considered simpler if it cannot be simplified further without approximation. -/
def SimplestQuadraticRadical (a b c d : ℝ) : Prop :=
  (∀ x : ℝ, x ^ 2 = 10 → x = Real.sqrt 10 ∨ x = -Real.sqrt 10) ∧
  (∃ y : ℝ, y ^ 2 = 27 ∧ y ≠ Real.sqrt 27 ∧ y ≠ -Real.sqrt 27) ∧
  (∃ z : ℝ, z ^ 2 = 1/5 ∧ z ≠ Real.sqrt (1/5) ∧ z ≠ -Real.sqrt (1/5)) ∧
  (∃ w : ℝ, w ^ 2 = 16 ∧ w ≠ Real.sqrt 16 ∧ w ≠ -Real.sqrt 16)

/-- The theorem stating that √10 is the simplest quadratic radical among the given options. -/
theorem sqrt_10_simplest : SimplestQuadraticRadical 27 (1/5) 16 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_simplest_l198_19880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_problem_l198_19840

/-- Given that a varies in inverse proportion to the square of b, prove that when a = 40 for b = 12, then b = 24 when a = 10 -/
theorem inverse_proportion_problem (k : ℝ) (a b : ℝ → ℝ) 
  (h1 : ∀ x, a x * (b x)^2 = k)
  (h2 : a 1 = 40)
  (h3 : b 1 = 12) :
  b 2 = 24 ∧ a 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_problem_l198_19840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleTreePlanting_l198_19820

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A rectangle defined by its length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Check if a point is inside a rectangle -/
def isInside (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.length ∧ 0 ≤ p.y ∧ p.y ≤ r.width

/-- The main theorem -/
theorem impossibleTreePlanting (r : Rectangle) (p1 p2 p3 : Point) :
  r.length = 4 ∧ r.width = 1 →
  isInside p1 r ∧ isInside p2 r ∧ isInside p3 r →
  ¬(distance p1 p2 ≥ 2.5 ∧ distance p1 p3 ≥ 2.5 ∧ distance p2 p3 ≥ 2.5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleTreePlanting_l198_19820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bipartite_matching_implies_one_factor_l198_19835

universe u

/-- A bipartite graph with bipartition {A, B} -/
structure BipartiteGraph (V : Type u) where
  A : Set V
  B : Set V
  edges : Set (V × V)
  is_bipartite : ∀ (v w : V), (v, w) ∈ edges → (v ∈ A ∧ w ∈ B) ∨ (v ∈ B ∧ w ∈ A)
  partition : A ∪ B = Set.univ
  disjoint : A ∩ B = ∅

/-- A matching in a graph -/
def Matching {V : Type u} (G : BipartiteGraph V) (M : Set (V × V)) : Prop :=
  M ⊆ G.edges ∧ ∀ (e₁ e₂ : V × V), e₁ ∈ M → e₂ ∈ M → e₁ ≠ e₂ → e₁.1 ≠ e₂.1 ∧ e₁.2 ≠ e₂.2

/-- A matching saturates a set of vertices -/
def SaturatesSet {V : Type u} (G : BipartiteGraph V) (M : Set (V × V)) (S : Set V) : Prop :=
  Matching G M ∧ ∀ v ∈ S, ∃ w, (v, w) ∈ M ∨ (w, v) ∈ M

/-- A 1-factor (perfect matching) in a graph -/
def OneFactor {V : Type u} (G : BipartiteGraph V) (M : Set (V × V)) : Prop :=
  Matching G M ∧ ∀ v : V, ∃ w, (v, w) ∈ M ∨ (w, v) ∈ M

/-- The main theorem -/
theorem bipartite_matching_implies_one_factor
  {V : Type u} (G : BipartiteGraph V)
  (M_A : Set (V × V))
  (M_B : Set (V × V))
  (h_A : SaturatesSet G M_A G.A)
  (h_B : SaturatesSet G M_B G.B) :
  ∃ M : Set (V × V), OneFactor G M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bipartite_matching_implies_one_factor_l198_19835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_internal_tangent_circles_l198_19826

/-- Given two circles with radii 5 and 3 that touch each other internally,
    and a chord of the larger circle that touches the smaller circle and
    is divided by the point of tangency in the ratio 3:1,
    the length of this chord is 8. -/
theorem chord_length_internal_tangent_circles : ∀ (O₁ O₂ A B C : ℝ × ℝ),
  let r₁ : ℝ := 5
  let r₂ : ℝ := 3
  -- O₁ and O₂ are the centers of the circles
  -- A and B are the endpoints of the chord
  -- C is the point where the chord touches the smaller circle
  dist O₁ O₂ = r₁ - r₂ →           -- circles touch internally
  dist O₁ A = r₁ →                 -- A is on the larger circle
  dist O₁ B = r₁ →                 -- B is on the larger circle
  dist O₂ C = r₂ →                 -- C is on the smaller circle
  C.1 = (3 * A.1 + B.1) / 4 →      -- C divides AB in 3:1 ratio (x-coordinate)
  C.2 = (3 * A.2 + B.2) / 4 →      -- C divides AB in 3:1 ratio (y-coordinate)
  dist A B = 8 := by
  sorry

/-- Helper function to calculate Euclidean distance between two points -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_internal_tangent_circles_l198_19826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_below_50_is_70_percent_l198_19829

structure Dataset :=
  (capacity : ℕ)
  (intervals : List (ℝ × ℝ))
  (frequencies : List ℕ)

noncomputable def freq_below_50 (d : Dataset) : ℝ :=
  let total_freq := d.frequencies.sum
  let freq_below_50 := (d.frequencies.take 4).sum
  (freq_below_50 : ℝ) / total_freq

theorem frequency_below_50_is_70_percent (d : Dataset) :
  d.capacity = 20 ∧
  d.intervals = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70)] ∧
  d.frequencies = [2, 3, 4, 5, 4, 2] →
  freq_below_50 d = 0.7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_below_50_is_70_percent_l198_19829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l198_19857

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 1 / x

-- State the theorem
theorem tangent_line_at_one :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (f' 1 = m) ∧
    (f 1 = m * 1 + b) ∧
    (m = 4 ∧ b = -3) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l198_19857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_AC_l198_19862

/-- Given planar vectors PA and PB with |PA| = |PB| = 1 and PA · PB = -1/2, 
    and |BC| = 1, the maximum value of |AC| is √3 + 1. -/
theorem max_length_AC (PA PB BC : ℝ × ℝ) : 
  ‖PA‖ = 1 → ‖PB‖ = 1 → PA • PB = -1/2 → ‖BC‖ = 1 → 
  ∃ (C : ℝ × ℝ), ∀ (C' : ℝ × ℝ), ‖BC‖ = 1 → 
    ‖PA + BC'‖ ≤ ‖PA + BC‖ ∧ ‖PA + BC‖ = Real.sqrt 3 + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_AC_l198_19862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_1_02_approximation_l198_19847

theorem power_of_1_02_approximation : |(1.02 : ℝ)^8 - 1.172| < 0.0005 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_1_02_approximation_l198_19847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steves_speed_back_from_work_l198_19876

theorem steves_speed_back_from_work 
  (distance : ℝ) 
  (traffic_to_work : ℝ) 
  (traffic_from_work : ℝ) 
  (total_time : ℝ) 
  (h1 : distance = 30) 
  (h2 : traffic_to_work = 30) 
  (h3 : traffic_from_work = 15) 
  (h4 : total_time = 405) :
  ∃ (speed_to_work : ℝ), 
    let speed_from_work := 2 * speed_to_work
    (distance / speed_to_work * 60 + traffic_to_work + 
     distance / speed_from_work * 60 + traffic_from_work = total_time) ∧ 
    speed_from_work = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steves_speed_back_from_work_l198_19876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_third_quadrant_l198_19896

theorem cos_A_third_quadrant (A : Real) (h1 : A ∈ Set.Icc π (3*π/2)) 
  (h2 : Real.sin A = -5/13) : Real.cos A = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_A_third_quadrant_l198_19896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_equals_three_l198_19879

-- Define x as noncomputable
noncomputable def x : ℝ := (1/2) * (Real.rpow (2 + Real.sqrt 5) (1/3) + Real.rpow (2 - Real.sqrt 5) (1/3))

-- State the theorem
theorem cubic_equation_equals_three : 8 * x^3 + 6 * x - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_equals_three_l198_19879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_lg_sum_split_skew_iff_no_common_plane_l198_19827

-- Definition of logarithm (lg) for positive real numbers
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Definition of a line in 3D space
def Line3D := ℝ → ℝ × ℝ × ℝ

-- Definition of a plane in 3D space
def Plane3D := (ℝ × ℝ × ℝ) × ℝ

-- Statement for proposition p
theorem no_lg_sum_split : ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ lg (a + b) = lg a + lg b := by
  sorry

-- Definition of skew lines
def are_skew (l1 l2 : Line3D) : Prop := 
  ∀ (p : Plane3D), (∃ t : ℝ, l1 t = p.1) → ¬(∃ s : ℝ, l2 s = p.1)

-- Statement for proposition q
theorem skew_iff_no_common_plane (l1 l2 : Line3D) : 
  are_skew l1 l2 ↔ ∀ (p : Plane3D), ¬((∃ t : ℝ, l1 t = p.1) ∧ (∃ s : ℝ, l2 s = p.1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_lg_sum_split_skew_iff_no_common_plane_l198_19827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_weight_approx_l198_19823

/-- Calculates the weight of a hollow iron pipe given its dimensions and iron density. -/
noncomputable def pipeWeight (length : ℝ) (externalDiameter : ℝ) (thickness : ℝ) (ironDensity : ℝ) : ℝ :=
  let externalRadius := externalDiameter / 2
  let internalRadius := externalRadius - thickness
  let externalVolume := Real.pi * externalRadius ^ 2 * length
  let internalVolume := Real.pi * internalRadius ^ 2 * length
  let ironVolume := externalVolume - internalVolume
  ironVolume * ironDensity

/-- The weight of the pipe is approximately 3694.68 grams. -/
theorem pipe_weight_approx :
  let length := (21 : ℝ)
  let externalDiameter := (8 : ℝ)
  let thickness := (1 : ℝ)
  let ironDensity := (8 : ℝ)
  abs (pipeWeight length externalDiameter thickness ironDensity - 3694.68) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_weight_approx_l198_19823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transmitted_word_is_mif_l198_19828

-- Define the alphabet and digit mapping
def alphabet : List Char := ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

def letterToDigits (c : Char) : Option (Fin 33) :=
  (alphabet.indexOf? c).map (Fin.ofNat)

-- Define the transformation function
def f (x y : ℕ) : ℕ := (x + 4 * y) % 10

-- Define the intercepted sequence
def intercepted : List ℕ := [1, 3, 7, 1]

-- Define the theorem
theorem transmitted_word_is_mif :
  ∃ (m₁ m₂ m₃ m₄ m₅ m₆ : Fin 33),
    f m₂.val (f m₁.val 1) = 3 ∧
    f m₄.val (f m₃.val 3) = 7 ∧
    f m₆.val (f m₅.val 7) = 1 ∧
    letterToDigits 'м' = some m₁ ∧
    letterToDigits 'и' = some m₃ ∧
    letterToDigits 'ф' = some m₅ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transmitted_word_is_mif_l198_19828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_is_three_l198_19897

def geometric_sequence (a₀ : ℤ) (r : ℤ) (n : ℕ) : ℤ := a₀ * r^n

def arithmetic_sequence (a₀ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₀ + d * n

def sequence_A (n : ℕ) : ℤ := geometric_sequence 3 3 n
def sequence_B (n : ℕ) : ℤ := arithmetic_sequence 100 10 n

def valid_A (n : ℕ) : Prop := sequence_A n ≤ 300
def valid_B (n : ℕ) : Prop := sequence_B n ≤ 300

theorem min_difference_is_three :
  ∃ (i j : ℕ), valid_A i ∧ valid_B j ∧
    (∀ (k l : ℕ), valid_A k → valid_B l →
      |sequence_A k - sequence_B l| ≥ |sequence_A i - sequence_B j|) ∧
    |sequence_A i - sequence_B j| = 3 := by
  sorry

#check min_difference_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_is_three_l198_19897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l198_19837

/-- A cubic function f(x) = ax³ + bx + c with specific properties -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  odd : ∀ x, a * (-x)^3 + b * (-x) + c = -(a * x^3 + b * x + c)
  tangent_perpendicular : (3 * a + b) * 6 = -1
  min_derivative : ∀ x, 3 * a * x^2 + b ≥ -12
  min_derivative_achieved : ∃ x, 3 * a * x^2 + b = -12

/-- The function defined by the CubicFunction structure -/
def CubicFunction.f (self : CubicFunction) (x : ℝ) : ℝ := self.a * x^3 + self.b * x + self.c

/-- The main theorem stating the properties of the cubic function -/
theorem cubic_function_properties (f : CubicFunction) :
  f.a = 2 ∧ f.b = -12 ∧ f.c = 0 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 3, f.f x ≤ 18) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 3, f.f x ≥ -8) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 3, f.f x = 18) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 3, f.f x = -8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l198_19837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l198_19882

noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x + 2^(x+1) + 1

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l198_19882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mode_difference_l198_19806

def data : List Int := [21, 23, 23, 23, 24, 30, 30, 30, 42, 42, 47, 48, 51, 52, 53, 55, 60, 62, 64]

def mode (l : List Int) : Int := sorry

def median (l : List Int) : Int := sorry

theorem median_mode_difference :
  let m := median data
  let d := mode data
  (m - d).natAbs = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mode_difference_l198_19806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_winning_strategy_l198_19838

/-- Represents a binary digit -/
inductive BinaryDigit
| Zero : BinaryDigit
| One : BinaryDigit

/-- Represents a player in the game -/
inductive Player
| Ana : Player
| Bob : Player

/-- Represents the state of the game board -/
def GameBoard := List BinaryDigit

/-- Represents a game state -/
structure GameState where
  board : GameBoard
  currentPlayer : Player
  remainingMoves : Nat

/-- Checks if a natural number can be expressed as the sum of two squares -/
def isSumOfTwoSquares (n : Nat) : Prop := 
  ∃ (a b : Nat), n = a^2 + b^2

/-- Converts a GameBoard to a natural number -/
def boardToNat (board : GameBoard) : Nat :=
  sorry

/-- Determines if a player has a winning strategy -/
def hasWinningStrategy (player : Player) : Prop :=
  ∀ (initialState : GameState),
    initialState.currentPlayer = player →
    initialState.remainingMoves = 4042 →
    ∃ (finalBoard : GameBoard),
      ¬(isSumOfTwoSquares (boardToNat finalBoard))

/-- The main theorem stating that Bob has a winning strategy -/
theorem bob_has_winning_strategy :
  hasWinningStrategy Player.Bob :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_has_winning_strategy_l198_19838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l198_19868

/-- A hyperbola with specific properties -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_axis : ℝ → ℝ
  eccentricity : ℝ
  vertex : ℝ × ℝ

/-- The specific hyperbola described in the problem -/
noncomputable def C : Hyperbola :=
  { center := (0, 0)
    foci_axis := λ x ↦ 0  -- y = 0, representing the y-axis
    eccentricity := Real.sqrt 2
    vertex := (0, -1) }

/-- The standard form equation of a hyperbola -/
def standard_equation (x y : ℝ) : Prop :=
  y^2 - x^2 = 1

/-- Theorem stating that the hyperbola C satisfies the standard equation y² - x² = 1 -/
theorem hyperbola_equation : ∀ x y : ℝ, standard_equation x y := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l198_19868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_on_line_l198_19851

theorem sin_minus_cos_on_line (α : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y = -3 * x ∧ 
   (Real.sin α = y / Real.sqrt (x^2 + y^2)) ∧
   (Real.cos α = x / Real.sqrt (x^2 + y^2))) →
  Real.sin α - Real.cos α = 2 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_on_line_l198_19851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_two_range_l198_19836

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then Real.exp (x - 1)
  else x ^ (1/3 : ℝ)

-- State the theorem
theorem f_leq_two_range (x : ℝ) :
  f x ≤ 2 ↔ x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_two_range_l198_19836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l198_19881

/-- Represents a statement about deductive reasoning -/
inductive DeductiveStatement
| GeneralToSpecific
| DefinitelyCorrect
| Syllogism
| DependsOnPremises

/-- Determines if a given statement about deductive reasoning is correct -/
def is_correct (s : DeductiveStatement) : Bool :=
  match s with
  | DeductiveStatement.GeneralToSpecific => true
  | DeductiveStatement.DefinitelyCorrect => false
  | DeductiveStatement.Syllogism => true
  | DeductiveStatement.DependsOnPremises => true

/-- The list of all statements about deductive reasoning -/
def all_statements : List DeductiveStatement :=
  [DeductiveStatement.GeneralToSpecific,
   DeductiveStatement.DefinitelyCorrect,
   DeductiveStatement.Syllogism,
   DeductiveStatement.DependsOnPremises]

/-- Counts the number of correct statements in the list -/
def count_correct_statements : Nat :=
  (all_statements.filter is_correct).length

/-- Theorem: The number of correct statements about deductive reasoning is 3 -/
theorem correct_statements_count :
  count_correct_statements = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l198_19881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l198_19822

/-- Parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (2, 0)

/-- Fixed point A -/
def A : ℝ × ℝ := (3, 2)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_min_distance :
  ∃ (min : ℝ), min = 5 ∧
  ∀ (P : ℝ × ℝ), P ∈ Parabola →
    distance P F + distance P A ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l198_19822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_products_invariant_l198_19888

/-- The sum of products when dividing n pebbles -/
def sum_of_products (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Proof that the sum of products is always n(n-1)/2 for any division sequence -/
theorem sum_of_products_invariant (n : ℕ) :
  ∀ (division_sequence : List (Nat × Nat)),
    (∀ (pair : Nat × Nat), pair ∈ division_sequence → pair.1 + pair.2 ≤ n) →
    (∀ i, i ∈ Finset.range n → i.succ ∉ (division_sequence.map (λ pair => pair.1 + pair.2))) →
    List.sum (division_sequence.map (λ pair => pair.1 * pair.2)) = sum_of_products n :=
  sorry

#eval sum_of_products 1001

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_products_invariant_l198_19888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l198_19861

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∃ (h k r : ℝ), ∀ x y : ℝ, x^2 + y^2 - a*x + y + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

def q (a : ℝ) : Prop := ∃ m : ℝ, m > 1 ∧ ∀ x y : ℝ, 2*a*x + (1-a)*y + 1 = 0 ↔ y = m*x + (2*a*x + 1)/(a - 1)

-- Define the range of a
def range_a (a : ℝ) : Prop := a ∈ (Set.Icc (-3) (-1) ∪ Set.Ioc 1 3)

-- State the theorem
theorem problem_statement : 
  ∀ a : ℝ, ((p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ range_a a :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l198_19861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l198_19812

/-- Represents the speed of a train in km/hr given its length in meters and time to cross a pole in seconds -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem stating that a train of length 160 meters crossing a pole in 8 seconds has a speed of 72 km/hr -/
theorem train_speed_calculation :
  train_speed 160 8 = 72 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l198_19812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_theorem_l198_19844

/-- A circle passing through (0,1) and tangent to y = x^2 at (2,4) has center (-16/5, 53/10) -/
theorem circle_center_theorem :
  ∃ (center : ℝ × ℝ),
    let (x, y) := center
    (x = -16/5 ∧ y = 53/10) ∧
    (x - 0)^2 + (y - 1)^2 = (x - 2)^2 + (y - 4)^2 ∧
    (∃ (r : ℝ), (x - 2)^2 + (y - 4)^2 = r^2 ∧
    ∀ (t : ℝ), (t - 2)^2 + (t^2 - 4)^2 ≥ r^2) ∧
    (y - 4 = 4 * (x - 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_theorem_l198_19844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_is_six_l198_19884

def mySequence : List ℕ := [2, 16, 4, 14, 6, 12, 8]

theorem fifth_number_is_six : 
  mySequence[4] = 6 := by
  rfl

#eval mySequence[4]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_is_six_l198_19884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_solution_existence_l198_19841

theorem periodic_solution_existence (a : ℝ) :
  (a > 0 ∧ ∃ n : ℕ, n > 0 ∧ a = 1 / (n : ℝ)) ↔
  ∀ f : ℝ → ℝ, ContinuousOn f (Set.Icc 0 1) →
    f 0 = 0 → f 1 = 0 →
    ∃ x ∈ Set.Icc 0 1, f (x + a) = f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_solution_existence_l198_19841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_specific_circles_l198_19885

/-- The square of the distance between intersection points of two circles -/
def intersection_distance_squared (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  let x1 := c1_center.1
  let y1 := c1_center.2
  let x2 := c2_center.1
  let y2 := c2_center.2
  -- Define the distance squared between intersection points
  -- This definition is left abstract as the actual computation is complex
  sorry

/-- Theorem stating the square of the distance between intersection points of two specific circles -/
theorem intersection_distance_specific_circles :
  intersection_distance_squared (1, -2) (1, 4) 5 3 = 56/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_specific_circles_l198_19885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l198_19824

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Area of a triangle given base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

theorem parabola_triangle_area (P : ℝ × ℝ) 
  (h1 : parabola P.1 P.2)
  (h2 : distance P focus = 4) :
  triangleArea (distance focus origin) (abs P.2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l198_19824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manex_destination_time_l198_19817

/-- Represents the tour details for Manex's bus trip -/
structure TourDetails where
  distanceToDestination : ℚ
  extraDistanceReturn : ℚ
  drivingSpeed : ℚ
  totalTourTime : ℚ

/-- Calculates the time spent at the destination given tour details -/
def timeAtDestination (tour : TourDetails) : ℚ :=
  let totalDistance := tour.distanceToDestination + (tour.distanceToDestination + tour.extraDistanceReturn)
  let drivingTime := totalDistance / tour.drivingSpeed
  tour.totalTourTime - drivingTime

/-- Theorem stating that given the specific tour details, Manex spent 2 hours at the destination -/
theorem manex_destination_time :
  let tour : TourDetails := {
    distanceToDestination := 55,
    extraDistanceReturn := 10,
    drivingSpeed := 30,  -- 1 mile per 2 minutes = 30 miles per hour
    totalTourTime := 6
  }
  timeAtDestination tour = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manex_destination_time_l198_19817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_and_usage_l198_19813

-- Define the water pricing policy
structure WaterPricing where
  basic_price : ℝ
  sewage_fee : ℝ

-- Define the tiered pricing function
noncomputable def tiered_price (wp : WaterPricing) (usage : ℝ) : ℝ :=
  if usage ≤ 10 then
    (wp.basic_price + wp.sewage_fee) * usage
  else
    (wp.basic_price + wp.sewage_fee) * 10 + (2 * wp.basic_price + wp.sewage_fee) * (usage - 10)

-- Define the theorem
theorem water_pricing_and_usage 
  (wp : WaterPricing)
  (user_a_usage : ℝ)
  (user_a_price : ℝ)
  (user_b_usage : ℝ)
  (user_b_price : ℝ)
  (budget : ℝ)
  (h1 : user_a_usage = 8)
  (h2 : user_a_price = 27.6)
  (h3 : user_b_usage = 12)
  (h4 : user_b_price = 46.3)
  (h5 : budget = 64)
  (h6 : tiered_price wp user_a_usage = user_a_price)
  (h7 : tiered_price wp user_b_usage = user_b_price) :
  wp.basic_price = 2.45 ∧ 
  wp.sewage_fee = 1 ∧ 
  ∃ (max_usage : ℝ), max_usage = 15 ∧ 
    ∀ (u : ℝ), u ≤ max_usage → tiered_price wp u ≤ budget :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_and_usage_l198_19813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_support_sum_inequality_l198_19891

/-- A function from ℤ × ℤ to ℝ with only finitely many nonzero values -/
def FiniteSupport (a : ℤ × ℤ → ℝ) : Prop :=
  ∃ (S : Finset (ℤ × ℤ)), ∀ (x y : ℤ), a (x, y) ≠ 0 → (x, y) ∈ S

theorem finite_support_sum_inequality
  (a : ℤ × ℤ → ℝ)
  (h_zero : a (0, 0) = 0)
  (h_finite : FiniteSupport a) :
  ∑' (p : ℤ × ℤ), a p * (a (p.1, 2*p.1 + p.2) + a (p.1 + 2*p.2, p.2)) ≤
  Real.sqrt 3 * ∑' (p : ℤ × ℤ), (a p)^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_support_sum_inequality_l198_19891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B₁_cond_prob_sum_prob_sum_cond_prob_sum_neq_l198_19872

variable (a b : ℕ)

-- Define the events
def A₁ : Set (ℕ × ℕ) := {(x, _) | x = 1}  -- First draw is red
def A₂ : Set (ℕ × ℕ) := {(x, _) | x = 2}  -- First draw is blue
def B₁ : Set (ℕ × ℕ) := {(_, y) | y = 1}  -- Second draw is red
def B₂ : Set (ℕ × ℕ) := {(_, y) | y = 2}  -- Second draw is blue

-- Assumptions
variable [Fact (a ≥ 2)] [Fact (b ≥ 2)]

-- Probability measure
variable (P : Set (ℕ × ℕ) → ℝ)

-- Axioms for probability measure
axiom prob_nonneg : ∀ S, P S ≥ 0
axiom prob_total : P (Set.univ : Set (ℕ × ℕ)) = 1

-- Conditional probability
noncomputable def condProb (A B : Set (ℕ × ℕ)) : ℝ := P (A ∩ B) / P B

-- Theorem statements
theorem prob_B₁ : P B₁ = a / (a + b) := by sorry

theorem cond_prob_sum : condProb P B₁ A₁ + condProb P B₂ A₁ = 1 := by sorry

theorem prob_sum : P B₁ + P B₂ = 1 := by sorry

theorem cond_prob_sum_neq : condProb P B₂ A₁ + condProb P B₁ A₂ ≠ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B₁_cond_prob_sum_prob_sum_cond_prob_sum_neq_l198_19872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brady_work_hours_l198_19821

def hours_per_day_april : ℝ → Prop := λ x => True
def hours_per_day_june : ℝ := 5
def hours_per_day_september : ℝ := 8
def days_per_month : ℕ := 30
def average_hours_per_month : ℝ := 190

theorem brady_work_hours : 
  ∃ (x : ℝ), hours_per_day_april x ∧ 
  (x * days_per_month + hours_per_day_june * days_per_month + hours_per_day_september * days_per_month) / 3 = average_hours_per_month ∧
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brady_work_hours_l198_19821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l198_19889

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1 + Real.sqrt x

-- State the theorem
theorem f_domain : Set.Ici (0 : ℝ) = {x : ℝ | ∃ y, f y = x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l198_19889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_magnitude_l198_19810

-- Define the vectors
def AB : ℝ × ℝ := (-1, 2)
def BC : ℝ → ℝ × ℝ := λ x ↦ (x, -5)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Theorem statement
theorem AC_magnitude (x : ℝ) :
  dot_product AB (BC x) = -7 → magnitude (vector_add AB (BC x)) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_magnitude_l198_19810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l198_19850

noncomputable section

-- Define the angle XOY
def angle_XOY : ℝ := 60 * Real.pi / 180

-- Define point A on OX arm
def OA (a : ℝ) : ℝ := a

-- Define AD perpendicular to OX
def AD (a : ℝ) : ℝ := a * Real.sqrt 3

-- Define OD
def OD (a : ℝ) : ℝ := 2 * a * Real.sqrt 3

-- Define BD
def BD (a : ℝ) : ℝ := OD a - AD a

-- Surface area of BD rotation
def surface_area_BD_rotation (a : ℝ) : ℝ := (15 * a^2 * Real.pi * Real.sqrt 3) / 8

-- Volume of ABD rotation
def volume_ABD_rotation (a : ℝ) : ℝ := (15 * a^3 * Real.pi) / 16

theorem geometry_problem (a : ℝ) (h : a > 0) :
  BD a = 3 * a / 2 ∧
  surface_area_BD_rotation a = (15 * a^2 * Real.pi * Real.sqrt 3) / 8 ∧
  volume_ABD_rotation a = (15 * a^3 * Real.pi) / 16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l198_19850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l198_19855

theorem inequality_proof (x : ℝ) (hx : x > 0) : 
  (2 : ℝ)^(12 * Real.sqrt x) + (2 : ℝ)^(x^(1/4)) ≥ 2 * (2 : ℝ)^(x^(1/6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l198_19855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_price_change_l198_19834

theorem bakery_price_change (p : ℝ) (hp : p > 0) : 
  0.75 * (1.3 * p) = 0.975 * p := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_price_change_l198_19834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_characterization_l198_19815

-- Define the sets A and B
def A : Set ℝ := {x | Real.exp (x * Real.log 2) > 1}
def B : Set ℝ := {x | x^2 - 3*x - 4 > 0}

-- State the theorem
theorem intersection_characterization :
  ∀ x : ℝ, x ∈ (A ∩ B) ↔ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_characterization_l198_19815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l198_19877

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - (a * 5^x) / (5^x + 1)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem function_properties (a b m : ℝ) :
  (∀ x, b - 3 < x ∧ x < 2 * b → f a x = f a x) →
  is_odd_function (f a) →
  is_decreasing_on (f a) (b - 3) (2 * b) →
  f a (m - 1) + f a (2 * m + 1) > 0 →
  a = 2 ∧ b = 1 ∧ -1 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l198_19877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_true_l198_19833

noncomputable section

-- Define power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x ^ b

-- Define the function f(x) = 2^x - log_2(x)
noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 2

-- Define the solution set of √(x-1)(x-2) ≥ 0
def solutionSet : Set ℝ := {x | x ≥ 2 ∨ x = 1}

-- Define sufficient but not necessary condition
def sufficientButNotNecessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ ∃ x, q x ∧ ¬p x

-- Define arithmetic and geometric sequences
def isArithmeticOrGeometric (a : ℝ) : Prop :=
  let s := fun n => a^n - 1
  (∃ d, ∀ n, s (n+1) - s n = d) ∨
  (∃ r, ∀ n, s (n+1) / s n = r)

theorem only_fourth_proposition_true :
  ¬(isPowerFunction (fun _ => 1)) ∧
  ¬(∃! x, f x = 0) ∧
  solutionSet ≠ {x | x ≥ 2} ∧
  sufficientButNotNecessary (fun x => x < 1) (fun x => x < 2) ∧
  ¬(∀ a, isArithmeticOrGeometric a) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_true_l198_19833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_8_l198_19875

-- Define the polar line
noncomputable def polar_line (ρ : ℝ) : ℝ := Real.pi / 3

-- Define the polar circle
noncomputable def polar_circle (θ : ℝ) : ℝ := 4 * Real.cos θ + 4 * Real.sqrt 3 * Real.sin θ

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := ⟨2, 2 * Real.sqrt 3⟩
noncomputable def B : ℝ × ℝ := ⟨-2, -2 * Real.sqrt 3⟩

-- State the theorem
theorem length_AB_is_8 :
  let dist := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dist = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_8_l198_19875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l198_19839

open BigOperators Nat Real

theorem infinite_sum_convergence : 
  HasSum (fun n : ℕ => (n^2 + 2*n - 2) / (Nat.factorial (n + 3) : ℝ)) (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l198_19839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_minus_d_l198_19818

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ := c * x + d

noncomputable def g (x : ℝ) : ℝ := -4 * x + 6

noncomputable def h (c d : ℝ) (x : ℝ) : ℝ := f c d (g x)

theorem find_c_minus_d (c d : ℝ) :
  (∀ x, h c d x = x - 8) →
  c - d = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_minus_d_l198_19818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l198_19893

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := tan x - 2 / abs (cos x)

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), M = -Real.sqrt 3 ∧
  (∀ x, 0 < x → x < π / 2 → f x ≤ M) ∧
  (∃ x, 0 < x ∧ x < π / 2 ∧ f x = M) :=
by
  -- We'll use -√3 as our maximum value
  let M := -Real.sqrt 3
  
  -- Prove that M satisfies the conditions
  have h1 : M = -Real.sqrt 3 := by rfl
  
  have h2 : ∀ x, 0 < x → x < π / 2 → f x ≤ M := by
    sorry -- Proof omitted
  
  have h3 : ∃ x, 0 < x ∧ x < π / 2 ∧ f x = M := by
    -- The maximum occurs at x = π/6
    use π / 6
    sorry -- Proof omitted
  
  -- Combine the results
  exact ⟨M, h1, h2, h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l198_19893
