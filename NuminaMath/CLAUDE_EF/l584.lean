import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_at_e_l584_58460

/-- Given a function f(x) = x + x(log x)², prove that xf'(x) = 2f(x) when x = e -/
theorem function_equality_at_e (x : ℝ) :
  let f : ℝ → ℝ := λ x => x + x * (Real.log x)^2
  let f' : ℝ → ℝ := λ x => 1 + (Real.log x)^2 + 2 * Real.log x
  x * f' x = 2 * f x ↔ x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_at_e_l584_58460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doug_wins_probability_l584_58471

/-- The probability that Doug hits more home runs than Ryan in one round of the Wiffle Ball Home Run Derby. -/
noncomputable def probability_doug_wins (doug_prob ryan_prob : ℝ) : ℝ :=
  (doug_prob * (1 - ryan_prob)) / (1 - doug_prob * ryan_prob)

/-- Theorem stating that the probability of Doug hitting more home runs than Ryan is 1/5,
    given Doug's probability of hitting a home run is 1/3 and Ryan's probability is 1/2. -/
theorem doug_wins_probability :
  probability_doug_wins (1/3) (1/2) = 1/5 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doug_wins_probability_l584_58471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animals_meet_on_ninth_day_l584_58407

/-- Represents the travel problem of a fine horse and a mule --/
structure TravelProblem where
  distance : ℚ  -- Total distance between Luoyang and Qi
  horse_initial : ℚ  -- Fine horse's initial daily travel
  horse_increase : ℚ  -- Fine horse's daily increase
  mule_initial : ℚ  -- Mule's initial daily travel
  mule_decrease : ℚ  -- Mule's daily decrease

/-- Calculates the total distance traveled by both animals after n days --/
def total_distance (p : TravelProblem) (n : ℕ) : ℚ :=
  -- Horse's distance (including return trip)
  (p.horse_initial * n + n * (n - 1) / 2 * p.horse_increase) * 2 +
  -- Mule's distance
  (p.mule_initial * n - n * (n - 1) / 2 * p.mule_decrease)

/-- The main theorem stating that the animals meet on the 9th day --/
theorem animals_meet_on_ninth_day (p : TravelProblem) 
  (h1 : p.distance = 1125)
  (h2 : p.horse_initial = 103)
  (h3 : p.horse_increase = 13)
  (h4 : p.mule_initial = 97)
  (h5 : p.mule_decrease = 1/2) :
  total_distance p 9 = 2 * p.distance := by
  sorry

#eval total_distance { distance := 1125, horse_initial := 103, horse_increase := 13, mule_initial := 97, mule_decrease := 1/2 } 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_animals_meet_on_ninth_day_l584_58407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_time_l584_58477

/-- The time (in minutes) it takes for two people to meet on a circular path -/
noncomputable def meeting_time (path_length : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  path_length / (speed1 + speed2)

/-- Theorem: Two people traveling in opposite directions on a circular path of 3000 meters,
    with speeds of 100 m/min and 150 m/min respectively, will meet after 12 minutes -/
theorem first_meeting_time :
  meeting_time 3000 100 150 = 12 := by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- Simplify the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_meeting_time_l584_58477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_area_l584_58496

/-- The function f(x) = 12 - x^2 -/
noncomputable def f (x : ℝ) : ℝ := 12 - x^2

/-- The slope of the tangent line we're looking for -/
def targetSlope : ℝ := -2

/-- The area of the triangle formed by the tangent line and coordinate axes -/
noncomputable def S (t : ℝ) : ℝ := (1/4) * (t + 12/t) * (12 + t^2)

theorem tangent_line_and_minimum_area :
  (∃ a b : ℝ, (∀ x : ℝ, a * x + b = targetSlope * x + 13) ∧
              (a = targetSlope)) ∧
  (∃ minArea : ℝ, minArea = 32 ∧ ∀ t : ℝ, S t ≥ minArea) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_minimum_area_l584_58496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouquet_count_l584_58498

/-- The number of different bouquets that can be purchased for exactly $60,
    given that roses cost $3 each, carnations cost $2 each, and tulips cost $4 each. -/
def number_of_bouquets : ℕ :=
  (Finset.filter (fun p => 3 * p.1 + 2 * p.2.1 + 4 * p.2.2 = 60)
    (Finset.product (Finset.range 61) (Finset.product (Finset.range 61) (Finset.range 61)))).card

/-- Theorem stating that the number of different bouquets is 21 -/
theorem bouquet_count : number_of_bouquets = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouquet_count_l584_58498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_filling_l584_58485

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Option ℕ

-- Define adjacency in the grid
def adjacent (i j k l : Fin 4) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨
  (i = k ∧ l.val + 1 = j.val) ∨
  (j = l ∧ i.val + 1 = k.val) ∨
  (j = l ∧ k.val + 1 = i.val)

-- Define a valid grid filling
def valid_filling (g : Grid) : Prop :=
  (∀ n, 1 ≤ n ∧ n ≤ 16 → ∃ i j, g i j = some n) ∧
  (∀ i j k l m n,
    g i j = some m → g k l = some n → m + 1 = n →
    adjacent i j k l)

-- Define the partially filled grid
def partial_grid : Grid :=
  λ i j ↦
    if i = 0 ∧ j = 2 then some 11
    else if i = 2 ∧ j = 0 then some 5
    else if i = 3 ∧ j = 3 then some 3
    else none

-- Theorem statement
theorem exists_valid_filling :
  ∃ g : Grid, g = partial_grid ∨ valid_filling g :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_filling_l584_58485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_pipes_count_l584_58470

/-- Represents the dimensions of a cylindrical container --/
structure CylinderDimensions where
  diameter : ℝ
  length : ℝ

/-- Calculates the volume of a cylinder given its dimensions --/
noncomputable def cylinderVolume (c : CylinderDimensions) : ℝ :=
  (Real.pi / 4) * c.diameter^2 * c.length

/-- The main theorem stating the number of small pipes required --/
theorem small_pipes_count
  (large_tank : CylinderDimensions)
  (small_pipe : CylinderDimensions)
  (h_large_diameter : large_tank.diameter = 12)
  (h_large_length : large_tank.length = 120)
  (h_small_diameter : small_pipe.diameter = 2)
  (h_small_length : small_pipe.length = 40) :
  (cylinderVolume large_tank) / (cylinderVolume small_pipe) = 108 := by
  sorry

#eval 108 -- Expected output: 108

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_pipes_count_l584_58470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l584_58415

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l584_58415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l584_58499

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x + 3)

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Icc 1 3, 
    -x^2 + 2*x + 3 > 0 →
    ∀ y ∈ Set.Icc 1 3, 
      x < y → f y < f x :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l584_58499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_and_circumcenter_l584_58422

noncomputable section

structure IsoscelesTriangle where
  a : ℝ  -- length of the base
  b : ℝ  -- length of the equal sides
  h : 0 < a ∧ 0 < b  -- positive lengths

def divides_area_in_half (t : IsoscelesTriangle) : Prop :=
  ∃ (x : ℝ), 0 < x ∧ x < t.a ∧ x * t.b = (t.a - x) * t.b

def divides_perimeter (t : IsoscelesTriangle) (l₁ l₂ : ℝ) : Prop :=
  (t.a + t.b = l₁ ∧ t.b + t.b = l₂) ∨ (t.a + t.b = l₂ ∧ t.b + t.b = l₁)

noncomputable def triangle_area (t : IsoscelesTriangle) : ℝ :=
  (t.a * t.b) / 2

def circumcenter_outside (t : IsoscelesTriangle) : Prop :=
  t.a^2 > 2 * t.b^2

theorem isosceles_triangle_area_and_circumcenter 
  (t : IsoscelesTriangle) 
  (h₁ : divides_area_in_half t) 
  (h₂ : divides_perimeter t 5 7) :
  (triangle_area t = 16/3 ∧ circumcenter_outside t) ∨
  (triangle_area t = 8 * Real.sqrt 5 / 3 ∧ ¬circumcenter_outside t) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_and_circumcenter_l584_58422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_tetrahedron_volume_ratio_l584_58489

/-- The volume of a regular tetrahedron with side length s -/
noncomputable def tetrahedronVolume (s : ℝ) : ℝ := s^3 * Real.sqrt 2 / 12

/-- The volume of a regular octahedron with side length a -/
noncomputable def octahedronVolume (a : ℝ) : ℝ := a^3 * Real.sqrt 2 / 3

/-- The side length of the octahedron formed by joining midpoints of a tetrahedron's edges -/
noncomputable def octahedronSideLength (s : ℝ) : ℝ := s / 2

theorem octahedron_tetrahedron_volume_ratio (s : ℝ) (h : s > 0) :
  octahedronVolume (octahedronSideLength s) / tetrahedronVolume s = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_tetrahedron_volume_ratio_l584_58489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trick_always_succeeds_l584_58448

/-- Represents the number of caskets in the circle. -/
def n : ℕ := 12

/-- Represents the box opened by the assistant. -/
def k : ℕ := 0

/-- Represents the positions of the two caskets containing coins. -/
def coin1 : ℕ := 0
def coin2 : ℕ := 0

/-- The template of boxes to be opened by the magician. -/
def template (k : ℕ) : Finset ℕ :=
  {(k + 1) % n, (k + 2) % n, (k + 5) % n, (k + 7) % n}

/-- Theorem stating that the template always includes the two caskets with coins. -/
theorem trick_always_succeeds (h1 : coin1 < n) (h2 : coin2 < n) (h3 : coin1 ≠ coin2) (h4 : k < n) 
  (h5 : k ≠ coin1) (h6 : k ≠ coin2) :
  coin1 ∈ template k ∨ coin2 ∈ template k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trick_always_succeeds_l584_58448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_discount_is_half_l584_58413

-- Define the promotional discount
noncomputable def promotional_discount : ℝ := 1 / 3

-- Define the coupon discount
noncomputable def coupon_discount : ℝ := 0.25

-- Theorem: The total discount is 50% off the original price
theorem total_discount_is_half :
  (1 - (1 - promotional_discount) * (1 - coupon_discount)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_discount_is_half_l584_58413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_d_values_l584_58419

noncomputable def g (d : ℝ) (x : ℝ) : ℝ := d / (3 * x - 4)

noncomputable def g_inv (d : ℝ) (y : ℝ) : ℝ := (d + 4 * y) / (3 * y)

theorem product_of_d_values :
  ∃ d₁ d₂ : ℝ, d₁ * d₂ = -8/3 ∧ 
  (g d₁ 3 = g_inv d₁ (d₁ + 2) ∧ g d₂ 3 = g_inv d₂ (d₂ + 2)) :=
by
  sorry

#check product_of_d_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_d_values_l584_58419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l584_58472

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

theorem f_monotone_increasing (ω : ℝ) (h1 : ω > 0) (h2 : ∃ T > 0, ∀ x, f ω (x + T) = f ω x ∧ T = 4 * Real.pi) :
  StrictMonoOn (f ω) (Set.Ioo (Real.pi / 2) Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l584_58472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l584_58430

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Representation of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  f₁ : Point
  f₂ : Point

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  h.c / h.a

/-- The theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (A B C : Point)
  (h_foci : h.f₁.x = -h.c ∧ h.f₁.y = 0 ∧ h.f₂.x = h.c ∧ h.f₂.y = 0)
  (h_AB : A.x = h.c ∧ B.x = h.c ∧ A.y = h.b^2 / h.a ∧ B.y = -h.b^2 / h.a)
  (h_C : C.x = 0 ∧ C.y = -h.b^2 / (2 * h.a))
  (h_perp : (A.x - C.x) * (B.x - h.f₁.x) + (A.y - C.y) * (B.y - h.f₁.y) = 0)
  : eccentricity h = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l584_58430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_throw_probabilities_l584_58454

/-- The probability of Player A making a successful free throw -/
noncomputable def prob_A : ℝ := 1/2

/-- The probability of Player B making a successful free throw -/
noncomputable def prob_B : ℝ := 2/5

/-- The probability of exactly one successful throw when each player takes one throw -/
noncomputable def prob_one_success : ℝ := prob_A * (1 - prob_B) + prob_B * (1 - prob_A)

/-- The probability of at least one successful throw when each player takes two throws -/
noncomputable def prob_at_least_one_success : ℝ := 1 - (1 - prob_A)^2 * (1 - prob_B)^2

theorem free_throw_probabilities :
  prob_one_success = 1/2 ∧ prob_at_least_one_success = 91/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_throw_probabilities_l584_58454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cos_2x_cos_2y_cos_2z_zero_l584_58462

theorem sum_cos_2x_cos_2y_cos_2z_zero
  (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0)
  (h3 : Real.cos x + Real.sin y + Real.cos z = 0) :
  Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cos_2x_cos_2y_cos_2z_zero_l584_58462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_odd_condition_l584_58401

noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def f (φ : ℝ) : ℝ → ℝ := λ x ↦ Real.sin (x + φ)

theorem sin_shift_odd_condition :
  (∃ φ ≠ 0, is_odd (f φ)) ∧ (∀ φ, φ = 0 → is_odd (f φ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_odd_condition_l584_58401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_decimal_to_binary_l584_58449

open Real

-- Define the proposition
def P (x : ℝ) : Prop := sin (x + cos x) = sqrt 3

-- Theorem 1: Negation of existence
theorem negation_of_existence : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

-- Theorem 2: Binary representation of 66
theorem decimal_to_binary : 
  (66 : Nat) = 0b1000010 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_decimal_to_binary_l584_58449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_evaluation_l584_58439

theorem fraction_evaluation : 
  ∃ (x : ℚ), (x = 8 / (5 * 42)) ∧ (round (100 * x) / 100 = 4 / 100) :=
by
  -- Define x as the rational number 8 / (5 * 42)
  let x : ℚ := 8 / (5 * 42)
  
  -- Prove existence
  use x
  
  apply And.intro
  
  -- Prove the first part of the conjunction
  · rfl  -- reflexivity proves x = x
  
  -- Prove the second part of the conjunction
  · sorry  -- We'll skip the detailed calculation for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_evaluation_l584_58439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l584_58479

open Real

theorem trigonometric_equation_solutions :
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, -π ≤ x ∧ x ≤ π) ∧
    (∀ x ∈ S, cos (4*x) + (cos (3*x))^2 + (cos (2*x))^3 + (cos x)^4 + (sin x)^2 = 0) ∧
    (Finset.card S = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solutions_l584_58479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_data_set_l584_58481

noncomputable def median (l : List ℝ) : ℝ := sorry
def count (l : List ℝ) (x : ℝ) : ℕ := sorry

def data_set (x : ℝ) : List ℝ := [1, x, 5, 7]

theorem average_of_data_set (x : ℝ) 
  (unique_mode : ∃! m, m ∈ data_set x ∧ (∀ y ∈ data_set x, (count (data_set x) y) ≤ (count (data_set x) m)))
  (median_is_six : median (data_set x) = 6) :
  (List.sum (data_set x)) / (List.length (data_set x)) = 5 := by
  sorry

#check average_of_data_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_data_set_l584_58481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_from_average_heights_l584_58495

/-- Represents the ratio of men to women -/
noncomputable def ratio_men_to_women (M W : ℝ) : ℝ := M / W

/-- Theorem stating the ratio of men to women given average heights -/
theorem ratio_from_average_heights 
  (avg_all avg_women avg_men : ℝ)
  (M W : ℝ)
  (h_avg_all : avg_all * (M + W) = avg_men * M + avg_women * W)
  (h_avg_all_val : avg_all = 180)
  (h_avg_women : avg_women = 170)
  (h_avg_men : avg_men = 185) :
  ratio_men_to_women M W = 2 := by
  sorry

#check ratio_from_average_heights

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_from_average_heights_l584_58495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l584_58409

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x - Real.pi / 4)

theorem f_monotone_increasing_interval :
  ∃ (a b : ℝ), a = -1/4 ∧ b = 3/4 ∧
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → a ≤ x → x < y → y ≤ b → f x < f y) := by
  sorry

#check f_monotone_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l584_58409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_count_l584_58402

theorem consecutive_integers_count (F : List Int) : 
  (∀ i j, i ∈ F → j ∈ F → i < j → ∀ k, i < k → k < j → k ∈ F) →  -- F consists of consecutive integers
  F.minimum = some (-4) →                                        -- -4 is the least integer in F
  (F.filter (· > 0)).maximum = some (7 : Int) →                  -- Largest positive integer is 7
  F.length = 12 :=                                               -- F contains 12 integers
by
  intro hConsecutive hMin hMax
  sorry  -- Proof details omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_count_l584_58402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_of_students_l584_58486

theorem smallest_number_of_students (n : ℕ) : n ≥ 200 → (
  let total_attended : ℚ := n / 4
  let both_competitions : ℚ := n / 40
  let hinting : ℚ := 33 * n / 200
  let cheating : ℚ := 11 * n / 100
  (3 * n / 4 : ℚ) = n - total_attended ∧
  both_competitions = total_attended / 10 ∧
  hinting = 1.5 * cheating ∧
  hinting + cheating - both_competitions = total_attended ∧
  (∃ m : ℤ, hinting = m) ∧
  (∃ m : ℤ, cheating = m)
) → n = 200 := by
  sorry

#check smallest_number_of_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_of_students_l584_58486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_six_theta_l584_58400

theorem tan_six_theta (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (6 * θ) = 21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_six_theta_l584_58400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_integers_multiple_six_difference_l584_58467

theorem seven_integers_multiple_six_difference (S : Finset ℕ) :
  S ⊆ Finset.range 2011 →
  S.card = 7 →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 6 ∣ (a - b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_integers_multiple_six_difference_l584_58467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan1_more_cost_effective_l584_58417

/-- Represents the cost function for Plan 1 -/
noncomputable def costPlan1 (a x : ℝ) : ℝ := 7 * a * ((x / 4) + (36 / x) - 1)

/-- Represents the cost function for Plan 2 -/
noncomputable def costPlan2 (a x : ℝ) : ℝ := 2 * a * (x + (126 / x)) - (21 * a / 2)

/-- The theorem stating that Plan 1 is more cost-effective than Plan 2 -/
theorem plan1_more_cost_effective (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), 0 < x ∧ x < 14 ∧
  ∀ (y : ℝ), y ≥ 14 →
  costPlan1 a x ≤ 35 * a ∧
  costPlan2 a y ≥ 35.5 * a :=
by
  sorry

#check plan1_more_cost_effective

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan1_more_cost_effective_l584_58417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_color_distinctness_l584_58487

/-- A color type with exactly 5 colors -/
inductive Color
| Red | Blue | Green | Yellow | Purple

/-- A point on the plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- A cross-shaped figure -/
def CrossShape (center : Point) : Set Point :=
  {p : Point | (p.x = center.x ∧ p.y - center.y ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∨
                (p.y = center.y ∧ p.x - center.x ∈ ({-2, -1, 0, 1, 2} : Set ℤ))}

/-- A 1x5 rectangular figure -/
def RectShape (topLeft : Point) : Set Point :=
  {p : Point | p.y = topLeft.y ∧ p.x - topLeft.x ∈ ({0, 1, 2, 3, 4} : Set ℤ)}

/-- All colors in a shape are distinct -/
def AllDistinct (c : Coloring) (s : Set Point) : Prop :=
  ∀ p q : Point, p ∈ s → q ∈ s → p ≠ q → c p ≠ c q

theorem color_distinctness 
  (c : Coloring) 
  (h : ∀ center : Point, AllDistinct c (CrossShape center)) :
  ∀ topLeft : Point, AllDistinct c (RectShape topLeft) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_color_distinctness_l584_58487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_50_value_l584_58482

def F : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to avoid missing cases error
  | 1 => 3
  | (n + 1) => (3 * F n + 1) / 3

theorem F_50_value : F 50 = 19 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_50_value_l584_58482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_tileIII_in_B_l584_58433

-- Define the structure for a tile
structure Tile :=
  (top : Nat) (right : Nat) (bottom : Nat) (left : Nat)

-- Define the four tiles
def tileI : Tile := ⟨5, 3, 1, 6⟩
def tileII : Tile := ⟨2, 6, 3, 5⟩
def tileIII : Tile := ⟨6, 1, 4, 2⟩
def tileIV : Tile := ⟨4, 5, 2, 1⟩

-- Define the set of all tiles
def allTiles : List Tile := [tileI, tileII, tileIII, tileIV]

-- Function to check if a tile can be placed in rectangle B
def canBePlacedInB (t : Tile) : Prop :=
  ∃ (a c d : Tile), a ∈ allTiles ∧ c ∈ allTiles ∧ d ∈ allTiles ∧
    a ≠ t ∧ c ≠ t ∧ d ≠ t ∧ a ≠ c ∧ a ≠ d ∧ c ≠ d ∧
    a.right = t.left ∧ c.top = t.bottom ∧ t.right = d.left

-- Theorem stating that only Tile III can be placed in rectangle B
theorem only_tileIII_in_B :
  ∀ t ∈ allTiles, canBePlacedInB t ↔ t = tileIII :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_tileIII_in_B_l584_58433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l584_58488

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - 2*a*x - 5 else a/x

def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem f_increasing_iff_a_in_range (a : ℝ) :
  isIncreasing (f a) ↔ a ∈ Set.Icc (-2) (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l584_58488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l584_58452

theorem smallest_n_for_sqrt_difference : 
  ∃ (n : ℕ), n > 0 ∧ 
    (∀ (m : ℕ), m > 0 → m < n → Real.sqrt (m : Real) - Real.sqrt ((m - 1) : Real) ≥ 0.005) ∧
    (Real.sqrt (n : Real) - Real.sqrt ((n - 1) : Real) < 0.005) ∧
    n = 10001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l584_58452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_condition_max_value_intersection_points_l584_58416

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 3*x

-- Helper definition for extreme point
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≤ f x ∨ f y ≥ f x

-- Part 1
theorem increasing_condition (a : ℝ) :
  (∀ x ≥ 1, ∀ y ≥ 1, x < y → f a x < f a y) ↔ a ≥ 0 :=
sorry

-- Part 2
theorem max_value (a : ℝ) :
  (∃ x, x = 1/3 ∧ is_extreme_point (f a) x) →
  (∃ M, M = 18 ∧ ∀ x ∈ Set.Icc (-a) 1, f a x ≤ M) :=
sorry

-- Part 3
theorem intersection_points (a : ℝ) :
  (∃ x, x = 1/3 ∧ is_extreme_point (f a) x) →
  (∃ S : Set ℝ, S = Set.Ioo (-7) (-3) ∪ Set.Ioi (-3) ∧
    ∀ b ∈ S, ∃! (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
      f a x = b*x ∧ f a y = b*y ∧ f a z = b*z) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_condition_max_value_intersection_points_l584_58416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_existence_l584_58404

theorem arithmetic_sequence_existence (n : Nat) (nums : Finset Nat) 
  (h1 : n = 5 ∨ n = 1989) (h2 : nums.card = n) (h3 : ∀ (x y : Nat), x ∈ nums → y ∈ nums → x ≠ y → x ≠ y) :
  ∃ (a d : Nat), 0 < d ∧ a ≤ d ∧
    ∃ (seq : Nat → Nat), (∀ k, seq k = a + k * d) ∧
      ∃ (subset : Finset Nat), subset ⊆ nums ∧
        (subset.card = 3 ∨ subset.card = 4) ∧
        (∀ x ∈ subset, ∃ k, seq k = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_existence_l584_58404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l584_58483

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- Define the logarithm base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the theorem
theorem a_range (a : ℝ) (h : f (log2 a) + f (log2 (1/a)) ≤ 2 * f 1) :
  1/2 ≤ a ∧ a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l584_58483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l584_58450

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x^2 + (2*x)/(x^2 + 1) + (x*(x + 5))/(x^2 + 3) + (3*(x + 3))/(x*(x^2 + 3))

/-- Theorem stating that f(x) ≥ 2x for all x > 0 -/
theorem f_lower_bound (x : ℝ) (hx : x > 0) : f x ≥ 2*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l584_58450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l584_58423

-- Define the curve C
noncomputable def curve_C (θ : Real) : Real × Real :=
  (Real.sqrt 3 * Real.cos θ, Real.sin θ)

-- Define the line l
def line_l (θ : Real) (ρ : Real) : Prop :=
  ρ * Real.sin (θ + Real.pi / 4) = 2 * Real.sqrt 2

-- State the theorem
theorem max_distance_curve_to_line :
  ∃ (max_dist : Real),
    max_dist = 3 * Real.sqrt 2 ∧
    ∀ (θ : Real),
      let (x, y) := curve_C θ
      ∀ (d : Real),
        (∃ (ρ θ_l : Real), line_l θ_l ρ ∧ d = Real.sqrt ((x - ρ * Real.cos θ_l)^2 + (y - ρ * Real.sin θ_l)^2)) →
        d ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_curve_to_line_l584_58423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l584_58411

/-- Represents an ellipse with semi-major axis a, semi-minor axis b, and semi-focal distance c -/
structure Ellipse (a b c : ℝ) where
  a_pos : a > 0
  b_pos : b > 0
  a_gt_b : a > b
  focal_condition : c^2 - b^2 + a*c < 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b c) : ℝ := c / a

/-- Theorem stating the range of eccentricity for the given ellipse -/
theorem eccentricity_range (e : Ellipse a b c) : 
  0 < eccentricity e ∧ eccentricity e < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l584_58411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_symmetry_quadratic_root_condition_unique_triangle_l584_58418

open Real

-- Define the function for proposition 1
noncomputable def f (x : ℝ) : ℝ := cos (x - π/4) * cos (x + π/4)

-- Proposition 1
theorem graph_symmetry : ∀ x : ℝ, f (π/2 + x) = f (π/2 - x) := by sorry

-- Proposition 3
theorem quadratic_root_condition (a : ℝ) : 
  (∃! x : ℝ, a*x^2 - 2*a*x - 1 = 0) ↔ a = -1 := by sorry

-- Define a triangle type for proposition 4
structure Triangle where
  AB : ℝ
  AC : ℝ
  angleB : ℝ

-- Proposition 4
theorem unique_triangle : 
  ∃! t : Triangle, t.AB = 1 ∧ t.AC = sqrt 3 ∧ t.angleB = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_symmetry_quadratic_root_condition_unique_triangle_l584_58418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_b_cycle_l584_58451

noncomputable def b : ℕ → ℝ
  | 0 => Real.sin (Real.pi / 18) ^ 2
  | n + 1 => 4 * b n * (1 - b n)

theorem smallest_n_for_b_cycle : (∃ n : ℕ, n > 0 ∧ b n = b 0) ∧ (∀ m : ℕ, 0 < m → m < 18 → b m ≠ b 0) ∧ b 18 = b 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_b_cycle_l584_58451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_tangency_difference_l584_58474

/-- The difference in y-coordinates between the center of a circle 
    and its tangency points with the parabola y = 4x^2 -/
theorem circle_parabola_tangency_difference : 
  let parabola := fun x : ℝ => 4 * x^2
  let tangency_point := (1/2 : ℝ)
  let circle_center_y := tangency_point^2 + 1/8
  circle_center_y - parabola tangency_point = -5/8 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_tangency_difference_l584_58474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_largest_and_smallest_l584_58484

/-- A predicate that checks if a number is a six-digit multiple of three
    containing at least one of each of the digits 0, 1, and 2, and no other digits. -/
def isValidNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧  -- Six-digit number
  n % 3 = 0 ∧                 -- Multiple of three
  ∃ (d0 d1 d2 d3 d4 d5 : ℕ),
    n = d0 * 100000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    ({d0, d1, d2, d3, d4, d5} : Finset ℕ) ⊆ {0, 1, 2} ∧
    0 ∈ ({d0, d1, d2, d3, d4, d5} : Finset ℕ) ∧
    1 ∈ ({d0, d1, d2, d3, d4, d5} : Finset ℕ) ∧
    2 ∈ ({d0, d1, d2, d3, d4, d5} : Finset ℕ)

/-- The largest number satisfying the conditions -/
def largestNumber : ℕ := 222210

/-- The smallest number satisfying the conditions -/
def smallestNumber : ℕ := 100002

theorem difference_of_largest_and_smallest :
  isValidNumber largestNumber ∧
  isValidNumber smallestNumber ∧
  (∀ n, isValidNumber n → smallestNumber ≤ n ∧ n ≤ largestNumber) ∧
  largestNumber - smallestNumber = 122208 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_largest_and_smallest_l584_58484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telescope_visual_range_increase_l584_58410

/-- Calculates the percentage increase given initial and final values -/
noncomputable def percentageIncrease (initial : ℝ) (final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem telescope_visual_range_increase :
  let initialRange : ℝ := 50
  let finalRange : ℝ := 750
  percentageIncrease initialRange finalRange = 1400 := by
  -- Unfold the definition of percentageIncrease
  unfold percentageIncrease
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_telescope_visual_range_increase_l584_58410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_zero_l584_58459

-- Define the polynomials
def p (x : ℝ) : ℝ := 3*x^3 - 6*x^2 - 4*x + 24
def q (x : ℝ) : ℝ := 4*x^3 + 8*x^2 - 20*x - 60

-- Define the equation
def equation (x : ℝ) : Prop := p x * q x = 0

-- Theorem statement
theorem sum_of_roots_is_zero :
  ∃ (S : Finset ℝ), (∀ x ∈ S, equation x) ∧ (S.sum id = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_zero_l584_58459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l584_58475

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 1/3) : Real.cos (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l584_58475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l584_58435

/-- An odd function defined on (-1, 1) that is monotonically decreasing on [0, 1) -/
def OddDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Ioo (-1) 1 → f (-x) = -f x) ∧
  (∀ x y, x ∈ Set.Ico 0 1 → y ∈ Set.Ico 0 1 → x < y → f x > f y)

theorem range_of_a (f : ℝ → ℝ) (h : OddDecreasingFunction f)
    (h_ineq : ∀ a, a ∈ Set.Ioo 0 1 → f (1 - a) + f (1 - a^2) < 0) :
  Set.Ioo 0 1 = {a : ℝ | ∃ x, f (1 - x) + f (1 - x^2) < 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l584_58435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l584_58446

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log 32 / Real.log x = Real.log 4 / Real.log 64 → x = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l584_58446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_one_eq_neg_one_l584_58424

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_eq_neg_one :
  (∀ x, f x + x^2 + x = -(f (-x) + (-x)^2 + (-x))) →  -- y = f(x) + x^2 + x is odd
  f 1 = 1 →                                           -- f(1) = 1
  g (-1) = -1 :=                                      -- g(-1) = -1
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_one_eq_neg_one_l584_58424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_and_coverage_l584_58436

/-- The number of radars -/
def n : ℕ := 8

/-- The radius of each radar's coverage area in km -/
noncomputable def r : ℝ := 17

/-- The width of the coverage ring in km -/
noncomputable def w : ℝ := 16

/-- The central angle between two adjacent radars in radians -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The distance from the center to each radar -/
noncomputable def center_to_radar : ℝ := 15 / Real.sin (θ / 2)

/-- The area of the coverage ring -/
noncomputable def coverage_area : ℝ := 480 * Real.pi / Real.tan (θ / 2)

theorem radar_placement_and_coverage :
  (center_to_radar = 15 / Real.sin (θ / 2)) ∧
  (coverage_area = 480 * Real.pi / Real.tan (θ / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_and_coverage_l584_58436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_theorem_l584_58463

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧
  (∀ (x y : ℝ), x^2 / k + y^2 / 2 = 1 ↔ x^2 / (a^2) + y^2 / (b^2) = 1)

/-- Definition of focal length for an ellipse -/
noncomputable def focal_length (k : ℝ) : ℝ :=
  let a := Real.sqrt (max k 2)
  let b := Real.sqrt (min k 2)
  Real.sqrt (a^2 - b^2)

/-- Theorem: For an ellipse with equation x²/k + y²/2 = 1 and focal length 2, k is either 1 or 3 -/
theorem ellipse_focal_length_theorem (k : ℝ) :
  is_ellipse k → focal_length k = 2 → k = 1 ∨ k = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_theorem_l584_58463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_divisor_count_implies_constant_gcd_subsequence_l584_58473

-- Define a sequence of positive integers
def PositiveIntegerSequence := ℕ → ℕ+

-- Define a property that all elements in the sequence have the same number of divisors
def SameDivisorCount (seq : PositiveIntegerSequence) :=
  ∃ k : ℕ, ∀ n : ℕ, (Nat.divisors (seq n).val).card = k

-- Define a subsequence
def Subsequence (seq : PositiveIntegerSequence) := ℕ → ℕ

-- Define the property that a subsequence has constant GCD
def ConstantGCD (seq : PositiveIntegerSequence) (subseq : Subsequence seq) :=
  ∃ d : ℕ+, ∀ m n : ℕ, Nat.gcd (seq (subseq m)).val (seq (subseq n)).val = d

-- State the theorem
theorem same_divisor_count_implies_constant_gcd_subsequence
  (seq : PositiveIntegerSequence) (h : SameDivisorCount seq) :
  ∃ subseq : Subsequence seq, ConstantGCD seq subseq :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_divisor_count_implies_constant_gcd_subsequence_l584_58473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_equality_l584_58412

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  a : ℝ
  b : ℝ

theorem ellipse_intersection_equality (G : Ellipse) (M : Point) (l : Line) 
    (h1 : G.b^2 = 2)
    (h2 : M.x^2 / G.a^2 + M.y^2 / 2 = 1)
    (h3 : M.x > 0)
    (h4 : M.y = 1)
    (h5 : l.a = Real.sqrt 2)
    (h6 : l.b = -2)
    (h7 : l.c ≠ 0)
    (h8 : ∃ A B : Point, A ≠ B ∧ 
      A.x^2 / G.a^2 + A.y^2 / 2 = 1 ∧ 
      B.x^2 / G.a^2 + B.y^2 / 2 = 1 ∧
      l.a * A.x + l.b * A.y + l.c = 0 ∧
      l.a * B.x + l.b * B.y + l.c = 0) :
  ∃ P Q : Point, 
    P.y = 0 ∧ Q.y = 0 ∧ 
    (P.x - M.x)^2 + (P.y - M.y)^2 = (Q.x - M.x)^2 + (Q.y - M.y)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_equality_l584_58412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_slope_l584_58403

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 12 - y^2 / 4 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (4, 0)

-- Define the line passing through the origin
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define the intersection points
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ hyperbola x y ∧ line k x y}

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem hyperbola_line_intersection_slope :
  ∀ k : ℝ,
  (∃ A B, A ∈ intersection_points k ∧ B ∈ intersection_points k ∧
    triangle_area right_focus A B = 8 * Real.sqrt 3) →
  k = 1/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_slope_l584_58403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_B_l584_58443

def B : Finset Nat := {2, 3, 4}

theorem proper_subsets_of_B :
  (Finset.powerset B).card - 1 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_B_l584_58443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l584_58457

/-- Given a parameterized line in 3D space, prove that the vector at t = -1 is (3, 11, 32) -/
theorem line_parameterization (r : ℝ → ℝ × ℝ × ℝ) 
  (h0 : r 0 = (2, 6, 16)) 
  (h1 : r 1 = (1, 1, 0)) :
  r (-1) = (3, 11, 32) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l584_58457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l584_58491

/-- f is a function defined as f(n) = g^n + 1 where g is an even positive integer -/
def f (g : ℕ) (n : ℕ) : ℕ := g^n + 1

/-- Main theorem statement -/
theorem f_properties (g : ℕ) (h_g_even : Even g) (h_g_pos : 0 < g) (n : ℕ) :
  (∀ k ∈ ({3, 5, 7} : Set ℕ), (f g n ∣ f g (k * n))) ∧
  (∀ m : ℕ, 1 < m → Nat.Coprime (f g n) (f g (2 * m * n))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l584_58491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l584_58426

/-- The speed of a train given its length, time to pass a person, and the person's speed in the opposite direction. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed_kmph : ℝ) :
  train_length = 160 →
  passing_time = 7.384615384615384 →
  person_speed_kmph = 8 →
  ∃ (train_speed_kmph : ℝ), abs (train_speed_kmph - 70) < 0.1 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l584_58426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l584_58493

-- Define the triangle ABC
theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Given conditions
  Real.cos A = Real.sqrt 3 / 3 →
  c = Real.sqrt 3 →
  a = 3 * Real.sqrt 2 →
  -- Conclusions to prove
  Real.sin C = 1 / 3 ∧
  (1 / 2) * a * b * Real.sin C = 5 * Real.sqrt 6 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l584_58493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_triple_angle_l584_58466

theorem cosine_triple_angle (α : ℝ) : 
  Real.sin (π / 3 - α) = 1 / 3 → Real.cos (π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_triple_angle_l584_58466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circleplus_equals_one_l584_58414

-- Define the ⊕ operation
noncomputable def circleplus (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Define a function to represent the nested operation
noncomputable def nestedCircleplus : ℕ → ℝ
| 0 => 1000
| n + 1 => circleplus (n + 2 : ℝ) (nestedCircleplus n)

-- Theorem statement
theorem nested_circleplus_equals_one : circleplus 1 (nestedCircleplus 998) = 1 := by
  sorry

#check nested_circleplus_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circleplus_equals_one_l584_58414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_triangle_distance_between_foot_points_l584_58490

-- Define the three lines
def l₁ (x y : ℝ) : Prop := 4 * x + y - 4 = 0
def l₂ (m x y : ℝ) : Prop := m * x + y = 0
def l₃ (m x y : ℝ) : Prop := x - m * y - 4 = 0

-- Define the condition that the lines do not intersect at the same point
def not_intersect_at_same_point (m : ℝ) : Prop := sorry

-- Theorem 1: The lines cannot form a triangle iff m = 4 or m = -1/4
theorem cannot_form_triangle (m : ℝ) : 
  (¬∃ (x y z : ℝ), l₁ x y ∧ l₂ m x y ∧ l₃ m z y) ↔ (m = 4 ∨ m = -1/4) := by
  sorry

-- Theorem 2: When l₃ is perpendicular to both l₁ and l₂, the distance between foot points is 4√17/17
theorem distance_between_foot_points (m : ℝ) :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₂ m x₂ y₂ → l₃ m x₁ y₁ → l₃ m x₂ y₂ → 
    (x₂ - x₁) * 4 + (y₂ - y₁) = 0 ∧ (x₂ - x₁) * m + (y₂ - y₁) = 0) →
  m = -4 →
  (|4| / Real.sqrt (4^2 + 1) : ℝ) = 4 * Real.sqrt 17 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_form_triangle_distance_between_foot_points_l584_58490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_l584_58469

/-- The cost of fencing around a circular field -/
theorem fencing_cost (diameter : ℝ) (cost_per_meter : ℝ) (π : ℝ) :
  diameter = 16 →
  cost_per_meter = 3 →
  π = 3.14 →
  ∃ (total_cost : ℝ), abs (total_cost - (π * diameter * cost_per_meter)) < 0.01 ∧
                       abs (total_cost - 150.72) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_l584_58469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_complete_sets_l584_58428

/-- Represents the fabric allocation and production for student uniforms -/
structure UniformProduction where
  total_fabric : ℚ
  jacket_fabric : ℚ
  trouser_fabric : ℚ
  jackets_per_3m : ℚ
  trousers_per_3m : ℚ

/-- Calculates the number of complete sets that can be produced -/
noncomputable def complete_sets (prod : UniformProduction) : ℚ :=
  min ((prod.jacket_fabric / 3) * prod.jackets_per_3m) ((prod.trouser_fabric / 3) * prod.trousers_per_3m)

/-- Theorem stating the maximum number of complete sets that can be produced -/
theorem max_complete_sets (prod : UniformProduction) 
  (h1 : prod.total_fabric = 600)
  (h2 : prod.jackets_per_3m = 2)
  (h3 : prod.trousers_per_3m = 3)
  (h4 : prod.jacket_fabric + prod.trouser_fabric = prod.total_fabric)
  (h5 : 2 * prod.jacket_fabric = 3 * prod.trouser_fabric) :
  complete_sets prod = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_complete_sets_l584_58428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l584_58494

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumGeometric (seq : GeometricSequence) (n : ℕ) : ℝ :=
  if seq.q = 1 then
    n * seq.a 1
  else
    seq.a 1 * (1 - seq.q^n) / (1 - seq.q)

theorem geometric_sequence_ratio (seq : GeometricSequence) 
  (h : 8 * seq.a 2 + seq.a 5 = 0) : 
  sumGeometric seq 3 / sumGeometric seq 2 = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l584_58494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l584_58432

/-- The y-coordinate of the third vertex of an equilateral triangle -/
noncomputable def third_vertex_y_coord : ℝ := 3 + 3 * Real.sqrt 3

/-- Proof that the y-coordinate of the third vertex of an equilateral triangle is 3 + 3√3 -/
theorem equilateral_triangle_third_vertex :
  let vertex1 : ℝ × ℝ := (1, 3)
  let vertex2 : ℝ × ℝ := (7, 3)
  ∀ (vertex3 : ℝ × ℝ),
    vertex3.1 > 0 → -- First quadrant condition for x-coordinate
    vertex3.2 > 0 → -- First quadrant condition for y-coordinate
    (vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2 = 
      (vertex2.1 - vertex3.1)^2 + (vertex2.2 - vertex3.2)^2 → -- Side length equality
    (vertex3.1 - vertex1.1)^2 + (vertex3.2 - vertex1.2)^2 = 
      (vertex2.1 - vertex3.1)^2 + (vertex2.2 - vertex3.2)^2 → -- Side length equality
    vertex3.2 = third_vertex_y_coord :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l584_58432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l584_58440

/-- Calculates the speed of a train in km/h given its length, the platform length, and the time to cross the platform. -/
noncomputable def train_speed (train_length platform_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / time
  3.6 * speed_ms

/-- Theorem: The speed of a train with length 180 m crossing a platform of length 208.92 m in 20 seconds is approximately 70.0056 km/h. -/
theorem train_speed_calculation :
  let train_length : ℝ := 180
  let platform_length : ℝ := 208.92
  let time : ℝ := 20
  abs (train_speed train_length platform_length time - 70.0056) < 0.0001 := by
  sorry

-- Cannot use #eval with noncomputable functions
-- #eval train_speed 180 208.92 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l584_58440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_coincide_l584_58437

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  a : ℝ  -- width
  b : ℝ  -- height
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The center of a rectangle -/
noncomputable def Rectangle.center (r : Rectangle) : Point :=
  { x := r.a / 2, y := r.b / 2 }

/-- A function that constructs the new rectangle based on the original -/
noncomputable def construct_new_rectangle (r : Rectangle) : Rectangle :=
  { a := r.a, b := r.b, h_positive := r.h_positive }  -- Placeholder implementation

/-- Theorem stating that the centers of the original and new rectangles coincide -/
theorem centers_coincide (r : Rectangle) : 
  r.center = (construct_new_rectangle r).center := by
  sorry

#check centers_coincide

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_coincide_l584_58437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowball_tower_volume_l584_58447

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The total volume of three spheres with radii 4, 6, and 8 inches -/
noncomputable def total_volume : ℝ :=
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8

theorem snowball_tower_volume :
  total_volume = 1056 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowball_tower_volume_l584_58447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l584_58456

/-- The time taken by two workers to complete a task together, given their individual completion times -/
theorem combined_work_time (a_time b_time : ℝ) (ha : a_time > 0) (hb : b_time > 0) :
  (a_time = 30 ∧ b_time = 55) →
  1 / (1 / a_time + 1 / b_time) = 330 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l584_58456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l584_58480

theorem triangle_side_ratio_range (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < B ∧ B < Real.pi / 2 →  -- B is acute
  8 * Real.sin A * Real.sin C = (Real.sin B) ^ 2 →  -- Given condition
  Real.sqrt 5 / 2 < (a + c) / b ∧ (a + c) / b < Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l584_58480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_l584_58427

theorem cosine_sine_sum (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1) 
  (h2 : Real.sin x + Real.sin y + Real.sin z = 1) : 
  Real.cos (2*x) + Real.cos (2*y) + 2*Real.cos (2*z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_l584_58427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l584_58406

-- Define the constants as noncomputable
noncomputable def a : ℝ := (0.3 : ℝ) ^ (0.4 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (0.4 : ℝ)
noncomputable def c : ℝ := Real.log 2 / Real.log 0.3

-- State the theorem
theorem relationship_abc : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l584_58406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l584_58478

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def dom_f_shifted : Set ℝ := Set.Icc (-2) 1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x / Real.sqrt (2 * x + 1)

-- State the theorem
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range g} = Set.Ioc (-1/2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l584_58478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l584_58476

theorem min_abs_difference (x y : ℤ) (hx : x > 0) (hy : y > 0) (h : x * y - 2 * x + 5 * y = 111) :
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a * b - 2 * a + 5 * b = 111 ∧
  ∀ (c d : ℤ), c > 0 → d > 0 → c * d - 2 * c + 5 * d = 111 →
  |a - b| ≤ |c - d| ∧
  |a - b| = 93 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l584_58476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_birthday_l584_58465

/-- Represents a possible birthday --/
structure Birthday where
  day : Nat
  month : Nat
  h_day_le_month : day ≤ month
  h_valid_day : day ≤ 31
  h_valid_month : month ≤ 12

/-- Represents the knowledge state of Fames and Weven --/
def KnowledgeState (b : Birthday) : Nat → Prop
| 0 => ∃ m, m ≤ 12 ∧ b.day ≤ m
| 1 => ∃ d, d ≤ 31 ∧ d ≤ b.month
| (n+2) => KnowledgeState b n ∧ KnowledgeState b (n+1)

/-- The theorem stating that July 7th is the only possible birthday --/
theorem unique_birthday : 
  ∃! b : Birthday, 
    (∀ n, KnowledgeState b n) ∧ 
    (∀ b' : Birthday, (∀ n, KnowledgeState b' n) → b'.day < b.day ∨ b'.month < b.month) :=
  by
    -- Construct the unique birthday (July 7th)
    let july7 : Birthday := ⟨7, 7, by simp, by simp, by simp⟩
    
    -- Prove existence
    apply Exists.intro july7
    
    -- Prove uniqueness and other conditions
    sorry -- Detailed proof would go here

#check unique_birthday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_birthday_l584_58465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convexity_condition_convex_m_set_equiv_l584_58420

-- Define the function f(x) with parameter m
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/12) * x^4 - (m/6) * x^3 - (3/2) * x^2

-- Define the second derivative of f(x)
noncomputable def f_second_deriv (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - 3

-- State the theorem
theorem convexity_condition (m : ℝ) :
  (∀ x ∈ Set.Ioo 1 3, f_second_deriv m x < 0) ↔ m > 2 :=
sorry

-- Define the set of m values that satisfy the convexity condition
def convex_m_set : Set ℝ := {m | ∀ x ∈ Set.Ioo 1 3, f_second_deriv m x < 0}

-- State that the convex_m_set is equivalent to the interval [2, +∞)
theorem convex_m_set_equiv :
  convex_m_set = Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convexity_condition_convex_m_set_equiv_l584_58420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l584_58431

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → -3^x ≤ a) ↔ a ∈ Set.Ici (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l584_58431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_evaluation_l584_58468

theorem definite_integral_evaluation : 
  ∫ (x : ℝ) in Set.Icc 0 1, (2 + Real.sqrt (1 - x^2)) = π/4 + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_evaluation_l584_58468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expanded_sequence_is_arithmetic_l584_58438

/-- An arithmetic progression with a given first term, common difference, and length. -/
def ArithmeticProgression (a d : ℚ) (n : ℕ) := fun i : ℕ => a + d * i

/-- Inserted arithmetic means between two terms. -/
noncomputable def InsertedMeans (a b : ℚ) (p : ℕ) := 
  fun i : ℕ => a + (b - a) / (p + 1 : ℚ) * i

/-- The entire sequence after inserting means. -/
noncomputable def ExpandedSequence (ap : ℕ → ℚ) (p : ℕ) : ℕ → ℚ :=
  fun i => 
    let q := i / (p + 1)
    let r := i % (p + 1)
    if r = 0 then ap q
    else InsertedMeans (ap q) (ap (q + 1)) p r

/-- Theorem stating that the expanded sequence is also an arithmetic progression. -/
theorem expanded_sequence_is_arithmetic (a d : ℚ) (n p : ℕ) :
  ∃ (a' d' : ℚ), ∀ i, ExpandedSequence (ArithmeticProgression a d n) p i = a' + d' * i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expanded_sequence_is_arithmetic_l584_58438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_180_l584_58405

/-- The sum of all odd divisors of 180 is 78 -/
theorem sum_odd_divisors_180 : 
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors 180)).sum id = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_odd_divisors_180_l584_58405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_value_l584_58461

-- Define the function that f is symmetric to
noncomputable def g (x : ℝ) : ℝ := 2^(-x) - 1

-- Define the symmetry condition
def symmetric_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetric_function_value :
  ∀ f : ℝ → ℝ, symmetric_about_y_eq_x f g → f 3 = -2 :=
by
  intro f h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_value_l584_58461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_coprime_l584_58445

def T : ℕ → ℕ
  | 0 => 2  -- Adding the case for 0
  | 1 => 2
  | (n + 2) => T (n + 1) ^ 2 - T (n + 1) + 1

theorem T_coprime {m n : ℕ} (h : m ≠ n) : Nat.gcd (T m) (T n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_coprime_l584_58445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l584_58442

def a : ℕ → ℚ
  | 0 => 1/2  -- Add a case for 0
  | n + 1 => 3 * a n / (a n + 3)

theorem a_formula : ∀ n : ℕ, a n = 3 / (n + 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l584_58442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_iff_equal_power_l584_58425

def power_of_two : ℕ → ℕ
| 0 => 0
| n + 1 => if (n + 1) % 2 = 0 then 1 + power_of_two (n / 2 + 1) else 0

def valid_partition (a b : ℕ) : Prop :=
  ∃ (H₁ H₂ : Set ℕ), 
    (∀ n : ℕ, n > 0 → n ∈ H₁ ∨ n ∈ H₂) ∧
    (∀ x y : ℕ, x ∈ H₁ ∧ y ∈ H₁ → x - y ≠ a ∧ x - y ≠ b) ∧
    (∀ x y : ℕ, x ∈ H₂ ∧ y ∈ H₂ → x - y ≠ a ∧ x - y ≠ b)

theorem partition_iff_equal_power (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  valid_partition a b ↔ power_of_two a = power_of_two b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_iff_equal_power_l584_58425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l584_58453

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 3 → b = 4 → C = π / 3 → c^2 = a^2 + b^2 - 2*a*b*Real.cos C → c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l584_58453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_integers_sum_of_powers_l584_58458

theorem consecutive_even_integers_sum_of_powers (a b c : ℤ) : 
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) →  -- The integers are even
  (b = a + 2 ∧ c = b + 2) →               -- They are consecutive
  (a^2 + b^2 + c^2 = 2450) →              -- Sum of squares is 2450
  (a^4 + b^4 + c^4 = 1881632) :=          -- Sum of fourth powers is 1881632
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_integers_sum_of_powers_l584_58458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_membership_problem_l584_58455

theorem set_membership_problem (U A B : Finset Nat) : 
  (Finset.card U = 192) →
  (Finset.card (U \ (A ∪ B)) = 59) →
  (Finset.card (A ∩ B) = 23) →
  (Finset.card A = 107) →
  (Finset.card B = 49) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_membership_problem_l584_58455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_areas_l584_58441

/-- Frustum of a right circular cone -/
structure Frustum where
  r1 : ℝ  -- lower base radius
  r2 : ℝ  -- upper base radius
  h : ℝ   -- height
  h_pos : h > 0
  r1_pos : r1 > 0
  r2_pos : r2 > 0
  r1_gt_r2 : r1 > r2

/-- The lateral surface area of a frustum -/
noncomputable def lateralSurfaceArea (f : Frustum) : ℝ :=
  Real.pi * (f.r1 + f.r2) * Real.sqrt (f.h^2 + (f.r1 - f.r2)^2)

/-- The total surface area of a frustum -/
noncomputable def totalSurfaceArea (f : Frustum) : ℝ :=
  lateralSurfaceArea f + Real.pi * (f.r1^2 + f.r2^2)

/-- Theorem stating the lateral and total surface areas of a specific frustum -/
theorem frustum_surface_areas :
  let f : Frustum := ⟨10, 4, 9, by norm_num, by norm_num, by norm_num, by norm_num⟩
  (lateralSurfaceArea f = 42 * Real.pi * Real.sqrt 13) ∧
  (totalSurfaceArea f = 116 * Real.pi + 42 * Real.pi * Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_areas_l584_58441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S₈_eq_neg_85_l584_58434

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (n : ℕ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
noncomputable def q : ℝ := sorry

/-- The first term of the geometric sequence -/
noncomputable def a₁ : ℝ := sorry

/-- Geometric sequence sum formula -/
axiom geom_sum_formula (n : ℕ) : S n = a₁ * (1 - q^n) / (1 - q)

/-- Condition: q ≠ 1 to avoid triviality -/
axiom q_neq_one : q ≠ 1

/-- Given condition: S₄ = -5 -/
axiom S₄_eq_neg_five : S 4 = -5

/-- Given condition: S₆ = 21S₂ -/
axiom S₆_eq_21S₂ : S 6 = 21 * S 2

/-- Theorem: If S₄ = -5 and S₆ = 21S₂, then S₈ = -85 -/
theorem S₈_eq_neg_85 : S 8 = -85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S₈_eq_neg_85_l584_58434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_changes_l584_58408

theorem faculty_changes (F : ℝ) : 
  F * (1 - 0.075) * (1 + 0.125) * (1 - 0.0325) * (1 + 0.098) * (1 - 0.1465) = 195 →
  ⌊F⌋ = 244 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_changes_l584_58408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_median_intersection_l584_58492

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the centroid of a triangle -/
noncomputable def triangleCentroid (A B C : Point3D) : Point3D :=
  { x := (A.x + B.x + C.x) / 3
  , y := (A.y + B.y + C.y) / 3
  , z := (A.z + B.z + C.z) / 3 }

/-- Calculates the centroid of a tetrahedron -/
noncomputable def tetrahedronCentroid (t : Tetrahedron) : Point3D :=
  { x := (t.A.x + t.B.x + t.C.x + t.D.x) / 4
  , y := (t.A.y + t.B.y + t.C.y + t.D.y) / 4
  , z := (t.A.z + t.B.z + t.C.z + t.D.z) / 4 }

/-- Represents a median of a tetrahedron -/
structure Median where
  vertex : Point3D
  oppositeFaceCentroid : Point3D

/-- Theorem: The four medians of a tetrahedron intersect at a single point,
    which divides each median in a 3:1 ratio from the vertex -/
theorem tetrahedron_median_intersection (t : Tetrahedron) :
  let medians := [
    Median.mk t.A (triangleCentroid t.B t.C t.D),
    Median.mk t.B (triangleCentroid t.A t.C t.D),
    Median.mk t.C (triangleCentroid t.A t.B t.D),
    Median.mk t.D (triangleCentroid t.A t.B t.C)
  ]
  ∃ (intersectionPoint : Point3D),
    (∀ m : Median, m ∈ medians →
      intersectionPoint = tetrahedronCentroid t) ∧
    (∀ m : Median, m ∈ medians →
      ∃ (r : ℝ), r = 3/4 ∧
        intersectionPoint.x = r * m.vertex.x + (1 - r) * m.oppositeFaceCentroid.x ∧
        intersectionPoint.y = r * m.vertex.y + (1 - r) * m.oppositeFaceCentroid.y ∧
        intersectionPoint.z = r * m.vertex.z + (1 - r) * m.oppositeFaceCentroid.z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_median_intersection_l584_58492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bills_gross_salary_l584_58421

/-- Represents Bill's financial situation -/
structure BillFinances where
  takehome : ℚ
  property_tax : ℚ
  sales_tax : ℚ
  income_tax_rate : ℚ

/-- Calculates the gross salary given Bill's financial situation -/
def calculate_gross_salary (bf : BillFinances) : ℚ :=
  (bf.takehome + bf.property_tax + bf.sales_tax) / (1 - bf.income_tax_rate)

/-- Theorem stating that Bill's gross salary is $50,000 given the conditions -/
theorem bills_gross_salary :
  let bf : BillFinances := {
    takehome := 40000,
    property_tax := 2000,
    sales_tax := 3000,
    income_tax_rate := 1/10
  }
  calculate_gross_salary bf = 50000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bills_gross_salary_l584_58421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l584_58444

/-- Theorem: Equation of a circle
    For any point (x, y) on a circle with radius R and center (a, b),
    the equation (x-a)^2 + (y-b)^2 = R^2 holds. -/
theorem circle_equation 
  (R : ℝ) (a b x y : ℝ) 
  (h : (x - a)^2 + (y - b)^2 = R^2) : 
  (∃ (t : ℝ), x = R * Real.cos t + a ∧ y = R * Real.sin t + b) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l584_58444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l584_58464

theorem triangle_angle_inequality (α β γ : ℝ) (h_triangle : α + β + γ = Real.pi) :
  (Real.cos α) / (Real.sin β * Real.sin γ) + 
  (Real.cos β) / (Real.sin γ * Real.sin α) + 
  (Real.cos γ) / (Real.sin α * Real.sin β) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l584_58464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l584_58429

-- Define the hyperbola
def is_hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote slope
noncomputable def asymptote_slope (a b : ℝ) : ℝ := b / a

-- Define the angle between asymptotes
noncomputable def angle_between_asymptotes (a b : ℝ) : ℝ :=
  Real.arctan ((2 * (asymptote_slope a b)) / (1 - (asymptote_slope a b)^2))

-- Theorem statement
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  angle_between_asymptotes a b = π/4 → a/b = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l584_58429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_n_values_l584_58497

-- Define the hyperbola equation
def hyperbola_equation (x y n : ℝ) : Prop :=
  x^2 / n + y^2 / (12 - n) = -1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 3

-- Theorem statement
theorem hyperbola_n_values :
  ∃ (n : ℝ), (∀ (x y : ℝ), hyperbola_equation x y n → 
    (n = -12 ∨ n = 24)) ∧
  eccentricity = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_n_values_l584_58497
