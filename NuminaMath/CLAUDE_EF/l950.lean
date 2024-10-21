import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_siskins_l950_95021

/-- Represents a row of poles where siskins can land -/
structure PoleRow where
  num_poles : Nat
  occupied : List Bool

/-- A siskin can land on an empty pole -/
def can_land (row : PoleRow) (pole : Nat) : Prop :=
  pole < row.num_poles ∧ pole < row.occupied.length ∧ ¬row.occupied[pole]!

/-- When a siskin lands, a siskin on a neighboring pole (if any) takes off -/
def neighbor_takes_off (row : PoleRow) (pole : Nat) : PoleRow :=
  sorry

/-- No more than one siskin can sit on each pole at a time -/
def valid_occupation (row : PoleRow) : Prop :=
  row.occupied.length = row.num_poles ∧
  ∀ i, i < row.num_poles → (row.occupied[i]! = true ∨ row.occupied[i]! = false)

/-- The number of occupied poles -/
def num_occupied (row : PoleRow) : Nat :=
  (row.occupied.filter id).length

/-- The maximum number of siskins that can simultaneously occupy the poles -/
theorem max_siskins (row : PoleRow) (h : row.num_poles = 25) :
  ∃ (max_occ : Nat), ∀ (valid_row : PoleRow),
    valid_row.num_poles = 25 →
    valid_occupation valid_row →
    num_occupied valid_row ≤ max_occ ∧
    max_occ = 24 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_siskins_l950_95021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_k_representation_of_7_51_l950_95011

theorem base_k_representation_of_7_51 :
  ∃ (k : ℕ+), 
    (k : ℝ) > 1 ∧ 
    (∃ (a b : ℕ), a < k ∧ b < k ∧
      (7 : ℝ) / 51 = (a : ℝ) / k + (b : ℝ) / k^2 + (a : ℝ) / k^3 + (b : ℝ) / k^4 + (a : ℝ) / k^5 + (b : ℝ) / k^6) →
    k = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_k_representation_of_7_51_l950_95011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_four_circles_l950_95019

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents four coplanar circles -/
def FourCoplanarCircles := Fin 4 → Circle

/-- Check if a point lies on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem max_intersection_points_four_circles (circles : FourCoplanarCircles) (l : Line) :
  ∃ (n : ℕ), n ≤ 8 ∧ 
  (∀ (m : ℕ), (∃ (points : Fin m → ℝ × ℝ), 
    (∀ i, ∃ j, point_on_circle (points i) (circles j)) ∧ 
    (∀ i, point_on_line (points i) l)) → m ≤ n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_four_circles_l950_95019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unclaimed_stickers_l950_95037

noncomputable section

-- Define the total pile of stickers
variable (x : ℝ)

-- Define the fractions each person should get based on the ratio
noncomputable def al_share : ℝ := 4/9
noncomputable def bert_share : ℝ := 1/3
noncomputable def carl_share : ℝ := 2/9

-- Define what each person actually takes
noncomputable def al_takes : ℝ := al_share
noncomputable def bert_takes : ℝ := (1 - al_share) * bert_share
noncomputable def carl_takes : ℝ := (1 - al_share - bert_takes) * carl_share

-- Theorem statement
theorem unclaimed_stickers :
  1 - al_takes - bert_takes - carl_takes = 230/243 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unclaimed_stickers_l950_95037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_savings_percentage_l950_95074

noncomputable def net_salary : ℚ := 3700

noncomputable def discretionary_income : ℚ := net_salary / 5

noncomputable def vacation_fund_percentage : ℚ := 30
noncomputable def eating_out_percentage : ℚ := 35
noncomputable def gifts_charitable_amount : ℚ := 111

theorem jill_savings_percentage :
  let savings_percentage := 100 - (vacation_fund_percentage + eating_out_percentage + (gifts_charitable_amount / discretionary_income * 100))
  savings_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_savings_percentage_l950_95074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l950_95033

/-- A parabola with vertex at origin and focus at (0,2) -/
structure Parabola where
  vertex : ℝ × ℝ := (0, 0)
  focus : ℝ × ℝ := (0, 2)

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  first_quadrant : 0 < point.1 ∧ 0 < point.2
  on_parabola : (point.1^2 : ℝ) = 8 * point.2

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_point_coordinates (p : Parabola) 
  (P : PointOnParabola p) (h : distance P.point p.focus = 50) :
  P.point = (8 * Real.sqrt 6, 48) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_coordinates_l950_95033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_numbers_l950_95096

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≤ 9
  h_ones : ones ≤ 9
  h_not_zero : tens ≠ 0

/-- The reverse of a two-digit number -/
def reverse (n : TwoDigitNumber) : Nat :=
  10 * n.ones + n.tens

/-- The sum of a number and its reverse equals 264 -/
def sumWithReverse (n : TwoDigitNumber) : Prop :=
  (10 * n.tens + n.ones) + reverse n = 264

/-- The sum of digits is even -/
def sumOfDigitsEven (n : TwoDigitNumber) : Prop :=
  Even (n.tens + n.ones)

theorem no_valid_numbers :
  ¬∃ (n : TwoDigitNumber), sumWithReverse n ∧ sumOfDigitsEven n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_numbers_l950_95096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_bathroom_visits_l950_95089

/-- Calculates the number of bathroom visits during a movie with intermissions -/
def bathroom_visits (movie_duration : ℕ) (intermission_duration : ℕ) (intermission_count : ℕ) 
  (bathroom_interval : ℕ) (walking_time : ℕ) : ℕ :=
  let total_duration := movie_duration + intermission_duration * intermission_count
  let effective_interval := bathroom_interval + walking_time
  (total_duration / effective_interval) + intermission_count

/-- Theorem stating that for the given movie scenario, the number of bathroom visits is 5 -/
theorem movie_bathroom_visits :
  bathroom_visits 150 15 2 50 5 = 5 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_bathroom_visits_l950_95089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l950_95068

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci F₁ and F₂
noncomputable def left_focus (a b : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 - b^2), 0)
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

-- Define vertices A and B
def left_vertex (a : ℝ) : ℝ × ℝ := (-a, 0)
def top_vertex (b : ℝ) : ℝ × ℝ := (0, b)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define a right triangle
def is_right_triangle (p q r : ℝ × ℝ) : Prop :=
  (distance p q)^2 + (distance q r)^2 = (distance p r)^2

-- Define the intersection of a line with the ellipse
def line_intersect_ellipse (k : ℝ) (a b : ℝ) (x : ℝ) : Prop :=
  ellipse a b x (k * x + 2)

-- Define perpendicularity of two vectors
def perpendicular (p q : ℝ × ℝ) : Prop :=
  p.1 * q.1 + p.2 * q.2 = 0

theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : distance (left_vertex a) (top_vertex b) = Real.sqrt 3)
  (h4 : is_right_triangle (top_vertex b) (left_focus a b) (right_focus a b))
  (k : ℝ) (h5 : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ line_intersect_ellipse k a b x₁ ∧ line_intersect_ellipse k a b x₂)
  (h6 : perpendicular (x₁, k * x₁ + 2) (x₂, k * x₂ + 2)) :
  (a = Real.sqrt 2 ∧ b = 1) ∧ (k = Real.sqrt 5 ∨ k = -Real.sqrt 5) := by
  sorry

#check ellipse_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l950_95068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dotProduct_range_l950_95097

/-- The ellipse with equation x²/4 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- The foci of the ellipse -/
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 3, 0)

/-- The dot product of vectors PF₁ and PF₂ -/
noncomputable def dotProduct (p : ℝ × ℝ) : ℝ :=
  (p.1 - F1.1) * (p.1 - F2.1) + (p.2 - F1.2) * (p.2 - F2.2)

theorem dotProduct_range :
  ∀ p ∈ Ellipse, -2 ≤ dotProduct p ∧ dotProduct p ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dotProduct_range_l950_95097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_in_row_15_l950_95030

def pascal_row (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (Nat.choose n)

theorem fifth_number_in_row_15 :
  (pascal_row 15).get? 4 = some 1365 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_in_row_15_l950_95030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_yield_l950_95053

-- Define the chemical reaction
structure Reaction where
  ch4 : ℚ
  co2 : ℚ
  c3h8 : ℚ
  h2o : ℚ

-- Define the stoichiometric ratios
def stoichiometric_ratio : Reaction :=
  { ch4 := 2
  , co2 := 5
  , c3h8 := 1
  , h2o := 4 }

-- Define the initial amounts of reactants
def initial_amounts : Reaction :=
  { ch4 := 3
  , co2 := 4
  , c3h8 := 5
  , h2o := 0 }

-- Define the theoretical yield percentage
def theoretical_yield : ℚ := 95 / 100

-- Function to calculate the limiting reactant
def limiting_reactant (r : Reaction) : String :=
  let ch4_limit := r.ch4 * stoichiometric_ratio.h2o / stoichiometric_ratio.ch4
  let co2_limit := r.co2 * stoichiometric_ratio.h2o / stoichiometric_ratio.co2
  let c3h8_limit := r.c3h8 * stoichiometric_ratio.h2o / stoichiometric_ratio.c3h8
  if ch4_limit ≤ co2_limit ∧ ch4_limit ≤ c3h8_limit then "CH4"
  else if co2_limit ≤ ch4_limit ∧ co2_limit ≤ c3h8_limit then "CO2"
  else "C3H8"

-- Function to calculate the actual yield of H2O
def actual_yield (r : Reaction) : ℚ :=
  let limiting := limiting_reactant r
  let yield := match limiting with
    | "CH4" => r.ch4 * stoichiometric_ratio.h2o / stoichiometric_ratio.ch4
    | "CO2" => r.co2 * stoichiometric_ratio.h2o / stoichiometric_ratio.co2
    | "C3H8" => r.c3h8 * stoichiometric_ratio.h2o / stoichiometric_ratio.c3h8
    | _ => 0
  yield * theoretical_yield

-- Theorem statement
theorem reaction_yield :
  limiting_reactant initial_amounts = "CO2" ∧
  actual_yield initial_amounts = 304 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_yield_l950_95053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_nth_root_l950_95076

theorem existence_of_nth_root (b n : ℕ) (hb : b > 1) (hn : n > 1)
  (h : ∀ k : ℕ, k > 1 → ∃ a_k : ℕ, k ∣ (b - a_k^n)) :
  ∃ A : ℕ, b = A^n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_nth_root_l950_95076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_supplementary_angles_l950_95056

theorem sin_supplementary_angles (α : ℝ) :
  Real.sin (π / 4 + α) = Real.sqrt 3 / 2 →
  Real.sin (3 * π / 4 - α) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_supplementary_angles_l950_95056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_fill_time_approx_l950_95023

/-- The length of the box in feet -/
noncomputable def box_length : ℝ := 10

/-- The width of the box in feet -/
noncomputable def box_width : ℝ := 8

/-- The depth of the box in feet -/
noncomputable def box_depth : ℝ := 4

/-- The rate at which sand is poured into the box in cubic feet per hour -/
noncomputable def pour_rate : ℝ := 6

/-- The volume of the box in cubic feet -/
noncomputable def box_volume : ℝ := box_length * box_width * box_depth

/-- The time it takes to fill the box in hours -/
noncomputable def fill_time : ℝ := box_volume / pour_rate

/-- Theorem stating that the fill time is approximately 53.33 hours -/
theorem box_fill_time_approx : 
  ∃ ε > 0, |fill_time - 53.33| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_fill_time_approx_l950_95023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_minus_pi_6_properties_l950_95080

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem sin_2x_minus_pi_6_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧
    T = Real.pi ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3))) ∧
    (∀ k : ℤ, StrictAntiOn f (Set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_minus_pi_6_properties_l950_95080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l950_95079

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (2 * α) = (2 * Real.sqrt 5 / 5) * Real.sin (α + π / 4)) : 
  Real.tan α = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l950_95079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l950_95015

-- Define the domain for both functions
def Domain := {x : ℝ | x > 0}

-- Define the two functions
noncomputable def f (x : Domain) : ℝ := Real.log (Real.sqrt x.val) / Real.log 2
noncomputable def g (x : Domain) : ℝ := (1/2) * (Real.log x.val / Real.log 2)

-- Theorem stating that the two functions are equal
theorem f_equals_g : ∀ x : Domain, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l950_95015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_transformation_l950_95093

-- Define the original inequality
def original_inequality (k a b c x : ℝ) : Prop :=
  (k / (x + a) + (x + b) / (x + c)) < 0

-- Define the solution set of the original inequality
def original_solution_set : Set ℝ :=
  Set.union (Set.Ioo (-1) (-1/3)) (Set.Ioo (1/2) 1)

-- Define the transformed inequality
def transformed_inequality (k a b c x : ℝ) : Prop :=
  (k * x / (a * x + 1) + (b * x + 1) / (c * x + 1)) < 0

-- Define the solution set of the transformed inequality
def transformed_solution_set : Set ℝ :=
  Set.union (Set.Ioo (-3) (-1)) (Set.Ioo 1 2)

-- State the theorem
theorem inequality_transformation (k a b c : ℝ) :
  (∀ x, original_inequality k a b c x ↔ x ∈ original_solution_set) →
  (∀ x, transformed_inequality k a b c x ↔ x ∈ transformed_solution_set) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_transformation_l950_95093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_side_heads_probability_l950_95034

-- Define the three types of coins
inductive Coin
| Normal
| DoubleHeads
| DoubleTails

-- Define the result of a coin flip
inductive FlipResult
| Heads
| Tails

-- Define the probability of selecting each coin
noncomputable def selectProb (c : Coin) : ℝ := 1 / 3

-- Define the probability of getting heads for each coin
noncomputable def headsProb (c : Coin) : ℝ :=
  match c with
  | Coin.Normal => 1 / 2
  | Coin.DoubleHeads => 1
  | Coin.DoubleTails => 0

-- Define the probability of the other side being heads given that heads was observed
noncomputable def otherSideHeadsProb : ℝ :=
  let totalHeadsProb := (headsProb Coin.Normal * selectProb Coin.Normal) +
                        (headsProb Coin.DoubleHeads * selectProb Coin.DoubleHeads) +
                        (headsProb Coin.DoubleTails * selectProb Coin.DoubleTails)
  let doubleHeadsContribution := headsProb Coin.DoubleHeads * selectProb Coin.DoubleHeads
  doubleHeadsContribution / totalHeadsProb

-- Theorem statement
theorem other_side_heads_probability :
  otherSideHeadsProb = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_side_heads_probability_l950_95034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_properties_l950_95052

structure TriangularPyramid where
  base : Set (ℝ × ℝ × ℝ)
  apex : ℝ × ℝ × ℝ
  volume : ℝ
  projection_height : ℝ

def is_right_angled_triangle (t : Set (ℝ × ℝ × ℝ)) : Prop := sorry

def lies_on_sphere (point : ℝ × ℝ × ℝ) (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop := sorry

def is_projection (point : ℝ × ℝ × ℝ) (base : Set (ℝ × ℝ × ℝ)) (apex : ℝ × ℝ × ℝ) : Prop := sorry

def is_midpoint (point : ℝ × ℝ × ℝ) (segment : Set (ℝ × ℝ × ℝ)) : Prop := sorry

noncomputable def surface_area_sphere (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

noncomputable def volume_sphere (radius : ℝ) : ℝ := (4/3) * Real.pi * radius^3

def hypotenuse (t : Set (ℝ × ℝ × ℝ)) : Set (ℝ × ℝ × ℝ) := sorry

theorem pyramid_sphere_properties (p : TriangularPyramid) 
  (h_base : is_right_angled_triangle p.base)
  (h_volume : p.volume = 4)
  (h_projection : p.projection_height = 3)
  (O : ℝ × ℝ × ℝ) (r : ℝ)
  (h_sphere : ∀ v ∈ p.base ∪ {p.apex}, lies_on_sphere v O r) :
  (∃ K, is_projection K p.base p.apex ∧
    (((K ∈ p.base → r ≥ 5/2) ∧
    (is_midpoint K (hypotenuse p.base) → r ≥ 13/6)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_properties_l950_95052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mean_after_removal_l950_95048

def original_count : ℕ := 60
def original_mean : ℚ := 47
def removed_numbers : List ℚ := [50, 60]

theorem new_mean_after_removal :
  let original_sum : ℚ := original_count * original_mean
  let removed_sum : ℚ := removed_numbers.sum
  let new_sum : ℚ := original_sum - removed_sum
  let new_count : ℕ := original_count - removed_numbers.length
  let new_mean : ℚ := new_sum / new_count
  (Int.floor (new_mean + 1/2) : ℤ) = 47 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_mean_after_removal_l950_95048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_AB_AC_l950_95051

def A : Fin 3 → ℝ := ![(-4), 3, 0]
def B : Fin 3 → ℝ := ![0, 1, 3]
def C : Fin 3 → ℝ := ![(-2), 4, (-2)]

def AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
def AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem cosine_angle_AB_AC :
  dot_product AB AC / (magnitude AB * magnitude AC) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_AB_AC_l950_95051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nedy_crackers_l950_95073

/-- The number of packs of crackers Nedy ate from Monday to Thursday -/
def crackers_mon_to_thu : ℕ := by sorry

/-- The number of packs of crackers Nedy ate on Friday -/
def crackers_friday : ℕ := 2 * crackers_mon_to_thu

/-- The total number of crackers Nedy ate from Monday to Friday -/
def total_crackers : ℕ := crackers_mon_to_thu + crackers_friday

theorem nedy_crackers : 
  total_crackers = 48 → crackers_mon_to_thu = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nedy_crackers_l950_95073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_theorem_l950_95024

/-- Represents a line in 3D space --/
structure Line3D where
  direction : Fin 3 → ℝ
  point : Fin 3 → ℝ

/-- Represents the number of intersecting lines --/
inductive NumIntersectingLines
  | Zero
  | Four

/-- 
Given two perpendicular skew lines and an acute angle, 
returns the number of lines that intersect both skew lines 
and form the given angle with each of them
-/
noncomputable def countIntersectingLines (line1 line2 : Line3D) (α : ℝ) : NumIntersectingLines :=
  sorry

/-- The main theorem to be proved --/
theorem intersecting_lines_theorem 
  (line1 line2 : Line3D) 
  (h_perp : (line1.direction 0) * (line2.direction 0) + 
            (line1.direction 1) * (line2.direction 1) + 
            (line1.direction 2) * (line2.direction 2) = 0)
  (h_skew : ∃ (t s : ℝ), 
    (line1.point 0) + t * (line1.direction 0) ≠ (line2.point 0) + s * (line2.direction 0) ∨
    (line1.point 1) + t * (line1.direction 1) ≠ (line2.point 1) + s * (line2.direction 1) ∨
    (line1.point 2) + t * (line1.direction 2) ≠ (line2.point 2) + s * (line2.direction 2))
  (α : ℝ)
  (h_acute : 0 < α ∧ α < Real.pi / 2) :
  countIntersectingLines line1 line2 α = 
    if α ≤ Real.pi / 4 then NumIntersectingLines.Zero else NumIntersectingLines.Four :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_theorem_l950_95024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_of_isosceles_triangle_l950_95060

/-- An isosceles triangle with two sides of length 7 and one side of length 10 -/
structure IsoscelesTriangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  is_isosceles : side_a = side_b ∧ side_a = 7 ∧ side_c = 10

/-- The inradius of a triangle -/
noncomputable def inradius (t : IsoscelesTriangle) : ℝ :=
  let s := (t.side_a + t.side_b + t.side_c) / 2
  let area := Real.sqrt (s * (s - t.side_a) * (s - t.side_b) * (s - t.side_c))
  area / s

/-- The theorem stating that the inradius of the given isosceles triangle is approximately 2.04125 -/
theorem inradius_of_isosceles_triangle :
  ∃ (t : IsoscelesTriangle), abs (inradius t - 2.04125) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_of_isosceles_triangle_l950_95060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_difference_l950_95083

def odd_set : Finset ℕ := Finset.filter (λ x => x % 2 = 1 ∧ x % 7 ≠ 0) (Finset.range 46 \ Finset.range 11)
def even_set : Finset ℕ := Finset.filter (λ x => x % 2 = 0 ∧ x % 9 ≠ 0 ∧ x % 13 ≠ 0) (Finset.range 53 \ Finset.range 16)

def odd_avg : ℚ := (Finset.sum odd_set id) / (odd_set.card : ℚ)
def even_avg : ℚ := (Finset.sum even_set id) / (even_set.card : ℚ)

theorem avg_difference : even_avg - odd_avg = 46 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_difference_l950_95083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_plus_one_l950_95066

def y : ℕ → ℝ
  | 0 => 150  -- Adding the base case for 0
  | 1 => 150
  | (k + 2) => y (k + 1) ^ 2 - y (k + 1)

theorem sum_reciprocal_y_plus_one :
  ∑' k, 1 / (y k + 1) = 1 / 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_plus_one_l950_95066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l950_95098

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 1

theorem inverse_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, Function.RightInverse g (f a) ∧ g 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l950_95098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natalie_bushes_needed_l950_95042

/-- The number of containers of blueberries yielded by each bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_for_trade : ℕ := 4

/-- The number of zucchinis received in trade for containers_for_trade containers -/
def zucchinis_received : ℕ := 3

/-- The target number of zucchinis Natalie wants to have -/
def target_zucchinis : ℕ := 72

/-- The minimum number of bushes Natalie needs to pick -/
def min_bushes : ℕ := 10

/-- Helper function to convert ℚ to ℕ by ceiling -/
noncomputable def ceilToNat (q : ℚ) : ℕ := Int.toNat ⌈q⌉

theorem natalie_bushes_needed :
  min_bushes = ceilToNat ((target_zucchinis : ℚ) / ((containers_per_bush : ℚ) * (zucchinis_received : ℚ) / (containers_for_trade : ℚ))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natalie_bushes_needed_l950_95042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_ratio_proof_l950_95075

theorem original_ratio_proof (x y : ℚ) (h1 : y = 15) (h2 : (x + 10) / y = 1) :
  x / y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_ratio_proof_l950_95075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_d_value_is_two_l950_95088

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a 6-digit number of the form 7d7,33e -/
def SpecialNumber (d e : Digit) : ℕ :=
  700000 + 10000 * d.val + 7000 + 330 + e.val

/-- Condition for the number to be divisible by 33 -/
def IsDivisibleBy33 (d e : Digit) : Prop :=
  (SpecialNumber d e) % 33 = 0

theorem max_d_value_is_two :
  ∀ d : Digit, (∃ e : Digit, IsDivisibleBy33 d e) →
  d.val ≤ 2 ∧ ∃ e : Digit, IsDivisibleBy33 (⟨2, by norm_num⟩ : Digit) e :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_d_value_is_two_l950_95088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_19_l950_95007

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define the original triangle vertices
noncomputable def A : Point := ⟨2, 3⟩
noncomputable def B : Point := ⟨5, 7⟩
noncomputable def C : Point := ⟨6, 2⟩

-- Define the reflection function across y = x
def reflect (p : Point) : Point :=
  ⟨p.y, p.x⟩

-- Calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

-- Theorem statement
theorem area_of_union_equals_19 :
  let originalArea := triangleArea A B C
  let reflectedArea := triangleArea (reflect A) (reflect B) (reflect C)
  originalArea + reflectedArea = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_19_l950_95007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_business_break_even_l950_95087

/-- Calculates the break-even point in units for a business -/
noncomputable def break_even_point (fixed_cost : ℝ) (variable_cost : ℝ) (selling_price : ℝ) : ℝ :=
  fixed_cost / (selling_price - variable_cost)

/-- Proves that the break-even point for the game manufacturing business is 600 units -/
theorem game_business_break_even :
  let fixed_cost : ℝ := 10410
  let variable_cost : ℝ := 2.65
  let selling_price : ℝ := 20
  ⌈break_even_point fixed_cost variable_cost selling_price⌉ = 600 := by
  sorry

#check game_business_break_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_business_break_even_l950_95087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_is_15_l950_95010

/-- The weight difference between a double bed and a single bed -/
noncomputable def weight_difference : ℝ :=
  let single_bed_weight := 50 / 5
  let bunk_bed_weight := 60 / 3
  let double_bed_weight := (180 - 2 * single_bed_weight - 3 * bunk_bed_weight) / 4
  double_bed_weight - single_bed_weight

/-- Theorem stating the weight difference between a double bed and a single bed -/
theorem weight_difference_is_15 : weight_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_is_15_l950_95010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_25n_count_l950_95036

theorem divisible_25n_count : 
  (Finset.filter (fun n : ℕ => 1 ≤ n ∧ n ≤ 9 ∧ (25 * n) % n = 0) (Finset.range 10)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_25n_count_l950_95036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l950_95016

/-- A pyramid with an equilateral triangular base and one lateral edge perpendicular to the base -/
structure Pyramid where
  a : ℝ  -- Side length of the equilateral triangular base
  b : ℝ  -- Length of the lateral edge perpendicular to the base
  a_pos : 0 < a  -- Side length is positive
  b_pos : 0 < b  -- Lateral edge length is positive

/-- The radius of the circumscribed sphere around the pyramid -/
noncomputable def circumscribed_sphere_radius (p : Pyramid) : ℝ :=
  (Real.sqrt (12 * p.a^2 + 9 * p.b^2)) / 6

/-- Theorem stating that the radius of the circumscribed sphere is as calculated -/
theorem circumscribed_sphere_radius_formula (p : Pyramid) :
  circumscribed_sphere_radius p = (Real.sqrt (12 * p.a^2 + 9 * p.b^2)) / 6 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l950_95016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_problem_solution_l950_95017

/-- The cost price of a watch given specific selling conditions -/
def watch_cost_price (loss_percent : ℝ) (gain_percent : ℝ) (price_difference : ℝ) : ℝ :=
  let cost_price : ℝ := 2000
  let original_selling_price := (1 - loss_percent / 100) * cost_price
  let new_selling_price := (1 + gain_percent / 100) * cost_price
  cost_price

/-- The specific instance of the watch problem -/
theorem watch_problem_solution :
  watch_cost_price 10 4 280 = 2000 := by
  -- Unfold the definition of watch_cost_price
  unfold watch_cost_price
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_problem_solution_l950_95017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_power_function_l950_95046

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the function f(x) = 1/x^2
noncomputable def f (x : ℝ) : ℝ := 1 / (x ^ 2)

-- Theorem statement
theorem f_is_power_function : isPowerFunction f := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_power_function_l950_95046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_MN_length_l950_95082

/-- Represents a trapezoid with side lengths a, b, c, d -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

/-- The length of MN in a trapezoid, where M and N are intersections of angle bisectors -/
noncomputable def length_MN (t : Trapezoid) : ℝ := (1/2) * |t.b + t.d - t.a - t.c|

/-- Theorem stating that the length of MN in a trapezoid is (1/2)|b + d - a - c| -/
theorem trapezoid_MN_length (t : Trapezoid) : 
  ∃ (M N : ℝ × ℝ), length_MN t = dist M N := by
  sorry

#check trapezoid_MN_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_MN_length_l950_95082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_problem_l950_95063

theorem coin_problem (total_value : ℚ) (h1 : total_value = 70) :
  ∃ (x : ℕ),
    (x : ℚ) + (x : ℚ) / 2 + (x : ℚ) / 4 = total_value ∧
    x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_problem_l950_95063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_alpha_sin_cos_theta_l950_95005

open Real

noncomputable section

-- Part 1
variable (α : ℝ) (x : ℝ)

axiom alpha_range : 90 * π / 180 < α ∧ α < 180 * π / 180
axiom cos_alpha : cos α = sqrt 2 / 4 * x
axiom point_on_terminal_side : x^2 + 5 = (cos α)^2 + (sin α)^2

theorem sin_tan_alpha :
  sin α = sqrt 10 / 4 ∧ tan α = -sqrt 15 / 3 := by sorry

-- Part 2
variable (θ : ℝ) (x_theta : ℝ)

axiom x_theta_nonzero : x_theta ≠ 0
axiom tan_theta : tan θ = -x_theta
axiom point_on_terminal_side_theta : x_theta^2 + (-1)^2 = (cos θ)^2 + (sin θ)^2

theorem sin_cos_theta :
  (sin θ = -sqrt 2 / 2 ∧ cos θ = sqrt 2 / 2) ∨
  (sin θ = -sqrt 2 / 2 ∧ cos θ = -sqrt 2 / 2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_tan_alpha_sin_cos_theta_l950_95005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_subset_from_intersection_l950_95084

-- Proposition 1
theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x ≠ 0 ∧ Real.sin x = 0) ∧
  (∀ x : ℝ, x = 0 → Real.sin x = 0) := by
  sorry

-- Proposition 4
theorem subset_from_intersection {α : Type*} (A B : Set α) :
  A ∩ B = A → A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_subset_from_intersection_l950_95084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_inequality_l950_95043

noncomputable def x : ℕ → ℝ
  | 0 => 0  -- Adding a case for 0 to cover all natural numbers
  | 1 => Real.sqrt 2
  | n + 1 => Real.sqrt (2 + x n)

theorem x_inequality (n : ℕ) (h : n ≥ 2) : (2 - x n) / (2 - x (n - 1)) > 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_inequality_l950_95043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mark_is_152_l950_95040

/-- Calculates the maximum mark for a paper given the passing percentage, 
    actual score, and the number of marks by which the candidate failed. -/
def calculate_max_mark (passing_percentage : ℚ) (actual_score : ℕ) (failed_by : ℕ) : ℕ :=
  Int.toNat (((actual_score + failed_by : ℚ) / passing_percentage).ceil)

/-- Theorem stating that for a paper with 42% passing mark, where a candidate
    scored 42 marks and failed by 22 marks, the maximum mark is 152. -/
theorem max_mark_is_152 :
  calculate_max_mark (42/100) 42 22 = 152 := by
  sorry

#eval calculate_max_mark (42/100) 42 22

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mark_is_152_l950_95040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l950_95026

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := y^2

/-- The line function -/
def line (y : ℝ) : ℝ := 2*y + 3

/-- The lower bound of the integral -/
def lower_bound : ℝ := -1

/-- The upper bound of the integral -/
def upper_bound : ℝ := 3

/-- The area of the enclosed figure -/
noncomputable def enclosed_area : ℝ := ∫ y in lower_bound..upper_bound, line y - parabola y

theorem enclosed_area_value : enclosed_area = 32/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l950_95026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_grade_sample_size_l950_95049

/-- Represents the ratio of students in each grade --/
structure GradeRatio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the number of students to be sampled from the third grade
    given the total sample size and the grade ratio --/
def sampleSizeThirdGrade (totalSample : ℕ) (ratio : GradeRatio) : ℕ :=
  totalSample * ratio.third / (ratio.first + ratio.second + ratio.third)

/-- Theorem stating that for a given grade ratio and total sample size,
    the number of students sampled from the third grade is 100 --/
theorem third_grade_sample_size 
  (ratio : GradeRatio) 
  (h1 : ratio.first = 2) 
  (h2 : ratio.second = 3) 
  (h3 : ratio.third = 5) 
  (totalSample : ℕ) 
  (h4 : totalSample = 200) :
  sampleSizeThirdGrade totalSample ratio = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_grade_sample_size_l950_95049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l950_95044

/-- Two concentric circles with center D -/
structure ConcentricCircles where
  center : EuclideanSpace ℝ (Fin 2)
  inner_radius : ℝ
  outer_radius : ℝ
  ratio : outer_radius = 4 * inner_radius

/-- The diameter of the larger circle -/
def Diameter (circles : ConcentricCircles) (a c : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖a - c‖ = 2 * circles.outer_radius

/-- A chord of the larger circle tangent to the smaller circle -/
def TangentChord (circles : ConcentricCircles) (b c : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖b - circles.center‖ = circles.outer_radius ∧ 
  ‖c - circles.center‖ = circles.outer_radius ∧
  ‖b - circles.center‖ * ‖c - circles.center‖ = circles.inner_radius * circles.outer_radius

theorem larger_circle_radius 
  (circles : ConcentricCircles) 
  (a b c : EuclideanSpace ℝ (Fin 2)) 
  (h1 : Diameter circles a c)
  (h2 : TangentChord circles b c)
  (h3 : ‖a - b‖ = 8) :
  circles.outer_radius = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l950_95044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_200_equals_20100_l950_95090

def a : ℕ → ℚ
  | 0 => 1  -- Add this case to handle n = 0
  | 1 => 1
  | n + 2 => a (n + 1) + (2 * a (n + 1)) / (n + 1)

theorem a_200_equals_20100 : a 200 = 20100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_200_equals_20100_l950_95090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l950_95065

/-- Given plane vectors a and b, prove properties about vector c and angle between vectors -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (3, 4)) (h2 : b = (-2, 1)) : 
  (∃ c : ℝ × ℝ, (∃ k : ℝ, c = k • (a + 2 • b)) ∧ ‖c‖ = Real.sqrt 37 → 
    c = (-1, 6) ∨ c = (1, -6)) ∧ 
  (∀ l : ℝ, 0 < (a.1 * (a.1 + l * b.1) + a.2 * (a.2 + l * b.2)) ↔ 
    l < 0 ∨ (0 < l ∧ l < 25 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l950_95065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_three_fourths_l950_95020

/-- The sum of the infinite series Σ(3n+2)/(n(n+1)(n+3)) for n from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (3 * n + 2) / (n * (n + 1) * (n + 3))

/-- Theorem stating that the sum of the infinite series equals 3/4 -/
theorem infinite_series_sum_eq_three_fourths : infinite_series_sum = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_three_fourths_l950_95020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_l950_95004

theorem sine_inequality : Real.sin (-5) > Real.sin 3 ∧ Real.sin 3 > Real.sin 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_l950_95004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_play_result_l950_95072

noncomputable def is_valid_choice (a b c : ℤ) : Prop :=
  a ∈ ({1, 2, 3, 4, 5} : Set ℤ) ∧
  b ∈ ({1, 2, 3, 4, 5} : Set ℤ) ∧
  c ∈ ({1, 2, 3, 4, 5} : Set ℤ) ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c

noncomputable def polynomial_min (a b c : ℤ) : ℝ :=
  (c : ℝ) - (b * b : ℝ) / (4 * a)

noncomputable def Ana_strategy (a b c : ℤ) : Prop :=
  ∀ a' b' c' : ℤ, is_valid_choice a' b' c' →
    polynomial_min a b c ≤ polynomial_min a' b' c'

noncomputable def Banana_strategy (a b c : ℤ) : Prop :=
  ∀ a' b' c' : ℤ, is_valid_choice a' b' c' →
    polynomial_min a b c ≥ polynomial_min a' b' c'

theorem optimal_play_result :
  ∃ a b c : ℤ,
    is_valid_choice a b c ∧
    Ana_strategy a b c ∧
    Banana_strategy a b c ∧
    100 * a + 10 * b + c = 451 := by
  sorry

#check optimal_play_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_play_result_l950_95072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_carpet_problem_l950_95009

theorem room_carpet_problem :
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    b > a ∧ 
    (a * b - (a - 4) * (b - 4)) / (a * b) = 1 / 3 ∧
    a > 0 ∧ b > 0
  ) (Finset.range 100 ×ˢ Finset.range 100)).card ∧ n = 3 := by
  sorry

#eval (Finset.filter (fun p : ℕ × ℕ => 
  let (a, b) := p
  b > a ∧ 
  (a * b - (a - 4) * (b - 4)) / (a * b) = 1 / 3 ∧
  a > 0 ∧ b > 0
) (Finset.range 100 ×ˢ Finset.range 100)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_carpet_problem_l950_95009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_nine_fifths_l950_95054

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 3) / (5*x - 9)

-- Theorem statement
theorem vertical_asymptote_at_nine_fifths :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ ∧ δ < ε →
    (∀ (x : ℝ), 0 < |x - 9/5| ∧ |x - 9/5| < δ → |f x| > 1/δ) := by
  sorry

#check vertical_asymptote_at_nine_fifths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_nine_fifths_l950_95054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcl_moles_in_reaction_l950_95069

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between C5H12O and HCl to produce H2O -/
structure Reaction where
  c5h12o : Moles
  hcl : Moles
  h2o : Moles

/-- The reaction is balanced if the number of moles of HCl equals the number of moles of H2O produced -/
def isBalanced (r : Reaction) : Prop := r.hcl = r.h2o

theorem hcl_moles_in_reaction (r : Reaction) 
  (h1 : r.c5h12o = (1 : ℝ))
  (h2 : r.h2o = (18 : ℝ))
  (h3 : isBalanced r) :
  r.hcl = (18 : ℝ) := by
  rw [isBalanced] at h3
  rw [h3, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcl_moles_in_reaction_l950_95069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l950_95094

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h : principal > 0) (t : time > 0) :
  (interest * 100) / (principal * time) = 4.5 ↔
  interest = principal * 4.5 * time / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l950_95094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_sum_l950_95055

theorem recurring_decimal_sum : 
  (123 / 999 : ℚ) + (45 / 9999 : ℚ) + (678 / 999999 : ℚ) = 128178 / 998001000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_sum_l950_95055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_music_beats_per_minute_l950_95041

/-- Calculates the beats per minute of music given daily listening time and weekly beat count. -/
noncomputable def beats_per_minute (daily_hours : ℝ) (weekly_beats : ℕ) : ℝ :=
  let daily_minutes := daily_hours * 60
  let weekly_minutes := daily_minutes * 7
  (weekly_beats : ℝ) / weekly_minutes

/-- Theorem stating that given 2 hours of daily listening and 168000 beats per week, the music has 200 beats per minute. -/
theorem music_beats_per_minute :
  beats_per_minute 2 168000 = 200 := by
  -- Unfold the definition of beats_per_minute
  unfold beats_per_minute
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_music_beats_per_minute_l950_95041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_sharing_ratio_l950_95086

/-- Represents the investment and profit shares of a partner over two years -/
structure PartnerShare where
  initial_investment : ℚ
  first_year_ratio : ℚ
  second_year_investment : ℚ
  second_year_ratio : ℚ

/-- Calculates the total share ratio for a partner over two years -/
def total_share_ratio (partner : PartnerShare) : ℚ :=
  (partner.first_year_ratio * 5 + partner.second_year_ratio * 3) / 45

/-- The main theorem stating the final profit-sharing ratio -/
theorem profit_sharing_ratio 
  (p q r : PartnerShare)
  (h_p_initial : p.initial_investment = 75000)
  (h_q_initial : q.initial_investment = 15000)
  (h_r_initial : r.initial_investment = 45000)
  (h_first_year : p.first_year_ratio + q.first_year_ratio + r.first_year_ratio = 1)
  (h_p_first : p.first_year_ratio = 4/9)
  (h_q_first : q.first_year_ratio = 3/9)
  (h_r_first : r.first_year_ratio = 2/9)
  (h_p_second : p.second_year_investment = p.initial_investment * 3/2)
  (h_q_second : q.second_year_investment = q.initial_investment * 3/4)
  (h_r_second : r.second_year_investment = r.initial_investment)
  (h_second_year : p.second_year_ratio + q.second_year_ratio + r.second_year_ratio = 1)
  (h_p_second_ratio : p.second_year_ratio = 3/15)
  (h_q_second_ratio : q.second_year_ratio = 5/15)
  (h_r_second_ratio : r.second_year_ratio = 7/15) :
  (total_share_ratio p, total_share_ratio q, total_share_ratio r) = (29/45, 30/45, 31/45) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_sharing_ratio_l950_95086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cans_theorem_l950_95008

/-- Represents the number of rooms that can be painted with one can of paint -/
def rooms_per_can (initial_rooms : ℕ) (final_rooms : ℕ) (lost_cans : ℕ) : ℚ :=
  (initial_rooms - final_rooms : ℚ) / lost_cans

/-- Represents the number of cans needed to paint a given number of rooms -/
def cans_needed (rooms : ℕ) (rooms_per_can : ℚ) : ℕ :=
  Int.toNat (Nat.ceil ((rooms : ℚ) / rooms_per_can))

theorem paint_cans_theorem (initial_rooms final_rooms lost_cans : ℕ) 
  (h1 : initial_rooms = 45)
  (h2 : final_rooms = 36)
  (h3 : lost_cans = 4) :
  cans_needed final_rooms (rooms_per_can initial_rooms final_rooms lost_cans) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cans_theorem_l950_95008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_positive_reals_l950_95032

-- Define the function f(x) = (1/2)^(1-x)
noncomputable def f (x : ℝ) : ℝ := (1/2)^(1-x)

-- Theorem statement
theorem f_range_is_positive_reals :
  Set.range f = Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_positive_reals_l950_95032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_S_l950_95012

/-- The set of integer points within or on a circle of radius 10 centered at the origin -/
def S : Set (ℤ × ℤ) := {p | p.1^2 + p.2^2 ≤ 100}

/-- The cardinality of set S is 317 -/
theorem cardinality_of_S : Finset.card (Finset.filter (fun p => p.1^2 + p.2^2 ≤ 100) (Finset.product (Finset.range 21) (Finset.range 21))) = 317 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_S_l950_95012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_test_disease_probability_l950_95064

/-- The probability of having the disease in the population -/
noncomputable def disease_probability : ℝ := 1 / 200

/-- The probability of not having the disease in the population -/
noncomputable def no_disease_probability : ℝ := 1 - disease_probability

/-- The probability of testing positive given that the person has the disease -/
noncomputable def true_positive_rate : ℝ := 1

/-- The probability of testing positive given that the person does not have the disease (false positive rate) -/
noncomputable def false_positive_rate : ℝ := 0.05

/-- The probability of testing positive -/
noncomputable def positive_test_probability : ℝ := 
  disease_probability * true_positive_rate + no_disease_probability * false_positive_rate

theorem positive_test_disease_probability : 
  (disease_probability * true_positive_rate) / positive_test_probability = 20 / 219 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_test_disease_probability_l950_95064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratatouille_yield_l950_95081

/-- Represents the ingredients and their quantities in pounds -/
structure Ingredients :=
  (eggplants : ℚ)
  (zucchini : ℚ)
  (tomatoes : ℚ)
  (onions : ℚ)
  (basil : ℚ)

/-- Represents the prices of ingredients per pound -/
structure Prices :=
  (eggplants : ℚ)
  (zucchini : ℚ)
  (tomatoes : ℚ)
  (onions : ℚ)
  (basil : ℚ)

/-- Calculates the total cost of ingredients -/
def totalCost (i : Ingredients) (p : Prices) : ℚ :=
  i.eggplants * p.eggplants +
  i.zucchini * p.zucchini +
  i.tomatoes * p.tomatoes +
  i.onions * p.onions +
  i.basil * p.basil

/-- Calculates the number of quarts produced -/
def quartsProduced (totalCost costPerQuart : ℚ) : ℚ :=
  totalCost / costPerQuart

theorem ratatouille_yield (i : Ingredients) (p : Prices) (costPerQuart : ℚ) :
  i.eggplants = 5 →
  i.zucchini = 4 →
  i.tomatoes = 4 →
  i.onions = 3 →
  i.basil = 1 →
  p.eggplants = 2 →
  p.zucchini = 2 →
  p.tomatoes = 7/2 →
  p.onions = 1 →
  p.basil = 5 →
  costPerQuart = 10 →
  quartsProduced (totalCost i p) costPerQuart = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratatouille_yield_l950_95081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_x_in_unit_interval_l950_95099

theorem inequality_holds_iff_x_in_unit_interval (x : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → (1 + x)^n ≤ 1 + (2^n - 1) * x) ↔ x ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_x_in_unit_interval_l950_95099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_exams_to_pass_l950_95035

/-- The minimum number of exams Jimmy must write to pass -/
def min_exams : ℕ :=
  let pass_score := 50
  let points_per_exam := 20
  let points_lost := 5
  let additional_points_can_lose := 5
  let total_points_can_lose := points_lost + additional_points_can_lose
  Nat.ceil ((pass_score + total_points_can_lose : ℚ) / points_per_exam)

/-- Theorem stating that Jimmy must write at least 3 exams to pass -/
theorem jimmy_exams_to_pass :
  min_exams = 3 := by
  rfl

#eval min_exams

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_exams_to_pass_l950_95035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_bounds_l950_95085

theorem integers_between_bounds : 
  {n : ℤ | (n : ℚ) > -13/10 ∧ (n : ℚ) < 28/10} = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_bounds_l950_95085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l950_95078

/-- Calculates the speed of a train given its length, the time it takes to pass a person,
    and the person's speed in the opposite direction. -/
noncomputable def train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed_kmph : ℝ) : ℝ :=
  let person_speed_mps := person_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_mps := relative_speed - person_speed_mps
  train_speed_mps * (3600 / 1000)

/-- Theorem stating that a 160m long train passing a person running at 6 kmph
    in the opposite direction in 6 seconds has a speed of 90 kmph. -/
theorem train_speed_problem : train_speed 160 6 6 = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l950_95078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculations_l950_95045

/-- Given an angle α with its vertex at the origin, its initial side coinciding with
    the non-negative half-axis of the x-axis, and its terminal side passing through
    point P(-3, √3), prove the following statements. -/
theorem angle_calculations (α : Real) 
    (h1 : Real.sin α = 1/2) 
    (h2 : Real.cos α = -Real.sqrt 3/2) : 
  (Real.tan (-α) + Real.sin (π/2 + α)) / (Real.cos (π - α) * Real.sin (-π - α)) = -2/3 ∧ 
  Real.tan (2 * α) = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculations_l950_95045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_region_area_is_four_polar_eq1_is_line_x_2_polar_eq2_is_line_y_2_bounded_region_is_square_with_area_four_l950_95018

/-- The area of the region bounded by r = 2sec θ, r = 2csc θ, the x-axis, and the y-axis -/
def bounded_region_area : ℝ := 4

/-- First polar equation bounding the region -/
noncomputable def polar_eq1 (θ : ℝ) : ℝ := 2 / Real.cos θ

/-- Second polar equation bounding the region -/
noncomputable def polar_eq2 (θ : ℝ) : ℝ := 2 / Real.sin θ

theorem bounded_region_area_is_four :
  bounded_region_area = 4 := by rfl

/-- Theorem stating that the first polar equation represents the line x = 2 -/
theorem polar_eq1_is_line_x_2 (θ : ℝ) (h : θ ≠ π/2 ∧ θ ≠ 3*π/2) :
  polar_eq1 θ * Real.cos θ = 2 := by
  sorry

/-- Theorem stating that the second polar equation represents the line y = 2 -/
theorem polar_eq2_is_line_y_2 (θ : ℝ) (h : θ ≠ 0 ∧ θ ≠ π) :
  polar_eq2 θ * Real.sin θ = 2 := by
  sorry

/-- The main theorem proving that the bounded region is a square with area 4 -/
theorem bounded_region_is_square_with_area_four :
  bounded_region_area = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_region_area_is_four_polar_eq1_is_line_x_2_polar_eq2_is_line_y_2_bounded_region_is_square_with_area_four_l950_95018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kingda_ka_minimum_height_l950_95050

/-- The minimum height required to ride Kingda Ka roller coaster -/
noncomputable def minimum_height_kingda_ka : ℚ := 140

/-- Mary's brother's height in centimeters -/
noncomputable def brother_height : ℚ := 180

/-- Mary's height as a fraction of her brother's height -/
noncomputable def mary_height_fraction : ℚ := 2/3

/-- Additional height Mary needs to grow to ride Kingda Ka -/
noncomputable def additional_height_needed : ℚ := 20

/-- Theorem stating the minimum height required to ride Kingda Ka -/
theorem kingda_ka_minimum_height :
  minimum_height_kingda_ka = mary_height_fraction * brother_height + additional_height_needed :=
by
  -- Convert the goal to normal form
  norm_num
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kingda_ka_minimum_height_l950_95050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_miles_to_average_speed_l950_95092

-- Define the initial conditions
def initial_distance : ℚ := 20
def initial_speed : ℚ := 40
def second_speed : ℚ := 70
def desired_average_speed : ℚ := 60

-- Define the function to calculate the additional distance
def additional_distance (x : ℚ) : ℚ :=
  (initial_distance + x) / ((initial_distance / initial_speed) + (x / second_speed))

-- Theorem statement
theorem additional_miles_to_average_speed :
  ∃ x : ℚ, x = 70 ∧ additional_distance x = desired_average_speed := by
  -- The proof goes here
  sorry

#eval additional_distance 70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_miles_to_average_speed_l950_95092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2004_l950_95067

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | n + 2 => (Real.sqrt 3 * sequence_a (n + 1) - 1) / (sequence_a (n + 1) + Real.sqrt 3)

theorem sequence_a_2004 : sequence_a 2004 = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_2004_l950_95067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_distribution_theorem_l950_95027

def distribute_papers (total_papers : ℕ) (num_experts : ℕ) 
  (expert1_papers : ℕ) (expert2_papers : ℕ) (expert3_papers : ℕ) (expert4_papers : ℕ) : ℕ :=
  (Nat.choose total_papers expert1_papers) * 
  (Nat.choose (total_papers - expert1_papers) expert2_papers) * 
  (Nat.choose (total_papers - expert1_papers - expert2_papers) expert3_papers) * 
  (Nat.factorial num_experts)

theorem paper_distribution_theorem : 
  distribute_papers 11 4 4 3 2 2 = 1663200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_distribution_theorem_l950_95027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_powers_of_two_iff_not_div_five_l950_95058

def lastDigit (n : ℕ) : ℕ := n % 10

def sequenceA (a₁ : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => sequenceA a₁ n + lastDigit (sequenceA a₁ n)

def containsInfinitelyManyPowersOfTwo (a₁ : ℕ) : Prop :=
  ∀ k : ℕ, ∃ n : ℕ, ∃ m : ℕ, sequenceA a₁ n = 2^m ∧ m ≥ k

theorem sequence_powers_of_two_iff_not_div_five (a₁ : ℕ) :
  containsInfinitelyManyPowersOfTwo a₁ ↔ ¬(5 ∣ a₁) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_powers_of_two_iff_not_div_five_l950_95058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l950_95028

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x*y + y*z + z*x) * (1/(x+y)^2 + 1/(y+z)^2 + 1/(z+x)^2) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l950_95028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fencing_is_69_l950_95002

/-- Calculates the total perimeter of three rectangular flowerbeds with given dimensions. -/
noncomputable def total_fencing (width1 : ℝ) : ℝ :=
  let length1 := 2 * width1 - 1
  let width2 := width1 - 2
  let length2 := length1 + 3
  let width3 := (width1 + width2) / 2
  let length3 := (length1 + length2) / 2
  2 * (width1 + length1) + 2 * (width2 + length2) + 2 * (width3 + length3)

/-- Theorem stating that the total fencing needed for the three flowerbeds is 69 meters. -/
theorem total_fencing_is_69 : total_fencing 4 = 69 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fencing_is_69_l950_95002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_white_fraction_l950_95013

/-- A pencil with black, white, and blue parts -/
structure Pencil where
  total_length : ℚ
  black_fraction : ℚ
  blue_length : ℚ

/-- The fraction of the remaining part after the black part that is white -/
def white_fraction (p : Pencil) : ℚ :=
  let black_length := p.black_fraction * p.total_length
  let white_length := p.total_length - black_length - p.blue_length
  let remaining_length := p.total_length - black_length
  white_length / remaining_length

/-- Theorem stating that for a pencil with given properties, 
    the fraction of the remaining part after the black part that is white is 1/2 -/
theorem pencil_white_fraction :
  ∀ (p : Pencil), 
    p.total_length = 8 → 
    p.black_fraction = 1/8 → 
    p.blue_length = 7/2 → 
    white_fraction p = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_white_fraction_l950_95013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_day_of_week_l950_95014

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Repr, DecidableEq

/-- Represents the answer given by Friday -/
inductive Answer
  | Yes
  | No
deriving Repr, DecidableEq

/-- Function type for asking a question about a specific day -/
def Question := DayOfWeek → Answer

/-- Represents the strategy for determining the day of the week -/
def Strategy := ℕ → DayOfWeek

/-- Checks if the given day is Friday -/
def isFriday (d : DayOfWeek) : Bool :=
  d == DayOfWeek.Friday

/-- The truthful answer for a given day and question -/
def truthfulAnswer (actual : DayOfWeek) (asked : DayOfWeek) : Answer :=
  if actual == asked then Answer.Yes else Answer.No

/-- The answer given by Friday based on the actual day and the day being asked about -/
def fridayAnswer (actual : DayOfWeek) (asked : DayOfWeek) : Answer :=
  if isFriday actual
  then truthfulAnswer actual asked
  else if truthfulAnswer actual asked == Answer.Yes then Answer.No else Answer.Yes

/-- Theorem stating that it's possible to determine the day of the week within 4 questions -/
theorem determine_day_of_week :
  ∃ (s : Strategy), ∀ (actual : DayOfWeek),
    ∃ (n : ℕ), n ≤ 4 ∧
      (∀ (m : ℕ), m < n →
        fridayAnswer actual (s m) = fridayAnswer actual (s m)) →
      s n = actual := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_day_of_week_l950_95014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_and_line_theorem_l950_95003

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 5 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - y - 9 = 0

-- Define the line ℓ
def line_l (x y : ℝ) : Prop := x = 4 ∨ 21*x + 20*y + 4 = 0

-- Theorem statement
theorem circles_and_line_theorem :
  (∀ x y, C1 x y → C2 x y → (2*x - y + 4 = 0)) ∧ 
  (∃ A B : ℝ × ℝ, 
    (line_l A.1 A.2 ∧ line_l B.1 B.2) ∧
    (C1 A.1 A.2 ∧ C1 B.1 B.2) ∧
    line_l 4 (-4) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 24)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_and_line_theorem_l950_95003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_train1_calculation_l950_95077

/-- The speed of the first train in km/h -/
def speed_train1 : ℝ := 120.016

/-- The length of the first train in meters -/
def length_train1 : ℝ := 260

/-- The speed of the second train in km/h -/
def speed_train2 : ℝ := 80

/-- The time taken for the trains to cross each other in seconds -/
def crossing_time : ℝ := 9

/-- The length of the second train in meters -/
def length_train2 : ℝ := 240.04

/-- Theorem stating that the calculated speed of the first train is correct -/
theorem speed_train1_calculation :
  let total_length : ℝ := length_train1 + length_train2
  let total_length_km : ℝ := total_length / 1000
  let crossing_time_hours : ℝ := crossing_time / 3600
  let relative_speed : ℝ := total_length_km / crossing_time_hours
  ∀ ε > 0, |speed_train1 - (relative_speed - speed_train2)| < ε := by
  sorry

#check speed_train1_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_train1_calculation_l950_95077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_two_beta_l950_95061

theorem sin_alpha_plus_two_beta
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = -5 / 13)
  (h4 : Real.sin β = 3 / 5) :
  Real.sin (α + 2 * β) = 33 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_two_beta_l950_95061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l950_95031

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def timeToCrossBridge (trainLength : ℝ) (trainSpeed : ℝ) (totalLength : ℝ) : ℝ :=
  let speedInMeterPerSecond := trainSpeed * 1000 / 3600
  let distanceToCover := totalLength
  distanceToCover / speedInMeterPerSecond

/-- Theorem: The time taken for the train to cross the bridge is 15.6 seconds -/
theorem train_bridge_crossing_time :
  let trainLength : ℝ := 180
  let trainSpeed : ℝ := 45
  let totalLength : ℝ := 195
  timeToCrossBridge trainLength trainSpeed totalLength = 15.6 := by
  -- Unfold the definition of timeToCrossBridge
  unfold timeToCrossBridge
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l950_95031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_inscribed_squares_l950_95039

-- Define the necessary structures
structure IsoscelesRightTriangle where
  side : ℝ
  side_positive : side > 0

structure Square where
  side : ℝ
  side_positive : side > 0

-- Define the necessary functions
def Square.area (s : Square) : ℝ := s.side ^ 2

def Square.inscribed_at_right_angle (s : Square) (t : IsoscelesRightTriangle) : Prop :=
  s.side = t.side / (1 + Real.sqrt 2)

def Square.inscribed_between_hypotenuse_and_legs (s : Square) (t : IsoscelesRightTriangle) : Prop :=
  s.side = t.side / Real.sqrt 2

theorem isosceles_right_triangle_inscribed_squares 
  (triangle : IsoscelesRightTriangle) 
  (square1 : Square) 
  (square2 : Square) 
  (h1 : Square.inscribed_at_right_angle square1 triangle) 
  (h2 : Square.inscribed_between_hypotenuse_and_legs square2 triangle) 
  (h3 : Square.area square1 = 784) : 
  Square.area square2 = 784 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_inscribed_squares_l950_95039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l950_95001

-- Define the ⊕ operation
noncomputable def circle_plus (a b : ℝ) : ℝ :=
  if a > b then a else b

-- Axioms for the ⊕ operation
axiom circle_plus_comm (a b : ℝ) : circle_plus a b = circle_plus b a

-- Theorem to prove
theorem range_of_x (x : ℝ) :
  (circle_plus (2 * x + 1) (x + 3) = x + 3) ↔ x < 2 := by
  sorry

#check range_of_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l950_95001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_digit_perfect_square_with_conditions_l950_95059

/-- A function that checks if a number is a perfect square --/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

/-- A function that returns the first two digits of a six-digit number --/
def firstTwoDigits (n : Nat) : Nat :=
  n / 10000

/-- A function that returns the middle two digits of a six-digit number --/
def middleTwoDigits (n : Nat) : Nat :=
  (n / 100) % 100

/-- A function that returns the last two digits of a six-digit number --/
def lastTwoDigits (n : Nat) : Nat :=
  n % 100

/-- A function that checks if all digits in a six-digit number are distinct --/
def allDigitsDistinct (n : Nat) : Prop :=
  let digits := [n / 100000, (n / 10000) % 10, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  List.Pairwise (· ≠ ·) digits

theorem no_six_digit_perfect_square_with_conditions :
  ¬∃ n : Nat,
    100000 ≤ n ∧ n < 1000000 ∧
    isPerfectSquare n ∧
    (∀ d, d ∈ [n / 100000, (n / 10000) % 10, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10] → d ≠ 0) ∧
    allDigitsDistinct n ∧
    isPerfectSquare (firstTwoDigits n) ∧
    isPerfectSquare (middleTwoDigits n) ∧
    isPerfectSquare (lastTwoDigits n) ∧
    firstTwoDigits n ≠ middleTwoDigits n ∧
    firstTwoDigits n ≠ lastTwoDigits n ∧
    middleTwoDigits n ≠ lastTwoDigits n :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_digit_perfect_square_with_conditions_l950_95059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_incenter_excenter_l950_95000

/-- The center of the inscribed circle of a triangle -/
noncomputable def circle_center (P Q R : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
sorry

/-- The center of the excircle opposite to vertex Q of a triangle PQR -/
noncomputable def excircle_center (P Q R : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
sorry

/-- Given a triangle PQR with side lengths PQ = 17, PR = 19, and QR = 20,
    the distance between the incenter and the excenter opposite to vertex Q
    is √1360 - √157. -/
theorem distance_incenter_excenter (P Q R : EuclideanSpace ℝ (Fin 2)) : 
  let d_PQ := ‖Q - P‖
  let d_PR := ‖R - P‖
  let d_QR := ‖R - Q‖
  let s := (d_PQ + d_PR + d_QR) / 2
  let area := Real.sqrt (s * (s - d_PQ) * (s - d_PR) * (s - d_QR))
  let incenter := circle_center P Q R
  let excenter := excircle_center P Q R
  d_PQ = 17 ∧ d_PR = 19 ∧ d_QR = 20 →
  ‖excenter - incenter‖ = Real.sqrt 1360 - Real.sqrt 157 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_incenter_excenter_l950_95000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_lambda_bound_l950_95091

theorem sequence_increasing_lambda_bound (a : ℕ+ → ℝ) (lambda : ℝ) :
  (∀ n : ℕ+, a n = n.val ^ 2 + lambda * n.val) →
  (∀ n : ℕ+, a n < a (n + 1)) →
  lambda > -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_lambda_bound_l950_95091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_l950_95062

theorem pizza_slices (total_pizzas : ℕ) (stephen_ate_percent : ℚ) (pete_ate_percent : ℚ) (slices_left : ℕ) : ℕ :=
  let total_slices := total_pizzas * slices_per_pizza
  let stephen_ate := stephen_ate_percent * total_slices
  let remaining_after_stephen := total_slices - stephen_ate
  let pete_ate := pete_ate_percent * remaining_after_stephen
  let final_remaining := remaining_after_stephen - pete_ate
  have h1 : total_pizzas = 2 := by sorry
  have h2 : stephen_ate_percent = 1/4 := by sorry
  have h3 : pete_ate_percent = 1/2 := by sorry
  have h4 : slices_left = 9 := by sorry
  have h5 : final_remaining = slices_left := by sorry
  slices_per_pizza
where
  slices_per_pizza : ℕ := 12

#check pizza_slices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_slices_l950_95062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l950_95038

-- Define the quadrilateral
def Quadrilateral := {p : ℝ × ℝ | p.1 + 2 * p.2 ≤ 6 ∧ p.1 + p.2 ≥ 3 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_side_length : 
  ∃ (p1 p2 : Quadrilateral), ∀ (q1 q2 : Quadrilateral), 
    distance p1.val p2.val = 3 * Real.sqrt 2 ∧ 
    distance q1.val q2.val ≤ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l950_95038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_l950_95029

-- Define the circle (O)
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the ellipse (E)
def myEllipse (x y : ℝ) : Prop := y^2/3 + x^2/2 = 1

-- Define a point P on the circle
def pointOnCircle (x₀ y₀ : ℝ) : Prop := myCircle x₀ y₀

-- Define the product of slopes of tangent lines
noncomputable def productOfSlopes (x₀ y₀ : ℝ) : ℝ := -(y₀^2 - 3)/(2 - x₀^2)

-- Theorem statement
theorem tangent_slopes_product (x₀ y₀ : ℝ) :
  pointOnCircle x₀ y₀ → productOfSlopes x₀ y₀ = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_l950_95029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_range_proof_l950_95070

noncomputable def sequence_a : ℕ → ℝ := sorry

noncomputable def sum_s (n : ℕ) : ℝ := (1/2 : ℝ) * n^2 + (1/2 : ℝ) * n

noncomputable def sequence_T : ℕ → ℝ := sorry

theorem sequence_and_range_proof (a : ℝ) :
  (∀ n : ℕ, sum_s n = (1/2 : ℝ) * n^2 + (1/2 : ℝ) * n) →
  (∀ n : ℕ, sequence_a (n + 1) = sum_s (n + 1) - sum_s n) →
  (∀ n : ℕ, sequence_T n > (1/3 : ℝ) * Real.log (1 - a) / Real.log a) →
  (∀ n : ℕ, sequence_a n = n) ∧ (0 < a ∧ a < 1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_range_proof_l950_95070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l950_95022

theorem intersection_range (l : ℝ) : 
  (l < 0) →
  (∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    (2 * |p₁.1| - p₁.2 - 4 = 0) ∧ (p₁.1^2 + l * p₁.2^2 = 4) ∧
    (2 * |p₂.1| - p₂.2 - 4 = 0) ∧ (p₂.1^2 + l * p₂.2^2 = 4)) →
  -1/4 < l ∧ l < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l950_95022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_distribution_l950_95006

def total_prize : ℝ := 40000

def prize_a (p₁ p₂ : ℝ) : ℝ := 10000 * p₁ + 20000 * p₂
def prize_b (p₁ p₂ : ℝ) : ℝ := 10000 * p₁ + 20000 * p₂
def prize_c (p₁ p₂ : ℝ) : ℝ := total_prize - prize_a p₁ p₂ - prize_b p₁ p₂

-- We need to define a probability measure for this problem
noncomputable def P (event : Prop) : ℝ := sorry

theorem prize_distribution (p₁ p₂ : ℝ) 
  (h₁ : p₁ + p₂ = 1) 
  (h₂ : p₁ ≥ 0) 
  (h₃ : p₂ ≥ 0) :
  (p₁ = 1/2 ∧ p₂ = 1/2 → P (prize_c p₁ p₂ = 10000) = 1/2) ∧
  (prize_a p₁ p₂ = prize_b p₁ p₂ ∧ prize_b p₁ p₂ = prize_c p₁ p₂ → p₁ = 2/3 ∧ p₂ = 1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_distribution_l950_95006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_progress_l950_95025

/-- Calculate the team's overall progress in meters given specific play gains and losses -/
theorem football_team_progress (x y z w v : ℝ) (h1 : x = -15) (h2 : y = 20) (h3 : z = 10) 
  (h4 : w = 25) (h5 : v = 5) : 
  (x + y - z + w + (0.5 * y) - v) * 0.9144 = 22.86 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_progress_l950_95025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l950_95071

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 2 * x - 7) / (4 * x^2 + 3 * x + 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x > N, |f x - 3/2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l950_95071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_lines_perpendicularity_l950_95047

structure Plane where
  -- Define a plane (placeholder)
  dummy : Unit

structure Line where
  -- Define a line (placeholder)
  dummy : Unit

def intersects (α β : Plane) (m : Line) : Prop :=
  -- Define intersection of two planes along a line (placeholder)
  True

def in_plane (l : Line) (p : Plane) : Prop :=
  -- Define a line being in a plane (placeholder)
  True

def perpendicular (l₁ l₂ : Line) : Prop :=
  -- Define perpendicularity of two lines (placeholder)
  True

def planes_perpendicular (α β : Plane) : Prop :=
  -- Define perpendicularity of two planes (placeholder)
  True

theorem planes_lines_perpendicularity 
  (α β : Plane) (m a b : Line) 
  (h1 : intersects α β m)
  (h2 : in_plane a α)
  (h3 : in_plane b β)
  (h4 : perpendicular b m) :
  (planes_perpendicular α β → perpendicular a b) ∧
  ¬(perpendicular a b → planes_perpendicular α β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_lines_perpendicularity_l950_95047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slide_step_difference_l950_95057

/-- The number of steps Frank takes between consecutive beacons -/
def frank_steps : ℕ := 60

/-- The number of slides Peter takes between consecutive beacons -/
def peter_slides : ℕ := 15

/-- The number of beacons -/
def num_beacons : ℕ := 31

/-- The distance in feet between the first and the 31st beacon -/
def total_distance : ℚ := 2640

/-- The difference between Peter's slide length and Frank's step length -/
noncomputable def length_difference : ℚ := 
  (total_distance / ((num_beacons - 1) * peter_slides)) - 
  (total_distance / ((num_beacons - 1) * frank_steps))

theorem slide_step_difference : length_difference = 22/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slide_step_difference_l950_95057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_690_degrees_l950_95095

open Real

theorem sin_690_degrees (h1 : ∀ x, sin (x + 2*π) = sin x)
                        (h2 : ∀ x, sin (-x) = -sin x)
                        (h3 : sin (π/6) = 1/2) :
  sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_690_degrees_l950_95095
