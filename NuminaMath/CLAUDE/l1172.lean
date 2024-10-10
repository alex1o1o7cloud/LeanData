import Mathlib

namespace sportswear_price_reduction_l1172_117270

/-- Given two equal percentage reductions that reduce a price from 560 to 315,
    prove that the equation 560(1-x)^2 = 315 holds true, where x is the decimal
    form of the percentage reduction. -/
theorem sportswear_price_reduction (x : ℝ) : 
  (∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 560 * (1 - x)^2 = 315) :=
by sorry

end sportswear_price_reduction_l1172_117270


namespace quadratic_perfect_square_constant_l1172_117282

/-- A quadratic expression can be expressed as the square of a binomial if and only if its discriminant is zero. -/
def is_perfect_square (a b c : ℝ) : Prop :=
  b ^ 2 - 4 * a * c = 0

theorem quadratic_perfect_square_constant (b : ℝ) :
  is_perfect_square 9 (-24) b → b = 16 := by
  sorry

end quadratic_perfect_square_constant_l1172_117282


namespace cow_count_is_ten_l1172_117247

/-- Represents the number of animals in a group -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- Theorem stating that the number of cows is 10 -/
theorem cow_count_is_ten (group : AnimalGroup) 
    (h : totalLegs group = 2 * totalHeads group + 20) : 
    group.cows = 10 := by
  sorry

#check cow_count_is_ten

end cow_count_is_ten_l1172_117247


namespace pseudo_periodic_minus_one_is_periodic_two_cos_pseudo_periodic_iff_omega_multiple_of_pi_l1172_117233

-- Define a pseudo-periodic function
def IsPseudoPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x, f (x + T) = T * f x

-- Theorem 1
theorem pseudo_periodic_minus_one_is_periodic_two (f : ℝ → ℝ) 
  (h : IsPseudoPeriodic f (-1)) : 
  ∀ x, f (x + 2) = f x := by sorry

-- Theorem 2
theorem cos_pseudo_periodic_iff_omega_multiple_of_pi (ω : ℝ) :
  IsPseudoPeriodic (λ x => Real.cos (ω * x)) T ↔ ∃ k : ℤ, ω = k * Real.pi := by sorry

end pseudo_periodic_minus_one_is_periodic_two_cos_pseudo_periodic_iff_omega_multiple_of_pi_l1172_117233


namespace third_row_chairs_l1172_117234

def chair_sequence (n : ℕ) : ℕ → ℕ
  | 1 => 14
  | 2 => 23
  | 3 => n
  | 4 => 41
  | 5 => 50
  | 6 => 59
  | _ => 0

theorem third_row_chairs :
  ∃ n : ℕ, 
    chair_sequence n 2 - chair_sequence n 1 = 9 ∧
    chair_sequence n 4 - chair_sequence n 2 = 18 ∧
    chair_sequence n 5 - chair_sequence n 4 = 9 ∧
    chair_sequence n 6 - chair_sequence n 5 = 9 ∧
    n = 32 := by
  sorry

end third_row_chairs_l1172_117234


namespace tangent_product_equals_two_l1172_117252

theorem tangent_product_equals_two (α β : Real) (h : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end tangent_product_equals_two_l1172_117252


namespace abs_sum_problem_l1172_117224

theorem abs_sum_problem (x y : ℝ) 
  (h1 : |x| + x + y = 8) 
  (h2 : x + |y| - y = 14) : 
  x + y = 2 := by sorry

end abs_sum_problem_l1172_117224


namespace trig_sum_equals_two_l1172_117285

theorem trig_sum_equals_two :
  Real.cos (0 : ℝ) ^ 4 +
  Real.cos (Real.pi / 2) ^ 4 +
  Real.sin (Real.pi / 4) ^ 4 +
  Real.sin (3 * Real.pi / 4) ^ 4 = 2 := by
  sorry

end trig_sum_equals_two_l1172_117285


namespace min_values_theorem_l1172_117206

theorem min_values_theorem (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h : (r + s - r * s) * (r + s + r * s) = r * s) : 
  (∃ (r' s' : ℝ), r' > 0 ∧ s' > 0 ∧ 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' ∧
    r + s - r * s ≥ -3 + 2 * Real.sqrt 3 ∧
    r + s + r * s ≥ 3 + 2 * Real.sqrt 3) ∧
  (r + s - r * s = -3 + 2 * Real.sqrt 3 ∨ r + s + r * s = 3 + 2 * Real.sqrt 3 → 
    r = Real.sqrt 3 ∧ s = Real.sqrt 3) :=
by sorry

end min_values_theorem_l1172_117206


namespace projection_onto_plane_l1172_117204

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ :=
  sorry

theorem projection_onto_plane (P : Plane) :
  project (2, 4, 7) P = (1, 3, 3) →
  project (6, -3, 8) P = (41/9, -40/9, 20/9) := by
  sorry

end projection_onto_plane_l1172_117204


namespace danny_wrappers_l1172_117280

/-- Represents Danny's collection of bottle caps and wrappers -/
structure Collection where
  initial_caps : ℕ
  found_caps : ℕ
  found_wrappers : ℕ
  total_caps : ℕ
  initial_wrappers : ℕ

/-- The theorem states that the number of wrappers Danny has now
    is equal to his initial number of wrappers plus the number of wrappers found -/
theorem danny_wrappers (c : Collection)
  (h1 : c.initial_caps = 6)
  (h2 : c.found_caps = 22)
  (h3 : c.found_wrappers = 8)
  (h4 : c.total_caps = 28)
  (h5 : c.total_caps = c.initial_caps + c.found_caps) :
  c.initial_wrappers + c.found_wrappers = c.initial_wrappers + 8 := by
  sorry


end danny_wrappers_l1172_117280


namespace qr_length_l1172_117264

/-- Given points P, Q, R on a line segment where Q is between P and R -/
structure LineSegment where
  P : ℝ
  Q : ℝ
  R : ℝ
  Q_between : P ≤ Q ∧ Q ≤ R

/-- The length of a line segment -/
def length (a b : ℝ) : ℝ := |b - a|

theorem qr_length (seg : LineSegment) 
  (h1 : length seg.P seg.R = 12)
  (h2 : length seg.P seg.Q = 3) : 
  length seg.Q seg.R = 9 := by
sorry

end qr_length_l1172_117264


namespace inscribed_square_area_l1172_117268

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A square inscribed in the region bounded by the parabola and x-axis -/
structure InscribedSquare where
  center : ℝ
  sideLength : ℝ
  top_touches_parabola : parabola (center + sideLength/2) = sideLength
  bottom_on_x_axis : center - sideLength/2 ≥ 0

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.sideLength^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end inscribed_square_area_l1172_117268


namespace percentage_relation_l1172_117220

theorem percentage_relation (A B C x y : ℝ) : 
  A > C ∧ C > B ∧ B > 0 →
  C = B * (1 + y / 100) →
  A = C * (1 + x / 100) →
  x = 100 * ((100 * (A - B)) / (100 + y)) :=
by sorry

end percentage_relation_l1172_117220


namespace ellipse_eccentricity_range_l1172_117210

/-- The eccentricity of an ellipse satisfies the given range -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let B := (0, b)
  let e := Real.sqrt (1 - (b^2 / a^2))
  (∀ p ∈ C, Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2) ≤ 2*b) →
  0 < e ∧ e ≤ Real.sqrt 2 / 2 := by
  sorry


end ellipse_eccentricity_range_l1172_117210


namespace cos_2alpha_plus_pi_3_l1172_117241

theorem cos_2alpha_plus_pi_3 (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin (α - π / 12) = 3 / 5) :
  Real.cos (2 * α + π / 3) = -24 / 25 := by
sorry

end cos_2alpha_plus_pi_3_l1172_117241


namespace range_of_a_l1172_117289

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) :=
sorry

end range_of_a_l1172_117289


namespace price_decrease_percentage_l1172_117258

-- Define the original and discounted prices
def original_price : ℚ := 12 / 3
def discounted_price : ℚ := 10 / 4

-- Define the percentage decrease
def percentage_decrease : ℚ := (original_price - discounted_price) / original_price * 100

-- Theorem statement
theorem price_decrease_percentage :
  percentage_decrease = 37.5 := by
  sorry

end price_decrease_percentage_l1172_117258


namespace ad_broadcast_solution_l1172_117269

/-- Represents the number of ads remaining after the k-th broadcasting -/
def remaining_ads (m : ℕ) (k : ℕ) : ℚ :=
  if k = 0 then m
  else (7/8 : ℚ) * remaining_ads m (k-1) - (7/8 : ℚ) * k

/-- The total number of ads broadcast up to and including the k-th insert -/
def ads_broadcast (m : ℕ) (k : ℕ) : ℚ :=
  m - remaining_ads m k

theorem ad_broadcast_solution (n : ℕ) (m : ℕ) (h1 : n > 1) 
  (h2 : ads_broadcast m n = m) 
  (h3 : ∀ k < n, ads_broadcast m k < m) :
  n = 7 ∧ m = 49 := by
  sorry


end ad_broadcast_solution_l1172_117269


namespace average_marks_combined_classes_l1172_117223

theorem average_marks_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 24) (h₂ : n₂ = 50) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 53.51 := by
  sorry

end average_marks_combined_classes_l1172_117223


namespace element_in_set_l1172_117212

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end element_in_set_l1172_117212


namespace sum_of_intersection_coordinates_l1172_117281

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 1)^2
def parabola2 (x y : ℝ) : Prop := x + 4 = (y - 3)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem sum_of_intersection_coordinates :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 8) :=
sorry

end sum_of_intersection_coordinates_l1172_117281


namespace first_grade_sample_size_l1172_117255

/-- Represents the ratio of students in each grade -/
structure GradeRatio :=
  (first second third fourth : ℕ)

/-- Calculates the number of students to be sampled from the first grade
    given the total sample size and the grade ratio -/
def sampleFirstGrade (totalSample : ℕ) (ratio : GradeRatio) : ℕ :=
  (totalSample * ratio.first) / (ratio.first + ratio.second + ratio.third + ratio.fourth)

/-- Theorem stating that for a sample size of 300 and a grade ratio of 4:5:5:6,
    the number of students to be sampled from the first grade is 60 -/
theorem first_grade_sample_size :
  let totalSample : ℕ := 300
  let ratio : GradeRatio := { first := 4, second := 5, third := 5, fourth := 6 }
  sampleFirstGrade totalSample ratio = 60 := by
  sorry


end first_grade_sample_size_l1172_117255


namespace correct_probability_l1172_117242

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
  | Clubs
  | Diamonds
  | Hearts
  | Spades

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten
  | Jack | Queen | King

/-- A card is a face card if it's a Jack, Queen, or King -/
def isFaceCard (r : Rank) : Bool :=
  match r with
  | Rank.Jack | Rank.Queen | Rank.King => true
  | _ => false

/-- The probability of drawing a club as the first card and a face card diamond as the second card -/
def consecutiveDrawProbability (d : Deck) : Rat :=
  (13 : Rat) / 884

theorem correct_probability (d : Deck) :
  consecutiveDrawProbability d = 13 / 884 := by
  sorry

end correct_probability_l1172_117242


namespace max_rooks_is_400_l1172_117250

/-- Represents a rectangular hole on a chessboard -/
structure Hole :=
  (x : Nat) (y : Nat) (width : Nat) (height : Nat)

/-- Represents a 300x300 chessboard with a hole -/
structure Board :=
  (hole : Hole)
  (is_valid : hole.x + hole.width < 300 ∧ hole.y + hole.height < 300)

/-- The maximum number of non-attacking rooks on a 300x300 board with a hole -/
def max_rooks (b : Board) : Nat :=
  sorry

/-- Theorem: The maximum number of non-attacking rooks is 400 for any valid hole -/
theorem max_rooks_is_400 (b : Board) : max_rooks b = 400 :=
  sorry

end max_rooks_is_400_l1172_117250


namespace quadratic_root_transformation_l1172_117248

theorem quadratic_root_transformation (k ℓ : ℝ) (r₁ r₂ : ℝ) : 
  (r₁^2 + k*r₁ + ℓ = 0) → 
  (r₂^2 + k*r₂ + ℓ = 0) → 
  ∃ v : ℝ, r₁^2^2 + (-k^2 + 2*ℓ)*r₁^2 + v = 0 ∧ r₂^2^2 + (-k^2 + 2*ℓ)*r₂^2 + v = 0 :=
by sorry

end quadratic_root_transformation_l1172_117248


namespace decimal_multiplication_and_composition_l1172_117253

theorem decimal_multiplication_and_composition : 
  (35 * 0.01 = 0.35) ∧ (0.875 = 875 * 0.001) := by
  sorry

end decimal_multiplication_and_composition_l1172_117253


namespace tan_theta_value_l1172_117291

open Complex

theorem tan_theta_value (θ : ℝ) :
  (↑(1 : ℂ) + I) * sin θ - (↑(1 : ℂ) + I * cos θ) ∈ {z : ℂ | z.re + z.im + 1 = 0} →
  tan θ = 1/2 := by
  sorry

end tan_theta_value_l1172_117291


namespace only_statement4_correct_l1172_117227

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point3D) : Point3D := ⟨p.x, -p.y, -p.z⟩
def symmetryYOZPlane (p : Point3D) : Point3D := ⟨-p.x, p.y, p.z⟩
def symmetryYAxis (p : Point3D) : Point3D := ⟨-p.x, p.y, -p.z⟩
def symmetryOrigin (p : Point3D) : Point3D := ⟨-p.x, -p.y, -p.z⟩

-- Define the statements
def statement1 (p : Point3D) : Prop := symmetryXAxis p = ⟨p.x, -p.y, p.z⟩
def statement2 (p : Point3D) : Prop := symmetryYOZPlane p = ⟨p.x, -p.y, -p.z⟩
def statement3 (p : Point3D) : Prop := symmetryYAxis p = ⟨-p.x, p.y, p.z⟩
def statement4 (p : Point3D) : Prop := symmetryOrigin p = ⟨-p.x, -p.y, -p.z⟩

-- Theorem to prove
theorem only_statement4_correct (p : Point3D) :
  ¬(statement1 p) ∧ ¬(statement2 p) ∧ ¬(statement3 p) ∧ (statement4 p) :=
sorry

end only_statement4_correct_l1172_117227


namespace factorial_simplification_l1172_117262

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 1320 := by
  sorry

end factorial_simplification_l1172_117262


namespace dwarf_ice_cream_problem_l1172_117293

theorem dwarf_ice_cream_problem :
  ∀ (n : ℕ) (vanilla chocolate fruit : ℕ),
    n = 10 →
    vanilla = n →
    chocolate = n / 2 →
    fruit = 1 →
    ∃ (truthful : ℕ),
      truthful = 4 ∧
      truthful + (n - truthful) = n ∧
      truthful + 2 * (n - truthful) = vanilla + chocolate + fruit :=
by sorry

end dwarf_ice_cream_problem_l1172_117293


namespace sin_cos_properties_l1172_117228

open Real

theorem sin_cos_properties : ¬(
  (∃ (T : ℝ), T > 0 ∧ T = π/2 ∧ ∀ (x : ℝ), sin (2*x) = sin (2*(x + T))) ∧
  (∀ (x : ℝ), cos x = cos (π - x))
) := by sorry

end sin_cos_properties_l1172_117228


namespace desired_average_l1172_117290

theorem desired_average (numbers : List ℕ) (h1 : numbers = [6, 16, 8, 22]) : 
  (numbers.sum / numbers.length : ℚ) = 13 := by
  sorry

end desired_average_l1172_117290


namespace pyramid_base_side_length_l1172_117211

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  pyramidHeight : ℝ
  /-- Radius of the hemisphere -/
  hemisphereRadius : ℝ
  /-- The hemisphere is tangent to all four faces and the base of the pyramid -/
  isTangent : Bool

/-- Calculate the side length of the square base of the pyramid -/
def calculateBaseSideLength (p : PyramidWithHemisphere) : ℝ :=
  sorry

/-- Theorem stating that for a pyramid of height 9 and hemisphere of radius 3,
    the side length of the base is 9 -/
theorem pyramid_base_side_length 
  (p : PyramidWithHemisphere) 
  (h1 : p.pyramidHeight = 9) 
  (h2 : p.hemisphereRadius = 3) 
  (h3 : p.isTangent = true) : 
  calculateBaseSideLength p = 9 := by
  sorry

end pyramid_base_side_length_l1172_117211


namespace perpendicular_vectors_m_value_l1172_117205

/-- Given two vectors a and b in R², prove that if they are perpendicular
    and a = (1, -1) and b = (m+1, 2m-4), then m = 5. -/
theorem perpendicular_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![m+1, 2*m-4]
  (∀ i, a i * b i = 0) → m = 5 := by
sorry

end perpendicular_vectors_m_value_l1172_117205


namespace ellipse_foci_distance_l1172_117273

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_distance (P : ℝ × ℝ) (h1 : is_on_ellipse P.1 P.2) 
  (h2 : distance P F1 = 2) : distance P F2 = 4 := by
  sorry

end ellipse_foci_distance_l1172_117273


namespace can_collection_increase_l1172_117288

/-- Proves that the daily increase in can collection is 5 cans --/
theorem can_collection_increase (initial_cans : ℕ) (days : ℕ) (total_cans : ℕ) 
  (h1 : initial_cans = 20)
  (h2 : days = 5)
  (h3 : total_cans = 150)
  (h4 : ∃ x : ℕ, total_cans = initial_cans * days + (days * (days - 1) / 2) * x) :
  ∃ x : ℕ, x = 5 ∧ total_cans = initial_cans * days + (days * (days - 1) / 2) * x := by
  sorry

end can_collection_increase_l1172_117288


namespace negation_of_universal_proposition_l1172_117217

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + 2*x ≥ 0) ↔
  (∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + 2*x < 0) :=
by sorry

end negation_of_universal_proposition_l1172_117217


namespace boat_speed_specific_boat_speed_l1172_117261

/-- The speed of a boat in still water given its travel times with and against a current. -/
theorem boat_speed (distance : ℝ) (time_against : ℝ) (time_with : ℝ) :
  distance > 0 ∧ time_against > 0 ∧ time_with > 0 →
  ∃ (boat_speed current_speed : ℝ),
    (boat_speed - current_speed) * time_against = distance ∧
    (boat_speed + current_speed) * time_with = distance ∧
    boat_speed = 15.6 := by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_boat_speed :
  ∃ (boat_speed current_speed : ℝ),
    (boat_speed - current_speed) * 8 = 96 ∧
    (boat_speed + current_speed) * 5 = 96 ∧
    boat_speed = 15.6 := by
  sorry

end boat_speed_specific_boat_speed_l1172_117261


namespace x_neq_zero_necessary_not_sufficient_l1172_117237

theorem x_neq_zero_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  ¬(∀ x : ℝ, x ≠ 0 → x > 0) :=
by sorry

end x_neq_zero_necessary_not_sufficient_l1172_117237


namespace fourth_angle_measure_l1172_117256

-- Define a quadrilateral type
structure Quadrilateral :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (angle4 : ℝ)

-- Define the property that the sum of angles in a quadrilateral is 360°
def sum_of_angles (q : Quadrilateral) : Prop :=
  q.angle1 + q.angle2 + q.angle3 + q.angle4 = 360

-- Theorem statement
theorem fourth_angle_measure (q : Quadrilateral) 
  (h1 : q.angle1 = 120)
  (h2 : q.angle2 = 85)
  (h3 : q.angle3 = 90)
  (h4 : sum_of_angles q) :
  q.angle4 = 65 := by
  sorry

end fourth_angle_measure_l1172_117256


namespace arithmetic_number_difference_l1172_117278

/-- A 4-digit number is arithmetic if its digits are distinct and form an arithmetic sequence. -/
def is_arithmetic (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a d : ℤ), 
    let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
    digits.map Int.ofNat = [a, a + d, a + 2*d, a + 3*d] ∧
    digits.toFinset.card = 4

/-- The largest arithmetic 4-digit number -/
def largest_arithmetic : ℕ := 9876

/-- The smallest arithmetic 4-digit number -/
def smallest_arithmetic : ℕ := 1234

theorem arithmetic_number_difference :
  is_arithmetic largest_arithmetic ∧
  is_arithmetic smallest_arithmetic ∧
  largest_arithmetic - smallest_arithmetic = 8642 :=
sorry

end arithmetic_number_difference_l1172_117278


namespace zoo_tickets_problem_l1172_117272

/-- Proves that for a family of seven people buying zoo tickets, where adult tickets 
    cost $21 and children's tickets cost $14, if the total cost is $119, 
    then the number of adult tickets purchased is 3. -/
theorem zoo_tickets_problem (adult_cost children_cost total_cost : ℕ) 
  (family_size : ℕ) (num_adults : ℕ) :
  adult_cost = 21 →
  children_cost = 14 →
  total_cost = 119 →
  family_size = 7 →
  num_adults + (family_size - num_adults) = family_size →
  num_adults * adult_cost + (family_size - num_adults) * children_cost = total_cost →
  num_adults = 3 := by
  sorry

end zoo_tickets_problem_l1172_117272


namespace min_value_f_and_m_plus_2n_l1172_117202

-- Define the function f
def f (x a : ℝ) : ℝ := x + |x - a|

-- State the theorem
theorem min_value_f_and_m_plus_2n :
  ∃ (a : ℝ),
    (∀ x, (f x a - 2)^4 ≥ 0 ∧ f x a ≤ 4) →
    (∃ x₀, ∀ x, f x a ≥ f x₀ a ∧ f x₀ a = 2) ∧
    (∀ m n : ℕ+, 1 / (m : ℝ) + 2 / (n : ℝ) = 2 →
      (m : ℝ) + 2 * (n : ℝ) ≥ 9/2) ∧
    (∃ m₀ n₀ : ℕ+, 1 / (m₀ : ℝ) + 2 / (n₀ : ℝ) = 2 ∧
      (m₀ : ℝ) + 2 * (n₀ : ℝ) = 9/2) :=
by sorry

end min_value_f_and_m_plus_2n_l1172_117202


namespace minor_axis_length_l1172_117235

def ellipse_equation (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

theorem minor_axis_length :
  ∃ (minor_axis_length : ℝ),
    minor_axis_length = 4 ∧
    ∀ (x y : ℝ), ellipse_equation x y →
      ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
        x^2 / a^2 + y^2 / b^2 = 1 ∧
        minor_axis_length = 2 * b :=
by sorry

end minor_axis_length_l1172_117235


namespace polar_bear_daily_fish_consumption_l1172_117286

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish_amount : ℝ := trout_amount + salmon_amount

theorem polar_bear_daily_fish_consumption :
  total_fish_amount = 0.6 := by sorry

end polar_bear_daily_fish_consumption_l1172_117286


namespace shopkeeper_pricing_l1172_117266

theorem shopkeeper_pricing (CP : ℝ) 
  (h1 : 0.75 * CP = 600) : 1.25 * CP = 1000 := by
  sorry

end shopkeeper_pricing_l1172_117266


namespace inverse_variation_problem_l1172_117249

theorem inverse_variation_problem (p q : ℝ) (k : ℝ) (h1 : k > 0) :
  (∀ x y, x * y = k → x > 0 → y > 0) →  -- inverse variation definition
  (1500 * 0.5 = k) →                    -- initial condition
  (3000 * q = k) →                      -- new condition
  q = 0.250 := by
sorry

end inverse_variation_problem_l1172_117249


namespace boys_neither_happy_nor_sad_l1172_117200

/-- Prove that the number of boys who are neither happy nor sad is 5 -/
theorem boys_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neither_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neither_children = 20 →
  total_boys = 17 →
  total_girls = 43 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = happy_children + sad_children + neither_children →
  total_children = total_boys + total_girls →
  (total_boys - happy_boys - (sad_children - sad_girls) : ℤ) = 5 := by
sorry


end boys_neither_happy_nor_sad_l1172_117200


namespace square_area_on_parabola_and_line_l1172_117218

theorem square_area_on_parabola_and_line : ∃ (a : ℝ), a > 0 ∧ 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 + 2*x₁ + 1 = 8) ∧ 
    (x₂^2 + 2*x₂ + 1 = 8) ∧ 
    a = (x₂ - x₁)^2) ∧ 
  a = 36 := by
sorry

end square_area_on_parabola_and_line_l1172_117218


namespace cousins_distribution_l1172_117295

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 4 rooms --/
def num_rooms : ℕ := 4

/-- There are 5 cousins --/
def num_cousins : ℕ := 5

/-- The theorem stating that there are 76 ways to distribute 5 cousins into 4 rooms --/
theorem cousins_distribution : distribute num_cousins num_rooms = 76 := by sorry

end cousins_distribution_l1172_117295


namespace sum_of_disk_areas_l1172_117216

/-- The number of disks placed on the circle -/
def n : ℕ := 15

/-- The radius of the large circle -/
def R : ℝ := 1

/-- Represents the arrangement of disks on the circle -/
structure DiskArrangement where
  /-- The radius of each small disk -/
  r : ℝ
  /-- The disks cover the entire circle -/
  covers_circle : r > 0
  /-- The disks do not overlap -/
  no_overlap : 2 * n * r ≤ 2 * π * R
  /-- Each disk is tangent to its neighbors -/
  tangent_neighbors : 2 * n * r = 2 * π * R

/-- The theorem stating the sum of areas of the disks -/
theorem sum_of_disk_areas (arrangement : DiskArrangement) :
  n * π * arrangement.r^2 = 105 * π - 60 * π * Real.sqrt 3 :=
sorry

end sum_of_disk_areas_l1172_117216


namespace rowing_time_calculation_l1172_117243

-- Define the given constants
def man_speed : ℝ := 6
def river_speed : ℝ := 3
def total_distance : ℝ := 4.5

-- Define the theorem
theorem rowing_time_calculation :
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := total_distance / 2
  let upstream_time := one_way_distance / upstream_speed
  let downstream_time := one_way_distance / downstream_speed
  let total_time := upstream_time + downstream_time
  total_time = 1 := by sorry

end rowing_time_calculation_l1172_117243


namespace y_divisibility_l1172_117232

def y : ℕ := 54 + 108 + 162 + 216 + 648 + 810 + 972

theorem y_divisibility :
  (∃ k : ℕ, y = 2 * k) ∧
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 6 * k) ∧
  (∃ k : ℕ, y = 9 * k) :=
by sorry

end y_divisibility_l1172_117232


namespace base9_multiplication_l1172_117213

/-- Converts a base 9 number represented as a list of digits to its decimal equivalent -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 9^i) 0

/-- Converts a decimal number to its base 9 representation as a list of digits -/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Multiplies two base 9 numbers -/
def multiplyBase9 (a b : List Nat) : List Nat :=
  decimalToBase9 ((base9ToDecimal a) * (base9ToDecimal b))

theorem base9_multiplication (a b : List Nat) :
  multiplyBase9 [3, 5, 4] [1, 2] = [1, 2, 5, 1] := by
  sorry

#eval multiplyBase9 [3, 5, 4] [1, 2]

end base9_multiplication_l1172_117213


namespace inequality_solution_l1172_117275

theorem inequality_solution (x : ℝ) :
  0 < x ∧ x < Real.pi →
  ((8 / (3 * Real.sin x - Real.sin (3 * x))) + 3 * (Real.sin x)^2 ≤ 5) ↔
  x = Real.pi / 2 := by sorry

end inequality_solution_l1172_117275


namespace remainder_of_power_700_l1172_117239

theorem remainder_of_power_700 (n : ℕ) (h : n^700 % 100 = 1) : n^700 % 100 = 1 := by
  sorry

end remainder_of_power_700_l1172_117239


namespace unknown_road_length_l1172_117257

/-- Represents a road network with four cities and five roads. -/
structure RoadNetwork where
  /-- The length of the first known road -/
  road1 : ℕ
  /-- The length of the second known road -/
  road2 : ℕ
  /-- The length of the third known road -/
  road3 : ℕ
  /-- The length of the fourth known road -/
  road4 : ℕ
  /-- The length of the unknown road -/
  x : ℕ

/-- The theorem stating that the only possible value for the unknown road length is 17 km. -/
theorem unknown_road_length (network : RoadNetwork) 
  (h1 : network.road1 = 10)
  (h2 : network.road2 = 5)
  (h3 : network.road3 = 8)
  (h4 : network.road4 = 21) :
  network.x = 17 := by
  sorry


end unknown_road_length_l1172_117257


namespace no_solution_steers_cows_l1172_117283

theorem no_solution_steers_cows : ¬∃ (s c : ℕ), 
  30 * s + 32 * c = 1200 ∧ c > s ∧ s > 0 ∧ c > 0 := by
  sorry

end no_solution_steers_cows_l1172_117283


namespace caps_first_week_l1172_117201

/-- The number of caps made in the first week -/
def first_week : ℕ := sorry

/-- The number of caps made in the second week -/
def second_week : ℕ := 400

/-- The number of caps made in the third week -/
def third_week : ℕ := 300

/-- The total number of caps made in four weeks -/
def total_caps : ℕ := 1360

theorem caps_first_week : 
  first_week = 320 ∧
  second_week = 400 ∧
  third_week = 300 ∧
  first_week + second_week + third_week + (first_week + second_week + third_week) / 3 = total_caps :=
by sorry

end caps_first_week_l1172_117201


namespace decimal_to_fraction_l1172_117279

theorem decimal_to_fraction (d : ℚ) (h : d = 0.34) : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d.gcd n = 1 ∧ (n : ℚ) / d = 0.34 ∧ n = 17 := by
  sorry

end decimal_to_fraction_l1172_117279


namespace find_p_value_l1172_117219

-- Define the polynomial (x+y)^9
def polynomial (x y : ℝ) : ℝ := (x + y)^9

-- Define the second term of the expansion
def second_term (x y : ℝ) : ℝ := 9 * x^8 * y

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := 36 * x^7 * y^2

-- Theorem statement
theorem find_p_value (p q : ℝ) : 
  p > 0 ∧ q > 0 ∧ p + q = 1 ∧ second_term p q = third_term p q → p = 4/5 := by
  sorry

end find_p_value_l1172_117219


namespace last_digit_power_difference_l1172_117215

def last_digit (n : ℤ) : ℕ := (n % 10).toNat

theorem last_digit_power_difference (x : ℤ) :
  last_digit (x^95 - 3^58) = 4 → last_digit (x^95) = 3 := by
  sorry

end last_digit_power_difference_l1172_117215


namespace largest_gcd_of_four_integers_l1172_117294

theorem largest_gcd_of_four_integers (a b c d : ℕ+) : 
  a + b + c + d = 1105 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ b) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ c) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ d) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ b ∧ k ∣ c) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ b ∧ k ∣ d) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ c ∧ k ∣ d) →
  (∀ g : ℕ, g ∣ a ∧ g ∣ b ∧ g ∣ c ∧ g ∣ d → g ≤ 221) ∧
  (∃ g : ℕ, g = 221 ∧ g ∣ a ∧ g ∣ b ∧ g ∣ c ∧ g ∣ d) :=
by sorry

end largest_gcd_of_four_integers_l1172_117294


namespace subtracted_number_for_perfect_square_l1172_117244

theorem subtracted_number_for_perfect_square : ∃ n : ℕ, (92555 : ℕ) - 139 = n^2 := by
  sorry

end subtracted_number_for_perfect_square_l1172_117244


namespace complement_intersection_theorem_l1172_117271

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {2, 4}
def N : Finset ℕ := {3, 5}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3, 5} := by sorry

end complement_intersection_theorem_l1172_117271


namespace quadratic_equation_roots_quadratic_equation_specific_roots_l1172_117265

theorem quadratic_equation_roots (m : ℝ) :
  let equation := fun x => m * x^2 - 2 * x + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
sorry

theorem quadratic_equation_specific_roots (m : ℝ) :
  let equation := fun x => m * x^2 - 2 * x + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0 ∧ x₁ * x₂ - x₁ - x₂ = 1/2) →
  m = -2 :=
sorry

end quadratic_equation_roots_quadratic_equation_specific_roots_l1172_117265


namespace remainder_problem_l1172_117231

theorem remainder_problem (N : ℤ) (h : N % 1423 = 215) :
  (N - (N / 109)^2) % 109 = 106 := by
sorry

end remainder_problem_l1172_117231


namespace not_divisible_by_three_l1172_117277

theorem not_divisible_by_three (n : ℕ+) 
  (h : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n.val = k) :
  ¬(3 ∣ n.val) := by
  sorry

end not_divisible_by_three_l1172_117277


namespace six_digit_divisibility_by_seven_l1172_117276

theorem six_digit_divisibility_by_seven (a b c d e f : ℕ) :
  (0 < a) →
  (a < 10) →
  (b < 10) →
  (c < 10) →
  (d < 10) →
  (e < 10) →
  (f < 10) →
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 7 = 0 →
  (100000 * f + 10000 * a + 1000 * b + 100 * c + 10 * d + e) % 7 = 0 := by
sorry

end six_digit_divisibility_by_seven_l1172_117276


namespace min_value_and_inequality_l1172_117222

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := |x - a| + |x + b|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 2) : 
  (a + b = 2) ∧ ¬(a^2 + a > 2 ∧ b^2 + b > 2) := by
  sorry


end min_value_and_inequality_l1172_117222


namespace shoe_price_calculation_l1172_117297

theorem shoe_price_calculation (initial_money : ℝ) (sweater_price : ℝ) (tshirt_price : ℝ) (refund_percentage : ℝ) (final_money : ℝ) :
  initial_money = 74 →
  sweater_price = 9 →
  tshirt_price = 11 →
  refund_percentage = 0.9 →
  final_money = 51 →
  ∃ (shoe_price : ℝ),
    shoe_price = 30 ∧
    final_money = initial_money - sweater_price - tshirt_price - shoe_price + refund_percentage * shoe_price :=
by
  sorry

end shoe_price_calculation_l1172_117297


namespace complex_square_simplify_l1172_117238

theorem complex_square_simplify :
  (4 - 3 * Complex.I) ^ 2 = 7 - 24 * Complex.I :=
by sorry

end complex_square_simplify_l1172_117238


namespace solution_set_inequality_l1172_117251

theorem solution_set_inequality (x : ℝ) :
  Set.Icc (-1/2 : ℝ) 1 ∪ Set.Ioo 1 3 =
  {x | (x + 5) / ((x - 1)^2) ≥ 2 ∧ x ≠ 1} :=
by sorry

end solution_set_inequality_l1172_117251


namespace A_neither_sufficient_nor_necessary_l1172_117259

-- Define propositions A and B
def prop_A (a b : ℝ) : Prop := a + b ≠ 4
def prop_B (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that A is neither sufficient nor necessary for B
theorem A_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, prop_A a b → prop_B a b) ∧
  ¬(∀ a b : ℝ, prop_B a b → prop_A a b) :=
by sorry

end A_neither_sufficient_nor_necessary_l1172_117259


namespace polynomial_roots_theorem_l1172_117208

-- Define the polynomial
def P (a b c : ℂ) (x : ℂ) : ℂ := x^4 - a*x^3 - b*x + c

-- Define the set of solutions
def SolutionSet : Set (ℂ × ℂ × ℂ) :=
  {(a, 0, 0) | a : ℂ} ∪
  {((-1 + Complex.I * Real.sqrt 3) / 2, 1, (-1 + Complex.I * Real.sqrt 3) / 2),
   ((-1 - Complex.I * Real.sqrt 3) / 2, 1, (-1 - Complex.I * Real.sqrt 3) / 2),
   ((1 - Complex.I * Real.sqrt 3) / 2, -1, (1 + Complex.I * Real.sqrt 3) / 2),
   ((1 + Complex.I * Real.sqrt 3) / 2, -1, (1 - Complex.I * Real.sqrt 3) / 2)}

-- The main theorem
theorem polynomial_roots_theorem (a b c : ℂ) :
  (∃ d : ℂ, {a, b, c, d} ⊆ {x : ℂ | P a b c x = 0} ∧ (a, b, c) ∈ SolutionSet) :=
by sorry

end polynomial_roots_theorem_l1172_117208


namespace smallest_n_for_divisible_by_20_l1172_117263

theorem smallest_n_for_divisible_by_20 :
  ∃ (n : ℕ), n = 9 ∧ n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 9 → m ≥ 4 →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T → b ∈ T → c ∈ T → d ∈ T →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    ¬(20 ∣ (a + b - c - d))) :=
by sorry

end smallest_n_for_divisible_by_20_l1172_117263


namespace arithmetic_sequence_difference_l1172_117207

/-- An arithmetic sequence with given first four terms -/
def arithmetic_sequence (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | 2 => 3*x + y
  | 3 => x + 2*y + 2
  | n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that y - x = 2 for the given arithmetic sequence -/
theorem arithmetic_sequence_difference (x y : ℝ) :
  let a := arithmetic_sequence x y
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →
  y - x = 2 := by
  sorry

#check arithmetic_sequence_difference

end arithmetic_sequence_difference_l1172_117207


namespace four_solutions_l1172_117254

/-- The number of integer solutions to x^4 + y^2 = 2y + 1 -/
def solution_count : ℕ := 4

/-- A function that checks if a pair of integers satisfies the equation -/
def satisfies_equation (x y : ℤ) : Prop :=
  x^4 + y^2 = 2*y + 1

/-- The theorem stating that there are exactly 4 integer solutions -/
theorem four_solutions :
  ∃! (solutions : Finset (ℤ × ℤ)), 
    solutions.card = solution_count ∧ 
    ∀ (x y : ℤ), (x, y) ∈ solutions ↔ satisfies_equation x y :=
sorry

end four_solutions_l1172_117254


namespace unique_solution_quadratic_inequality_l1172_117246

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 - a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
by sorry

end unique_solution_quadratic_inequality_l1172_117246


namespace equation_solution_l1172_117292

theorem equation_solution : ∃! (x : ℝ), (81 : ℝ) ^ (x - 2) / (9 : ℝ) ^ (x - 1) = (729 : ℝ) ^ (3 * x - 1) := by
  sorry

end equation_solution_l1172_117292


namespace simplify_expression_l1172_117203

theorem simplify_expression (x y : ℝ) : (x - y) * (x + y) + (x - y)^2 = 2*x^2 - 2*x*y := by
  sorry

end simplify_expression_l1172_117203


namespace division_problem_l1172_117299

theorem division_problem (total : ℚ) (a_amt b_amt c_amt : ℚ) : 
  total = 544 →
  a_amt = (2/3) * b_amt →
  b_amt = (1/4) * c_amt →
  a_amt + b_amt + c_amt = total →
  b_amt = 96 := by
sorry

end division_problem_l1172_117299


namespace intersection_empty_implies_a_range_complement_intersection_when_a_minimum_l1172_117245

-- Define set A
def A (a : ℝ) : Set ℝ :=
  {y | y^2 - (a^2 + a + 1)*y + a*(a^2 + 1) > 0}

-- Define set B
def B : Set ℝ :=
  {y | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = x^2 - x + 1}

-- Theorem 1
theorem intersection_empty_implies_a_range (a : ℝ) :
  A a ∩ B = ∅ → 1 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2
theorem complement_intersection_when_a_minimum :
  let a : ℝ := -2
  (∀ x : ℝ, x^2 + 1 ≥ a*x) →
  (Set.compl (A a) ∩ B) = {y : ℝ | 2 ≤ y ∧ y ≤ 4} :=
sorry

end intersection_empty_implies_a_range_complement_intersection_when_a_minimum_l1172_117245


namespace fraction_of_woodwind_and_brass_players_l1172_117274

theorem fraction_of_woodwind_and_brass_players (total_students : ℝ) : 
  let woodwind_last_year := (1 / 2 : ℝ) * total_students
  let brass_last_year := (2 / 5 : ℝ) * total_students
  let percussion_last_year := (1 / 10 : ℝ) * total_students
  let woodwind_this_year := (1 / 2 : ℝ) * woodwind_last_year
  let brass_this_year := (3 / 4 : ℝ) * brass_last_year
  let percussion_this_year := percussion_last_year
  let total_this_year := woodwind_this_year + brass_this_year + percussion_this_year
  (woodwind_this_year + brass_this_year) / total_this_year = (11 / 20 : ℝ) :=
by sorry

end fraction_of_woodwind_and_brass_players_l1172_117274


namespace mia_darwin_money_multiple_l1172_117267

theorem mia_darwin_money_multiple (darwin_money mia_money : ℕ) (multiple : ℚ) : 
  darwin_money = 45 →
  mia_money = 110 →
  mia_money = multiple * darwin_money + 20 →
  multiple = 2 := by sorry

end mia_darwin_money_multiple_l1172_117267


namespace hotel_towels_l1172_117229

theorem hotel_towels (num_rooms : ℕ) (people_per_room : ℕ) (total_towels : ℕ) : 
  num_rooms = 10 →
  people_per_room = 3 →
  total_towels = 60 →
  total_towels / (num_rooms * people_per_room) = 2 := by
sorry

end hotel_towels_l1172_117229


namespace tennis_balls_order_l1172_117236

theorem tennis_balls_order (white yellow : ℕ) : 
  white = yellow →
  white / (yellow + 90) = 8 / 13 →
  white + yellow = 288 :=
by sorry

end tennis_balls_order_l1172_117236


namespace quadratic_always_nonnegative_implies_a_range_l1172_117214

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → a ∈ Set.Icc (-2) 2 :=
by
  sorry

end quadratic_always_nonnegative_implies_a_range_l1172_117214


namespace complex_arithmetic_result_l1172_117230

def B : ℂ := 5 - 2 * Complex.I
def N : ℂ := -5 + 2 * Complex.I
def T : ℂ := 2 * Complex.I
def Q : ℂ := 3

theorem complex_arithmetic_result : B - N + T - Q = 7 - 2 * Complex.I := by
  sorry

end complex_arithmetic_result_l1172_117230


namespace parabola_properties_l1172_117225

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- Theorem about properties of the parabola y = x^2 + 2x - 3 -/
theorem parabola_properties :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, -3)
  ∀ x ∈ Set.Icc (-3 : ℝ) 2,
  (f A.1 = A.2 ∧ f B.1 = B.2 ∧ f C.1 = C.2) ∧
  (A.1 < B.1) ∧
  (∃ (y_max y_min : ℝ), 
    (∀ x' ∈ Set.Icc (-3 : ℝ) 2, f x' ≤ y_max ∧ f x' ≥ y_min) ∧
    y_max - y_min = 9) :=
by sorry

end parabola_properties_l1172_117225


namespace g36_values_product_l1172_117260

def is_valid_g (g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 3 * g (a^2 + b^2) = (g a)^2 + (g b)^2 + g a * g b

def possible_g36_values (g : ℕ → ℕ) : Set ℕ :=
  {x : ℕ | ∃ h : is_valid_g g, g 36 = x}

theorem g36_values_product (g : ℕ → ℕ) (h : is_valid_g g) :
  (Finset.card (Finset.image g {36})) * (Finset.sum (Finset.image g {36}) id) = 2 := by
  sorry

end g36_values_product_l1172_117260


namespace consecutive_integers_average_l1172_117287

theorem consecutive_integers_average (x : ℤ) : 
  (((x - 9) + (x - 7) + (x - 5) + (x - 3) + (x - 1) + (x + 1) + (x + 3) + (x + 5) + (x + 7) + (x + 9)) / 10 : ℚ) = 31/2 →
  ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 : ℚ) = 49/2 :=
by
  sorry

end consecutive_integers_average_l1172_117287


namespace sum_of_factors_l1172_117221

theorem sum_of_factors (m n p q : ℤ) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (5 - m) * (5 - n) * (5 - p) * (5 - q) = 9 →
  m + n + p + q = 20 := by
sorry

end sum_of_factors_l1172_117221


namespace class_size_l1172_117296

/-- The number of people who like both baseball and football -/
def both : ℕ := 5

/-- The number of people who only like baseball -/
def only_baseball : ℕ := 2

/-- The number of people who only like football -/
def only_football : ℕ := 3

/-- The number of people who like neither baseball nor football -/
def neither : ℕ := 6

/-- The total number of people in the class -/
def total : ℕ := both + only_baseball + only_football + neither

theorem class_size : total = 16 := by sorry

end class_size_l1172_117296


namespace ratio_A_B_between_zero_and_one_l1172_117240

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem ratio_A_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end ratio_A_B_between_zero_and_one_l1172_117240


namespace gcd_lcm_sum_for_special_case_l1172_117209

theorem gcd_lcm_sum_for_special_case (a b : ℕ) (h : a = 1999 * b) :
  Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end gcd_lcm_sum_for_special_case_l1172_117209


namespace amanda_final_pay_l1172_117226

/-- Calculate Amanda's final pay after deductions and penalties --/
theorem amanda_final_pay 
  (regular_wage : ℝ) 
  (regular_hours : ℝ) 
  (overtime_rate : ℝ) 
  (overtime_hours : ℝ) 
  (commission : ℝ) 
  (tax_rate : ℝ) 
  (insurance_rate : ℝ) 
  (other_expenses : ℝ) 
  (penalty_rate : ℝ) 
  (h1 : regular_wage = 50)
  (h2 : regular_hours = 8)
  (h3 : overtime_rate = 1.5)
  (h4 : overtime_hours = 2)
  (h5 : commission = 150)
  (h6 : tax_rate = 0.15)
  (h7 : insurance_rate = 0.05)
  (h8 : other_expenses = 40)
  (h9 : penalty_rate = 0.2) :
  let total_earnings := regular_wage * regular_hours + 
                        regular_wage * overtime_rate * overtime_hours + 
                        commission
  let deductions := total_earnings * tax_rate + 
                    total_earnings * insurance_rate + 
                    other_expenses
  let earnings_after_deductions := total_earnings - deductions
  let penalty := earnings_after_deductions * penalty_rate
  let final_pay := earnings_after_deductions - penalty
  final_pay = 416 := by sorry

end amanda_final_pay_l1172_117226


namespace number_divided_by_3000_l1172_117284

theorem number_divided_by_3000 : 
  ∃ x : ℝ, x / 3000 = 0.008416666666666666 ∧ x = 25.25 :=
by sorry

end number_divided_by_3000_l1172_117284


namespace parabola_shift_l1172_117298

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 5 * x^2

-- Define the horizontal shift
def horizontal_shift : ℝ := 2

-- Define the vertical shift
def vertical_shift : ℝ := 3

-- Define the resulting parabola after shifts
def shifted_parabola (x : ℝ) : ℝ := 5 * (x + horizontal_shift)^2 + vertical_shift

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  sorry

end parabola_shift_l1172_117298
