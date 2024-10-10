import Mathlib

namespace total_carrots_is_nine_l3819_381989

-- Define the number of carrots grown by Sandy
def sandy_carrots : ℕ := 6

-- Define the number of carrots grown by Sam
def sam_carrots : ℕ := 3

-- Define the total number of carrots
def total_carrots : ℕ := sandy_carrots + sam_carrots

-- Theorem stating that the total number of carrots is 9
theorem total_carrots_is_nine : total_carrots = 9 := by
  sorry

end total_carrots_is_nine_l3819_381989


namespace scaling_transformation_theorem_l3819_381935

/-- Scaling transformation -/
def scaling (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 3 * y)

/-- Transformed curve equation -/
def transformed_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

/-- Original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  4 * x^2 + 9 * y^2 = 1

/-- Theorem: If the transformed curve satisfies the equation,
    then the original curve satisfies its corresponding equation -/
theorem scaling_transformation_theorem :
  ∀ x y : ℝ,
  let (x'', y'') := scaling x y
  transformed_curve x'' y'' → original_curve x y :=
by
  sorry

end scaling_transformation_theorem_l3819_381935


namespace elizabeth_subtraction_l3819_381939

theorem elizabeth_subtraction (n : ℕ) (h1 : n = 50) (h2 : n^2 + 101 = (n+1)^2) : n^2 - (n-1)^2 = 99 := by
  sorry

end elizabeth_subtraction_l3819_381939


namespace M_intersect_N_empty_l3819_381936

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1}

-- Theorem statement
theorem M_intersect_N_empty : M ∩ (N.image Prod.fst) = ∅ := by
  sorry

end M_intersect_N_empty_l3819_381936


namespace g_inverse_of_f_l3819_381934

-- Define the original function f
def f (x : ℝ) : ℝ := 4 - 5 * x^2

-- Define the inverse function g
def g (x : ℝ) : Set ℝ := {y : ℝ | y^2 = (4 - x) / 5 ∧ y ≥ 0 ∨ y^2 = (4 - x) / 5 ∧ y < 0}

-- Theorem stating that g is the inverse of f
theorem g_inverse_of_f : 
  ∀ x ∈ Set.range f, ∀ y ∈ g x, f y = x ∧ y ∈ Set.range f :=
sorry

end g_inverse_of_f_l3819_381934


namespace impossible_coin_probabilities_l3819_381993

theorem impossible_coin_probabilities :
  ¬∃ (p₁ p₂ : ℝ), 0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧
    (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
    p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end impossible_coin_probabilities_l3819_381993


namespace benny_eggs_l3819_381912

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Benny bought -/
def dozens_bought : ℕ := 7

/-- Theorem: Benny bought 84 eggs -/
theorem benny_eggs : dozens_bought * eggs_per_dozen = 84 := by
  sorry

end benny_eggs_l3819_381912


namespace rectangle_y_value_l3819_381998

/-- Given a rectangle with vertices at (-1, y), (7, y), (-1, 3), and (7, 3),
    where y is positive and the area is 72 square units, y must equal 12. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) (h2 : 8 * (y - 3) = 72) : y = 12 := by
  sorry

end rectangle_y_value_l3819_381998


namespace triangle_sin_c_equals_one_l3819_381948

theorem triangle_sin_c_equals_one 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : a = 1) 
  (h2 : b = Real.sqrt 3) 
  (h3 : A + C = 2 * B) 
  (h4 : 0 < A ∧ A < π) 
  (h5 : 0 < B ∧ B < π) 
  (h6 : 0 < C ∧ C < π) 
  (h7 : A + B + C = π) 
  (h8 : a / Real.sin A = b / Real.sin B) 
  (h9 : b / Real.sin B = c / Real.sin C) 
  : Real.sin C = 1 := by
  sorry

end triangle_sin_c_equals_one_l3819_381948


namespace girls_fraction_l3819_381940

theorem girls_fraction (total : ℕ) (middle : ℕ) 
  (h_total : total = 800)
  (h_middle : middle = 330)
  (h_primary : total - middle = 470) :
  ∃ (girls boys : ℕ),
    girls + boys = total ∧
    (7 : ℚ) / 10 * girls + (2 : ℚ) / 5 * boys = total - middle ∧
    (girls : ℚ) / total = 5 / 8 :=
by sorry

end girls_fraction_l3819_381940


namespace ellipse_equation_l3819_381962

/-- Given an ellipse with focal distance 8 and the sum of distances from any point 
    on the ellipse to the two foci being 10, prove that its standard equation is 
    either x²/25 + y²/9 = 1 or y²/25 + x²/9 = 1 -/
theorem ellipse_equation (focal_distance : ℝ) (sum_distances : ℝ) 
  (h1 : focal_distance = 8) (h2 : sum_distances = 10) :
  (∃ x y : ℝ, x^2/25 + y^2/9 = 1) ∨ (∃ x y : ℝ, y^2/25 + x^2/9 = 1) :=
sorry

end ellipse_equation_l3819_381962


namespace min_overlap_beethoven_vivaldi_l3819_381954

theorem min_overlap_beethoven_vivaldi (total : ℕ) (beethoven : ℕ) (vivaldi : ℕ) 
  (h1 : total = 200) 
  (h2 : beethoven = 160) 
  (h3 : vivaldi = 130) : 
  ∃ (both : ℕ), both ≥ 90 ∧ 
    (∀ (x : ℕ), x < 90 → ¬(x ≤ beethoven ∧ x ≤ vivaldi ∧ beethoven + vivaldi - x ≤ total)) :=
by
  sorry

#check min_overlap_beethoven_vivaldi

end min_overlap_beethoven_vivaldi_l3819_381954


namespace pilot_miles_flown_l3819_381922

theorem pilot_miles_flown (tuesday_miles : ℕ) (thursday_miles : ℕ) (total_weeks : ℕ) (total_miles : ℕ) : 
  thursday_miles = 1475 → 
  total_weeks = 3 → 
  total_miles = 7827 → 
  total_miles = total_weeks * (tuesday_miles + thursday_miles) → 
  tuesday_miles = 1134 := by
sorry

end pilot_miles_flown_l3819_381922


namespace fib_100_mod_8_l3819_381946

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Theorem statement
theorem fib_100_mod_8 : fib 100 % 8 = 3 := by
  sorry

end fib_100_mod_8_l3819_381946


namespace flip_ratio_l3819_381960

/-- The number of flips in a triple-flip -/
def tripleFlip : ℕ := 3

/-- The number of flips in a double-flip -/
def doubleFlip : ℕ := 2

/-- The number of triple-flips Jen performed -/
def jenFlips : ℕ := 16

/-- The number of double-flips Tyler performed -/
def tylerFlips : ℕ := 12

/-- Theorem stating that the ratio of Tyler's flips to Jen's flips is 1:2 -/
theorem flip_ratio :
  (tylerFlips * doubleFlip : ℚ) / (jenFlips * tripleFlip) = 1 / 2 := by
  sorry

end flip_ratio_l3819_381960


namespace min_baskets_needed_l3819_381950

/-- Represents the number of points earned for scoring a basket -/
def points_per_basket : ℤ := 3

/-- Represents the number of points deducted for missing a basket -/
def points_per_miss : ℤ := 1

/-- Represents the total number of shots taken -/
def total_shots : ℕ := 12

/-- Represents the minimum score Xiao Li wants to achieve -/
def min_score : ℤ := 28

/-- Calculates the score based on the number of baskets made -/
def score (baskets_made : ℕ) : ℤ :=
  points_per_basket * baskets_made - points_per_miss * (total_shots - baskets_made)

/-- Proves that Xiao Li needs to make at least 10 baskets to score at least 28 points -/
theorem min_baskets_needed :
  ∀ baskets_made : ℕ, baskets_made ≤ total_shots →
    (∀ n : ℕ, n < baskets_made → score n < min_score) →
    score baskets_made ≥ min_score →
    baskets_made ≥ 10 :=
by sorry

end min_baskets_needed_l3819_381950


namespace lines_per_page_l3819_381956

theorem lines_per_page (total_lines : ℕ) (num_pages : ℕ) (lines_per_page : ℕ) 
  (h1 : total_lines = 150)
  (h2 : num_pages = 5)
  (h3 : lines_per_page * num_pages = total_lines) :
  lines_per_page = 30 := by
  sorry

end lines_per_page_l3819_381956


namespace fifteenth_term_of_sequence_l3819_381979

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem fifteenth_term_of_sequence :
  let a₁ : ℤ := -3
  let d : ℤ := 4
  arithmetic_sequence a₁ d 15 = 53 := by
sorry

end fifteenth_term_of_sequence_l3819_381979


namespace necessary_not_sufficient_condition_l3819_381933

theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∀ c, c ≠ 0 → (a * c^2 > b * c^2 → a > b)) ∧
  (∃ c, a > b ∧ ¬(a * c^2 > b * c^2)) :=
sorry

end necessary_not_sufficient_condition_l3819_381933


namespace bridge_length_calculation_l3819_381999

theorem bridge_length_calculation (train_length : ℝ) (signal_post_time : ℝ) (bridge_time : ℝ) :
  train_length = 600 →
  signal_post_time = 40 →
  bridge_time = 480 →
  let train_speed := train_length / signal_post_time
  let total_distance := train_speed * bridge_time
  let bridge_length := total_distance - train_length
  bridge_length = 6600 := by
  sorry

end bridge_length_calculation_l3819_381999


namespace right_triangle_leg_square_l3819_381937

theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) : ∃ b : ℝ, b^2 = 2*a + 4 := by
  sorry

end right_triangle_leg_square_l3819_381937


namespace probability_theorem_l3819_381964

/-- Represents the number of athletes in each association -/
structure Associations where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of athletes selected from each association -/
structure SelectedAthletes where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Calculates the probability of selecting at least one athlete from A5 or A6 -/
def probability_A5_or_A6 (total_selected : ℕ) (doubles_team_size : ℕ) : ℚ :=
  let favorable_outcomes := (total_selected - 2) * 2 + 1
  let total_outcomes := total_selected.choose doubles_team_size
  favorable_outcomes / total_outcomes

/-- Main theorem statement -/
theorem probability_theorem (assoc : Associations) (selected : SelectedAthletes) :
    assoc.A = 27 ∧ assoc.B = 9 ∧ assoc.C = 18 →
    selected.A = 3 ∧ selected.B = 1 ∧ selected.C = 2 →
    probability_A5_or_A6 (selected.A + selected.B + selected.C) 2 = 3/5 := by
  sorry


end probability_theorem_l3819_381964


namespace red_car_position_implies_816_cars_l3819_381904

/-- The position of the red car in the parking lot -/
structure CarPosition where
  left : ℕ
  right : ℕ
  front : ℕ
  back : ℕ

/-- The dimensions of the parking lot -/
def parkingLotDimensions (pos : CarPosition) : ℕ × ℕ :=
  (pos.left + pos.right - 1, pos.front + pos.back - 1)

/-- The total number of cars in the parking lot -/
def totalCars (pos : CarPosition) : ℕ :=
  let (width, length) := parkingLotDimensions pos
  width * length

/-- Theorem stating that given the red car's position, the total number of cars is 816 -/
theorem red_car_position_implies_816_cars (pos : CarPosition)
    (h_left : pos.left = 19)
    (h_right : pos.right = 16)
    (h_front : pos.front = 14)
    (h_back : pos.back = 11) :
    totalCars pos = 816 := by
  sorry

#eval totalCars ⟨19, 16, 14, 11⟩

end red_car_position_implies_816_cars_l3819_381904


namespace angle_measure_proof_l3819_381928

theorem angle_measure_proof : Real.arccos (Real.sin (19 * π / 180)) = 71 * π / 180 := by
  sorry

end angle_measure_proof_l3819_381928


namespace turner_rollercoaster_rides_l3819_381981

/-- The number of times Turner wants to ride the rollercoaster -/
def R : ℕ := sorry

/-- The cost in tickets for one ride on the rollercoaster -/
def rollercoasterCost : ℕ := 4

/-- The cost in tickets for one ride on the Catapult -/
def catapultCost : ℕ := 4

/-- The cost in tickets for one ride on the Ferris wheel -/
def ferrisWheelCost : ℕ := 1

/-- The number of times Turner wants to ride the Catapult -/
def catapultRides : ℕ := 2

/-- The number of times Turner wants to ride the Ferris wheel -/
def ferrisWheelRides : ℕ := 1

/-- The total number of tickets Turner needs -/
def totalTickets : ℕ := 21

theorem turner_rollercoaster_rides :
  R * rollercoasterCost + catapultRides * catapultCost + ferrisWheelRides * ferrisWheelCost = totalTickets ∧ R = 3 := by
  sorry

end turner_rollercoaster_rides_l3819_381981


namespace snarks_are_twerks_and_quarks_l3819_381986

variable (U : Type) -- Universe of discourse

-- Define the predicates
variable (Snark Garble Twerk Quark : U → Prop)

-- State the given conditions
variable (h1 : ∀ x, Snark x → Garble x)
variable (h2 : ∀ x, Twerk x → Garble x)
variable (h3 : ∀ x, Snark x → Quark x)
variable (h4 : ∀ x, Quark x → Twerk x)

-- State the theorem to be proved
theorem snarks_are_twerks_and_quarks :
  ∀ x, Snark x → Twerk x ∧ Quark x := by sorry

end snarks_are_twerks_and_quarks_l3819_381986


namespace opposite_sides_range_l3819_381973

/-- Given two points A and B on opposite sides of a line, prove the range of y₀ -/
theorem opposite_sides_range (y₀ : ℝ) : 
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (1, y₀)
  let line (x y : ℝ) : ℝ := x - 2*y + 5
  (line A.1 A.2) * (line B.1 B.2) < 0 → y₀ > 3 :=
by sorry

end opposite_sides_range_l3819_381973


namespace triangle_abc_properties_l3819_381931

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  ((b - c)^2 = a^2 - b*c) →
  (a = 3) →
  (Real.sin C = 2 * Real.sin B) →
  (a = 2 * b * Real.sin (C/2)) →
  (b = 2 * c * Real.sin (A/2)) →
  (c = 2 * a * Real.sin (B/2)) →
  (A = π/3 ∧ (1/2 * b * c * Real.sin A = 3*Real.sqrt 3/2)) := by
  sorry

end triangle_abc_properties_l3819_381931


namespace sphere_radius_from_cone_volume_l3819_381909

/-- Given a cone with radius 2 inches and height 6 inches, and a sphere with the same physical volume
    but half the density of the cone's material, the radius of the sphere is ∛12 inches. -/
theorem sphere_radius_from_cone_volume (cone_radius : ℝ) (cone_height : ℝ) (sphere_radius : ℝ) :
  cone_radius = 2 →
  cone_height = 6 →
  (1 / 3) * Real.pi * cone_radius^2 * cone_height = (4 / 3) * Real.pi * sphere_radius^3 / 2 →
  sphere_radius = (12 : ℝ)^(1/3) := by
  sorry

end sphere_radius_from_cone_volume_l3819_381909


namespace shaded_area_sum_l3819_381902

/-- Represents a triangle in the hexagon -/
structure Triangle :=
  (size : Nat)
  (area : ℝ)

/-- The hexagon composed of equilateral triangles -/
structure Hexagon :=
  (unit_triangle : Triangle)
  (small : Triangle)
  (medium : Triangle)
  (large : Triangle)

/-- The theorem stating the area of the shaded part -/
theorem shaded_area_sum (h : Hexagon) 
  (h_unit : h.unit_triangle.area = 10)
  (h_small : h.small.size = 1)
  (h_medium : h.medium.size = 6)
  (h_large : h.large.size = 13) :
  h.small.area + h.medium.area + h.large.area = 110 := by
  sorry


end shaded_area_sum_l3819_381902


namespace committee_count_l3819_381968

theorem committee_count (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 1) :
  (Nat.choose (n - k) (m - k)) = 35 := by
  sorry

end committee_count_l3819_381968


namespace cubic_quadratic_relation_l3819_381963

theorem cubic_quadratic_relation (A B C D : ℝ) (u v w : ℝ) (p q : ℝ) :
  (A * u^3 + B * u^2 + C * u + D = 0) →
  (A * v^3 + B * v^2 + C * v + D = 0) →
  (A * w^3 + B * w^2 + C * w + D = 0) →
  (u^2 + p * u^2 + q = 0) →
  (v^2 + p * v^2 + q = 0) →
  (p = (B^2 - 2*C) / A^2) :=
by sorry

end cubic_quadratic_relation_l3819_381963


namespace quadratic_factorization_l3819_381969

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end quadratic_factorization_l3819_381969


namespace intersection_area_is_nine_l3819_381903

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the intersection of two rectangles -/
structure Intersection where
  rect1 : Rectangle
  rect2 : Rectangle
  isSquare : Bool
  area : ℝ

/-- Theorem: Area of intersection between two specific rectangles -/
theorem intersection_area_is_nine 
  (r1 : Rectangle) 
  (r2 : Rectangle) 
  (i : Intersection) 
  (h1 : r1.width = 4 ∧ r1.height = 12) 
  (h2 : r2.width = 3 ∧ r2.height = 7) 
  (h3 : i.rect1 = r1 ∧ i.rect2 = r2) 
  (h4 : i.isSquare = true) : 
  i.area = 9 := by
  sorry


end intersection_area_is_nine_l3819_381903


namespace min_value_of_f_max_value_of_sum_squares_max_value_is_tight_l3819_381983

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 3|

-- Theorem for the minimum value of f
theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 5/2 := by sorry

-- Theorem for the maximum value of a² + 2b² + 3c²
theorem max_value_of_sum_squares (a b c : ℝ) (h : a^4 + b^4 + c^4 = 5/2) :
  a^2 + 2*b^2 + 3*c^2 ≤ Real.sqrt (35/2) := by sorry

-- Theorem to show that the upper bound is tight
theorem max_value_is_tight :
  ∃ (a b c : ℝ), a^4 + b^4 + c^4 = 5/2 ∧ a^2 + 2*b^2 + 3*c^2 = Real.sqrt (35/2) := by sorry

end min_value_of_f_max_value_of_sum_squares_max_value_is_tight_l3819_381983


namespace square_sum_from_means_l3819_381917

theorem square_sum_from_means (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h_arithmetic : (p + q + r) / 3 = 10)
  (h_geometric : (p * q * r) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/p + 1/q + 1/r) = 4) :
  p^2 + q^2 + r^2 = 576 := by
sorry

end square_sum_from_means_l3819_381917


namespace yvonne_success_probability_l3819_381913

theorem yvonne_success_probability 
  (p_xavier : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_yvonne_not_zelda : ℝ) 
  (h1 : p_xavier = 1/3) 
  (h2 : p_zelda = 5/8) 
  (h3 : p_xavier_yvonne_not_zelda = 0.0625) : 
  ∃ p_yvonne : ℝ, p_yvonne = 0.5 ∧ 
    p_xavier * p_yvonne * (1 - p_zelda) = p_xavier_yvonne_not_zelda :=
by sorry

end yvonne_success_probability_l3819_381913


namespace total_points_proof_l3819_381919

def sam_points : ℕ := 75
def friend_points : ℕ := 12

theorem total_points_proof :
  sam_points + friend_points = 87 := by sorry

end total_points_proof_l3819_381919


namespace inequality_proof_l3819_381900

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^3 + b^3 + 3*a*b = 1) (h2 : c + d = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 + (d + 1/d)^3 ≥ 40 := by
sorry

end inequality_proof_l3819_381900


namespace average_score_bounds_l3819_381906

def score_distribution : List (ℕ × ℕ) :=
  [(100, 2), (90, 9), (80, 17), (70, 28), (60, 36), (50, 7), (48, 1)]

def total_students : ℕ := (score_distribution.map Prod.snd).sum

def min_score_sum : ℕ := (score_distribution.map (λ (s, n) => s * n)).sum

def max_score_sum : ℕ := (score_distribution.map (λ (s, n) => 
  if s = 100 then s * n else (s + 9) * n)).sum

theorem average_score_bounds :
  (min_score_sum : ℚ) / total_students > 68 ∧
  (max_score_sum : ℚ) / total_students < 78 := by
  sorry

end average_score_bounds_l3819_381906


namespace existence_of_three_quadratic_polynomials_l3819_381972

/-- A quadratic polynomial of the form ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ :=
  p.b ^ 2 - 4 * p.a * p.c

/-- A quadratic polynomial has two distinct real roots if its discriminant is positive -/
def has_two_distinct_real_roots (p : QuadraticPolynomial) : Prop :=
  discriminant p > 0

/-- The sum of two quadratic polynomials -/
def sum_polynomials (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  { a := p.a + q.a, b := p.b + q.b, c := p.c + q.c }

/-- A quadratic polynomial has no real roots if its discriminant is negative -/
def has_no_real_roots (p : QuadraticPolynomial) : Prop :=
  discriminant p < 0

theorem existence_of_three_quadratic_polynomials :
  ∃ (p₁ p₂ p₃ : QuadraticPolynomial),
    (has_two_distinct_real_roots p₁) ∧
    (has_two_distinct_real_roots p₂) ∧
    (has_two_distinct_real_roots p₃) ∧
    (has_no_real_roots (sum_polynomials p₁ p₂)) ∧
    (has_no_real_roots (sum_polynomials p₁ p₃)) ∧
    (has_no_real_roots (sum_polynomials p₂ p₃)) := by
  sorry

end existence_of_three_quadratic_polynomials_l3819_381972


namespace hyperbola_equation_l3819_381974

def is_hyperbola (a b : ℝ) (h : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1

def has_asymptote (h : ℝ → ℝ → Prop) : Prop :=
  ∀ x, h x (Real.sqrt 3 * x)

def has_focus (h : ℝ → ℝ → Prop) : Prop :=
  h 2 0

theorem hyperbola_equation (a b : ℝ) (h : ℝ → ℝ → Prop)
  (ha : a > 0) (hb : b > 0)
  (h_hyp : is_hyperbola a b h)
  (h_asym : has_asymptote h)
  (h_focus : has_focus h) :
  ∀ x y, h x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

end hyperbola_equation_l3819_381974


namespace find_n_l3819_381938

def is_valid_n (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ r : ℕ, r > 0 ∧ 
    (2287 % n = r) ∧ 
    (2028 % n = r) ∧ 
    (1806 % n = r)

theorem find_n : 
  (∀ m : ℕ, is_valid_n m → m ≤ 37) ∧ 
  is_valid_n 37 :=
sorry

end find_n_l3819_381938


namespace parametric_to_cartesian_l3819_381996

theorem parametric_to_cartesian (t : ℝ) :
  let x := 3 * t + 6
  let y := 5 * t - 8
  y = (5/3) * x - 18 := by
sorry

end parametric_to_cartesian_l3819_381996


namespace matrix_multiplication_result_l3819_381977

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 5, 0]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 6; -1, 2]
  A * B = !![2, 14; 0, 30] := by sorry

end matrix_multiplication_result_l3819_381977


namespace tan_arccos_three_fifths_l3819_381901

theorem tan_arccos_three_fifths :
  Real.tan (Real.arccos (3/5)) = 4/3 := by
  sorry

end tan_arccos_three_fifths_l3819_381901


namespace rectangle_area_increase_l3819_381975

theorem rectangle_area_increase (x y : ℝ) :
  let original_area := 1
  let new_length := 1 + x / 100
  let new_width := 1 + y / 100
  let new_area := new_length * new_width
  let area_increase_percentage := (new_area - original_area) / original_area * 100
  area_increase_percentage = x + y + (x * y / 100) :=
by sorry

end rectangle_area_increase_l3819_381975


namespace abs_diff_opposite_for_negative_l3819_381910

theorem abs_diff_opposite_for_negative (x : ℝ) (h : x < 0) : |x - (-x)| = -2*x := by
  sorry

end abs_diff_opposite_for_negative_l3819_381910


namespace division_remainder_l3819_381976

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 760 → divisor = 36 → quotient = 21 → 
  dividend = divisor * quotient + remainder → remainder = 4 := by
sorry

end division_remainder_l3819_381976


namespace sugar_calculation_l3819_381925

theorem sugar_calculation (original_sugar : ℚ) (recipe_fraction : ℚ) : 
  original_sugar = 7 + 1/3 →
  recipe_fraction = 2/3 →
  recipe_fraction * original_sugar = 4 + 8/9 := by
sorry

end sugar_calculation_l3819_381925


namespace divisibility_in_sequence_l3819_381951

theorem divisibility_in_sequence (x : Fin 2020 → ℤ) :
  ∃ i j : Fin 2020, i ≠ j ∧ (x j - x i) % 2019 = 0 := by
  sorry

end divisibility_in_sequence_l3819_381951


namespace f_4_3_2_1_l3819_381959

/-- The mapping f from (a₁, a₂, a₃, a₄) to (b₁, b₂, b₃, b₄) based on the equation
    x^4 + a₁x³ + a₂x² + a₃x + a₄ = (x+1)^4 + b₁(x+1)³ + b₂(x+1)² + b₃(x+1) + b₄ -/
def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem f_4_3_2_1 : f 4 3 2 1 = (0, -3, 4, -1) := by
  sorry

end f_4_3_2_1_l3819_381959


namespace train_speed_l3819_381991

/-- The speed of a train given its length, time to cross a person, and the person's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 1200 →
  crossing_time = 71.99424046076314 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), 
    (abs (train_speed - 63.00468) < 0.00001) ∧ 
    (train_speed * 1000 / 3600 - man_speed * 1000 / 3600) * crossing_time = train_length :=
by sorry

end train_speed_l3819_381991


namespace range_of_a_l3819_381915

-- Define propositions A and B
def PropA (x : ℝ) : Prop := (x - 1)^2 < 9
def PropB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Define the set of x satisfying proposition A
def SetA : Set ℝ := {x | PropA x}

-- Define the set of x satisfying proposition B for a given a
def SetB (a : ℝ) : Set ℝ := {x | PropB x a}

-- Define the condition that A is sufficient but not necessary for B
def ASufficientNotNecessary (a : ℝ) : Prop :=
  SetA ⊂ SetB a ∧ SetA ≠ SetB a

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, ASufficientNotNecessary a ↔ a < -4 :=
sorry

end range_of_a_l3819_381915


namespace theta_range_l3819_381924

theorem theta_range (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi))
  (h2 : Real.cos θ ^ 5 - Real.sin θ ^ 5 < 7 * (Real.sin θ ^ 3 - Real.cos θ ^ 3)) :
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end theta_range_l3819_381924


namespace sum_at_two_and_neg_two_l3819_381952

/-- A cubic polynomial Q with specific values at 0, 1, and -1 -/
structure CubicPolynomial (m : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ a b c d : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + d
  value_at_zero : Q 0 = m
  value_at_one : Q 1 = 3 * m
  value_at_neg_one : Q (-1) = 4 * m

/-- The sum of the polynomial values at 2 and -2 is 22m -/
theorem sum_at_two_and_neg_two (m : ℝ) (Q : CubicPolynomial m) :
  Q.Q 2 + Q.Q (-2) = 22 * m := by
  sorry

end sum_at_two_and_neg_two_l3819_381952


namespace distribute_5_8_l3819_381926

/-- The number of ways to distribute n different items into m boxes with at most one item per box -/
def distribute (n m : ℕ) : ℕ :=
  (m - n + 1).factorial * (m.choose n)

/-- Theorem: The number of ways to distribute 5 different items into 8 boxes
    with at most one item per box is 6720 -/
theorem distribute_5_8 : distribute 5 8 = 6720 := by
  sorry

#eval distribute 5 8

end distribute_5_8_l3819_381926


namespace expression_evaluation_l3819_381971

theorem expression_evaluation :
  65 + (160 / 8) + (35 * 12) - 450 - (504 / 7) = -17 := by
  sorry

end expression_evaluation_l3819_381971


namespace ferry_speed_difference_l3819_381966

/-- Proves that the difference in speed between ferry Q and ferry P is 3 km/h -/
theorem ferry_speed_difference : 
  ∀ (Vp Vq : ℝ) (time_p time_q : ℝ) (distance_p distance_q : ℝ),
  Vp = 6 →  -- Ferry P's speed
  time_p = 3 →  -- Ferry P's travel time
  distance_p = Vp * time_p →  -- Ferry P's distance
  distance_q = 2 * distance_p →  -- Ferry Q's distance is twice Ferry P's
  time_q = time_p + 1 →  -- Ferry Q's travel time is 1 hour longer
  Vq = distance_q / time_q →  -- Ferry Q's speed
  Vq - Vp = 3 := by
sorry

end ferry_speed_difference_l3819_381966


namespace paper_folding_thickness_l3819_381905

theorem paper_folding_thickness (initial_thickness : ℝ) (target_thickness : ℝ) : 
  initial_thickness > 0 → target_thickness > 0 →
  (∃ n : ℕ, (2^n : ℝ) * initial_thickness > target_thickness) →
  (∀ m : ℕ, m < 8 → (2^m : ℝ) * initial_thickness ≤ target_thickness) →
  (∃ n : ℕ, n = 8 ∧ (2^n : ℝ) * initial_thickness > target_thickness) :=
by sorry

end paper_folding_thickness_l3819_381905


namespace complex_number_quadrant_l3819_381953

theorem complex_number_quadrant : ∃ (z : ℂ), z = (Complex.I : ℂ) / (1 + Complex.I) ∧ 0 < z.re ∧ 0 < z.im := by
  sorry

end complex_number_quadrant_l3819_381953


namespace sin_cos_sum_18_12_l3819_381949

theorem sin_cos_sum_18_12 : 
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_18_12_l3819_381949


namespace lace_per_ruffle_length_l3819_381980

/-- Proves that given the conditions of Carolyn's dress trimming,
    the length of lace used for each ruffle is 20 cm. -/
theorem lace_per_ruffle_length
  (cuff_length : ℝ)
  (hem_length : ℝ)
  (num_cuffs : ℕ)
  (num_ruffles : ℕ)
  (lace_cost_per_meter : ℝ)
  (total_spent : ℝ)
  (h1 : cuff_length = 50)
  (h2 : hem_length = 300)
  (h3 : num_cuffs = 2)
  (h4 : num_ruffles = 5)
  (h5 : lace_cost_per_meter = 6)
  (h6 : total_spent = 36)
  : (total_spent / lace_cost_per_meter * 100 -
     (num_cuffs * cuff_length + hem_length / 3 + hem_length)) / num_ruffles = 20 := by
  sorry

end lace_per_ruffle_length_l3819_381980


namespace fruits_given_to_jane_l3819_381990

def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def initial_apples : ℕ := 21
def fruits_left : ℕ := 15

def total_initial_fruits : ℕ := initial_plums + initial_guavas + initial_apples

theorem fruits_given_to_jane :
  total_initial_fruits - fruits_left = 40 := by sorry

end fruits_given_to_jane_l3819_381990


namespace tangent_line_at_half_l3819_381984

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x - 1)

theorem tangent_line_at_half (x y : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → 
    |(f (1/2 + h) - f (1/2)) / h - 2| < ε) →
  (y = 2 * x ↔ y - f (1/2) = 2 * (x - 1/2)) :=
sorry

end tangent_line_at_half_l3819_381984


namespace combine_equations_l3819_381911

theorem combine_equations : 
  (15 / 5 = 3) → (24 - 3 = 21) → (24 - 15 / 3 = 21) :=
by
  sorry

end combine_equations_l3819_381911


namespace eleven_divides_reversible_integer_l3819_381967

/-- A 5-digit positive integer with the first three digits the same as its first three digits in reverse order -/
def ReversibleInteger (z : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    z = 10000 * a + 1000 * b + 100 * c + 10 * a + b

theorem eleven_divides_reversible_integer (z : ℕ) (h : ReversibleInteger z) : 
  11 ∣ z :=
sorry

end eleven_divides_reversible_integer_l3819_381967


namespace trig_expression_simplification_l3819_381945

theorem trig_expression_simplification (α β : Real) :
  (Real.sin (α + β))^2 - Real.sin α^2 - Real.sin β^2 /
  ((Real.sin (α + β))^2 - Real.cos α^2 - Real.cos β^2) = 
  -Real.tan α * Real.tan β := by
  sorry

end trig_expression_simplification_l3819_381945


namespace track_length_l3819_381988

/-- The length of a circular track given two cyclists' speeds and meeting time -/
theorem track_length (speed_a speed_b meeting_time : ℝ) : 
  speed_a = 36 →
  speed_b = 72 →
  meeting_time = 19.99840012798976 →
  (speed_b - speed_a) * meeting_time / 60 = 11.999040076793856 :=
by sorry

end track_length_l3819_381988


namespace mod_equivalence_unique_solution_l3819_381942

theorem mod_equivalence_unique_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 4897 [ZMOD 9] ∧ n = 1 := by
  sorry

end mod_equivalence_unique_solution_l3819_381942


namespace find_s_value_l3819_381923

/-- Given a function g(x) = 3x^4 + 2x^3 - x^2 - 4x + s, 
    prove that s = -4 when g(-1) = 0 -/
theorem find_s_value (s : ℝ) : 
  (let g := λ x : ℝ => 3*x^4 + 2*x^3 - x^2 - 4*x + s
   g (-1) = 0) → s = -4 := by
  sorry

end find_s_value_l3819_381923


namespace student_number_problem_l3819_381929

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 102 → x = 120 := by
  sorry

end student_number_problem_l3819_381929


namespace net_increase_theorem_l3819_381927

/-- Represents the different types of vehicles -/
inductive VehicleType
  | Car
  | Motorcycle
  | Van

/-- Represents the different phases of the play -/
inductive PlayPhase
  | BeforeIntermission
  | Intermission
  | AfterIntermission

/-- Initial number of vehicles in the back parking lot -/
def initialVehicles : VehicleType → ℕ
  | VehicleType.Car => 50
  | VehicleType.Motorcycle => 75
  | VehicleType.Van => 25

/-- Arrival rate per hour for each vehicle type during regular play time -/
def arrivalRate : VehicleType → ℕ
  | VehicleType.Car => 70
  | VehicleType.Motorcycle => 120
  | VehicleType.Van => 30

/-- Departure rate per hour for each vehicle type during regular play time -/
def departureRate : VehicleType → ℕ
  | VehicleType.Car => 40
  | VehicleType.Motorcycle => 60
  | VehicleType.Van => 20

/-- Duration of each phase in hours -/
def phaseDuration : PlayPhase → ℚ
  | PlayPhase.BeforeIntermission => 1
  | PlayPhase.Intermission => 1/2
  | PlayPhase.AfterIntermission => 3/2

/-- Net increase rate per hour for each vehicle type during a given phase -/
def netIncreaseRate (v : VehicleType) (p : PlayPhase) : ℚ :=
  match p with
  | PlayPhase.BeforeIntermission => (arrivalRate v - departureRate v : ℚ)
  | PlayPhase.Intermission => (arrivalRate v * 3/2 : ℚ)
  | PlayPhase.AfterIntermission => (arrivalRate v - departureRate v : ℚ)

/-- Total net increase for a given vehicle type -/
def totalNetIncrease (v : VehicleType) : ℚ :=
  (netIncreaseRate v PlayPhase.BeforeIntermission * phaseDuration PlayPhase.BeforeIntermission) +
  (netIncreaseRate v PlayPhase.Intermission * phaseDuration PlayPhase.Intermission) +
  (netIncreaseRate v PlayPhase.AfterIntermission * phaseDuration PlayPhase.AfterIntermission)

/-- Theorem stating the net increase for each vehicle type -/
theorem net_increase_theorem :
  ⌊totalNetIncrease VehicleType.Car⌋ = 127 ∧
  ⌊totalNetIncrease VehicleType.Motorcycle⌋ = 240 ∧
  ⌊totalNetIncrease VehicleType.Van⌋ = 47 := by
  sorry


end net_increase_theorem_l3819_381927


namespace a_plus_b_minus_c_power_2004_l3819_381997

theorem a_plus_b_minus_c_power_2004 (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = a*b + b*c + a*c) 
  (h2 : a = 1) : 
  (a + b - c)^2004 = 1 := by
  sorry

end a_plus_b_minus_c_power_2004_l3819_381997


namespace inequality_proof_l3819_381914

theorem inequality_proof (a b : ℝ) : (a^4 + b^4) * (a^2 + b^2) ≥ (a^3 + b^3)^2 := by
  sorry

end inequality_proof_l3819_381914


namespace survey_results_l3819_381918

/-- Represents the distribution of students in the survey --/
structure StudentDistribution :=
  (high_proactive : ℕ)
  (high_not_proactive : ℕ)
  (average_proactive : ℕ)
  (average_not_proactive : ℕ)

/-- Calculates the chi-square test statistic --/
def chi_square (d : StudentDistribution) : ℚ :=
  let n := d.high_proactive + d.high_not_proactive + d.average_proactive + d.average_not_proactive
  let a := d.high_proactive
  let b := d.high_not_proactive
  let c := d.average_proactive
  let d := d.average_not_proactive
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem about the survey results --/
theorem survey_results (d : StudentDistribution) 
  (h1 : d.high_proactive = 18)
  (h2 : d.high_not_proactive = 7)
  (h3 : d.average_proactive = 6)
  (h4 : d.average_not_proactive = 19) :
  ∃ (X : ℕ → ℚ),
    X 0 = 57/100 ∧ 
    X 1 = 19/50 ∧ 
    X 2 = 1/20 ∧
    (X 0 * 0 + X 1 * 1 + X 2 * 2 = 12/25) ∧
    chi_square d > 10828/1000 := by
  sorry

end survey_results_l3819_381918


namespace symmetric_points_sum_l3819_381941

/-- Two points are symmetric with respect to the y-axis if their y-coordinates are equal
    and their x-coordinates are negatives of each other. -/
def symmetric_y_axis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2 ∧ p.1 = -q.1

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_y_axis (3, a) (b, 2) → a + b = -1 := by
  sorry

end symmetric_points_sum_l3819_381941


namespace geometric_sequence_common_ratio_l3819_381908

/-- A geometric sequence with first term a₁, nth term aₙ, sum Sₙ, and common ratio q. -/
structure GeometricSequence where
  a₁ : ℝ
  aₙ : ℝ
  Sₙ : ℝ
  n : ℕ
  q : ℝ
  geom_seq : a₁ * q^(n-1) = aₙ
  sum_formula : Sₙ = a₁ * (1 - q^n) / (1 - q)

/-- The common ratio of a geometric sequence with a₁ = 2, aₙ = -64, and Sₙ = -42 is -2. -/
theorem geometric_sequence_common_ratio :
  ∀ (seq : GeometricSequence),
    seq.a₁ = 2 →
    seq.aₙ = -64 →
    seq.Sₙ = -42 →
    seq.q = -2 := by
  sorry

end geometric_sequence_common_ratio_l3819_381908


namespace blue_markers_count_l3819_381944

theorem blue_markers_count (total : ℝ) (red : ℝ) (h1 : total = 64.0) (h2 : red = 41.0) :
  total - red = 23.0 := by
  sorry

end blue_markers_count_l3819_381944


namespace unique_k_for_integer_roots_l3819_381978

/-- The polynomial f(x) parameterized by k -/
def f (k : ℤ) (x : ℝ) : ℝ := x^3 - (k-3)*x^2 - 11*x + (4*k-8)

/-- A root of f is an x such that f(x) = 0 -/
def is_root (k : ℤ) (x : ℝ) : Prop := f k x = 0

/-- All roots of f are integers -/
def all_roots_integer (k : ℤ) : Prop :=
  ∀ x : ℝ, is_root k x → ∃ n : ℤ, x = n

/-- The main theorem: k = 5 is the only integer for which all roots of f are integers -/
theorem unique_k_for_integer_roots :
  ∃! k : ℤ, all_roots_integer k ∧ k = 5 := by sorry

end unique_k_for_integer_roots_l3819_381978


namespace complex_fraction_equality_l3819_381965

theorem complex_fraction_equality : 
  (3 / 11) * ((1 + 1 / 3) * (1 + 1 / (2^2 - 1)) * (1 + 1 / (3^2 - 1)) * 
               (1 + 1 / (4^2 - 1)) * (1 + 1 / (5^2 - 1)))^5 = 9600000/2673 := by
  sorry

end complex_fraction_equality_l3819_381965


namespace fifteenth_digit_of_sum_one_eighth_one_sixth_l3819_381985

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- Theorem: The 15th digit after the decimal point in the sum of 1/8 and 1/6 is 6 -/
theorem fifteenth_digit_of_sum_one_eighth_one_sixth : 
  sumDecimalRepresentations (1/8) (1/6) 15 = 6 := by sorry

end fifteenth_digit_of_sum_one_eighth_one_sixth_l3819_381985


namespace quadratic_roots_property_l3819_381920

theorem quadratic_roots_property (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x, x^2 + a*x + 3 = 0) ∧  -- equation has roots
  (x₁ ≠ x₂) ∧  -- roots are distinct
  (x₁^2 + a*x₁ + 3 = 0) ∧  -- x₁ is a root
  (x₂^2 + a*x₂ + 3 = 0) ∧  -- x₂ is a root
  (x₁^3 - 39/x₂ = x₂^3 - 39/x₁)  -- given condition
  → 
  a = 4 ∨ a = -4 :=
by sorry

end quadratic_roots_property_l3819_381920


namespace parallelogram_base_l3819_381955

theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 120 →
  height = 10 →
  area = base * height →
  base = 12 := by
  sorry

end parallelogram_base_l3819_381955


namespace f_at_point_four_l3819_381957

def f (x : ℝ) : ℝ := 3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1

theorem f_at_point_four : f 0.4 = 5.885248 := by
  sorry

end f_at_point_four_l3819_381957


namespace largest_side_of_special_rectangle_l3819_381907

/-- A rectangle with specific properties --/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 120
  area_eq : length * width = 1920

/-- The largest side of a special rectangle is 101 --/
theorem largest_side_of_special_rectangle (r : SpecialRectangle) : 
  max r.length r.width = 101 := by
  sorry

end largest_side_of_special_rectangle_l3819_381907


namespace section_point_representation_l3819_381982

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points
variable (C D Q : V)

-- Define the condition that Q is on line segment CD with ratio 4:1
def is_section_point (C D Q : V) : Prop :=
  ∃ (t : ℝ), t ∈ Set.Icc (0 : ℝ) (1 : ℝ) ∧ Q = (1 - t) • C + t • D ∧ (1 - t) / t = 4

-- The theorem
theorem section_point_representation (h : is_section_point C D Q) :
  Q = (1/5 : ℝ) • C + (4/5 : ℝ) • D :=
sorry

end section_point_representation_l3819_381982


namespace sunny_subsets_l3819_381961

theorem sunny_subsets (m n : ℕ) (S : Finset ℕ) (h1 : n ≥ m) (h2 : m ≥ 2) (h3 : Finset.card S = n) :
  ∃ T : Finset (Finset ℕ), (∀ X ∈ T, X ⊆ S ∧ m ∣ (Finset.sum X id)) ∧ Finset.card T ≥ 2^(n - m + 1) :=
sorry

end sunny_subsets_l3819_381961


namespace triangle_division_ratio_l3819_381970

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the sum of squares of distances from vertices to division points -/
def S (t : Triangle) (n : ℕ) : ℝ := sorry

/-- Theorem: The ratio of S to the sum of squared side lengths is a specific rational function of n -/
theorem triangle_division_ratio (t : Triangle) (n : ℕ) (h : n > 0) :
  S t n / (t.a^2 + t.b^2 + t.c^2) = (n - 1) * (5 * n - 1) / (6 * n) := by
  sorry

end triangle_division_ratio_l3819_381970


namespace fixed_point_exponential_function_l3819_381958

/-- The function f(x) = a^(2-x) + 2 always passes through the point (2, 3) for all a > 0 and a ≠ 1 -/
theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2 - x) + 2
  f 2 = 3 := by sorry

end fixed_point_exponential_function_l3819_381958


namespace value_of_s_l3819_381921

theorem value_of_s (a b c w s p : ℕ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ w ≠ 0 ∧ s ≠ 0 ∧ p ≠ 0)
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ w ∧ a ≠ s ∧ a ≠ p)
  (h3 : b ≠ c ∧ b ≠ w ∧ b ≠ s ∧ b ≠ p)
  (h4 : c ≠ w ∧ c ≠ s ∧ c ≠ p)
  (h5 : w ≠ s ∧ w ≠ p)
  (h6 : s ≠ p)
  (eq1 : a + b = w)
  (eq2 : w + c = s)
  (eq3 : s + a = p)
  (eq4 : b + c + p = 16) : 
  s = 8 := by
sorry

end value_of_s_l3819_381921


namespace equal_sampling_probability_l3819_381994

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Define a function to represent the probability of an individual being sampled
def samplingProbability (method : SamplingMethod) (individual : ℕ) : ℝ :=
  sorry

-- State the theorem
theorem equal_sampling_probability (method : SamplingMethod) (individual1 individual2 : ℕ) :
  samplingProbability method individual1 = samplingProbability method individual2 :=
sorry

end equal_sampling_probability_l3819_381994


namespace range_of_a_for_unique_solution_l3819_381947

/-- The range of 'a' for which the equation lg(x-1) + lg(3-x) = lg(x-a) has exactly one solution for x, where 1 < x < 3 -/
theorem range_of_a_for_unique_solution : 
  ∀ a : ℝ, (∃! x : ℝ, 1 < x ∧ x < 3 ∧ Real.log (x - 1) + Real.log (3 - x) = Real.log (x - a)) ↔ 
  (a ≥ 3/4 ∧ a < 3) := by
sorry

end range_of_a_for_unique_solution_l3819_381947


namespace f_minimum_F_monotonicity_intersection_property_l3819_381932

noncomputable section

def f (x : ℝ) : ℝ := x * (1 + Real.log x)

def f' (x : ℝ) : ℝ := Real.log x + 2

def F (a : ℝ) (x : ℝ) : ℝ := a * x^2 + f' x

theorem f_minimum (x : ℝ) (hx : x > 0) :
  f x ≥ -1 / Real.exp 2 ∧ 
  f (1 / Real.exp 2) = -1 / Real.exp 2 :=
sorry

theorem F_monotonicity (a : ℝ) (x : ℝ) (hx : x > 0) :
  (a ≥ 0 → ∀ y > 0, x < y → F a x < F a y) ∧
  (a < 0 → ∃ c > 0, (∀ y, 0 < y ∧ y < c → F a x < F a y) ∧
                    (∀ y > c, F a y < F a x)) :=
sorry

theorem intersection_property (k x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  k = (f' x₂ - f' x₁) / (x₂ - x₁) →
  x₁ < 1 / k ∧ 1 / k < x₂ :=
sorry

end f_minimum_F_monotonicity_intersection_property_l3819_381932


namespace quadratic_roots_range_l3819_381930

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2 * Real.sqrt a * x₁ + 2 * a - 1 = 0 ∧ 
                 x₂^2 - 2 * Real.sqrt a * x₂ + 2 * a - 1 = 0) →
  0 ≤ a ∧ a ≤ 1 := by
sorry

end quadratic_roots_range_l3819_381930


namespace special_trapezoid_smaller_side_l3819_381987

/-- A trapezoid with specific angle and side length properties -/
structure SpecialTrapezoid where
  /-- The angle at one end of the larger base -/
  angle1 : ℝ
  /-- The angle at the other end of the larger base -/
  angle2 : ℝ
  /-- The length of the larger lateral side -/
  larger_side : ℝ
  /-- The length of the smaller lateral side -/
  smaller_side : ℝ
  /-- Constraint: angle1 is 60 degrees -/
  angle1_is_60 : angle1 = 60
  /-- Constraint: angle2 is 30 degrees -/
  angle2_is_30 : angle2 = 30
  /-- Constraint: larger_side is 6√3 -/
  larger_side_is_6root3 : larger_side = 6 * Real.sqrt 3

/-- Theorem: In a SpecialTrapezoid, the smaller lateral side has length 6 -/
theorem special_trapezoid_smaller_side (t : SpecialTrapezoid) : t.smaller_side = 6 := by
  sorry

end special_trapezoid_smaller_side_l3819_381987


namespace f_simplification_l3819_381995

def f (x : ℝ) : ℝ := (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1

theorem f_simplification (x : ℝ) : f x = 32 * x^5 := by
  sorry

end f_simplification_l3819_381995


namespace paulson_spending_percentage_l3819_381916

theorem paulson_spending_percentage 
  (income : ℝ) 
  (expenditure : ℝ) 
  (savings : ℝ) 
  (h1 : expenditure + savings = income) 
  (h2 : 1.2 * income - 1.1 * expenditure = 1.5 * savings) : 
  expenditure = 0.75 * income :=
sorry

end paulson_spending_percentage_l3819_381916


namespace football_team_right_handed_players_l3819_381943

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h_total : total_players = 70)
  (h_throwers : throwers = 49)
  (h_throwers_right_handed : throwers ≤ total_players)
  (h_non_throwers_division : (total_players - throwers) % 3 = 0)
  : throwers + ((total_players - throwers) * 2 / 3) = 63 := by
  sorry

end football_team_right_handed_players_l3819_381943


namespace new_person_weight_is_87_l3819_381992

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_group_size : ℕ) (average_increase : ℝ) (replaced_person_weight : ℝ) : ℝ :=
  replaced_person_weight + initial_group_size * average_increase

/-- Theorem stating that the weight of the new person is 87 kg -/
theorem new_person_weight_is_87 :
  new_person_weight 8 4 55 = 87 := by
  sorry

#eval new_person_weight 8 4 55

end new_person_weight_is_87_l3819_381992
