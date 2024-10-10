import Mathlib

namespace students_taking_one_subject_l3987_398773

/-- Given information about students taking geometry and history classes,
    prove that the number of students taking either geometry or history
    but not both is 35. -/
theorem students_taking_one_subject (total_geometry : ℕ)
                                    (both_subjects : ℕ)
                                    (history_only : ℕ)
                                    (h1 : total_geometry = 40)
                                    (h2 : both_subjects = 20)
                                    (h3 : history_only = 15) :
  (total_geometry - both_subjects) + history_only = 35 := by
  sorry


end students_taking_one_subject_l3987_398773


namespace additional_racks_needed_additional_racks_needed_is_one_l3987_398797

-- Define the given constants
def flour_per_bag : ℕ := 12
def bags_of_flour : ℕ := 5
def cups_per_pound : ℕ := 3
def pounds_per_rack : ℕ := 5
def owned_racks : ℕ := 3

-- Define the theorem
theorem additional_racks_needed : ℕ :=
  let total_flour : ℕ := flour_per_bag * bags_of_flour
  let total_pounds : ℕ := total_flour / cups_per_pound
  let capacity : ℕ := owned_racks * pounds_per_rack
  let remaining : ℕ := total_pounds - capacity
  (remaining + pounds_per_rack - 1) / pounds_per_rack

-- Proof
theorem additional_racks_needed_is_one : additional_racks_needed = 1 := by
  sorry

end additional_racks_needed_additional_racks_needed_is_one_l3987_398797


namespace simplify_expression_l3987_398791

theorem simplify_expression (y : ℝ) : 3 * y - 5 * y^2 + 12 - (7 - 3 * y + 5 * y^2) = -10 * y^2 + 6 * y + 5 := by
  sorry

end simplify_expression_l3987_398791


namespace contrapositive_odd_product_l3987_398748

theorem contrapositive_odd_product (a b : ℤ) : 
  (((a % 2 = 1 ∧ b % 2 = 1) → (a * b) % 2 = 1) ↔ 
   ((a * b) % 2 ≠ 1 → (a % 2 ≠ 1 ∨ b % 2 ≠ 1))) ∧
  (∀ a b : ℤ, (a * b) % 2 ≠ 1 → (a % 2 ≠ 1 ∨ b % 2 ≠ 1)) := by
  sorry

end contrapositive_odd_product_l3987_398748


namespace arithmetic_sum_formula_main_theorem_l3987_398714

-- Define the sum of an arithmetic sequence from 1 to n
def arithmeticSum (n : ℕ) : ℕ := n * (1 + n) / 2

-- Define the sum of the odd numbers from 1 to 69
def oddSum : ℕ := (1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21 + 23 + 25 + 27 + 29 + 31 + 33 + 35 + 37 + 39 + 41 + 43 + 45 + 47 + 49 + 51 + 53 + 55 + 57 + 59 + 61 + 63 + 65 + 67 + 69)

-- Theorem stating the correctness of the arithmetic sum formula
theorem arithmetic_sum_formula (n : ℕ) : 
  (List.range n).sum = arithmeticSum n :=
by sorry

-- Given condition
axiom odd_sum_condition : 3 * oddSum = 3675

-- Main theorem to prove
theorem main_theorem (n : ℕ) :
  (List.range n).sum = n * (1 + n) / 2 :=
by sorry

end arithmetic_sum_formula_main_theorem_l3987_398714


namespace parabola_directrix_a_value_l3987_398760

/-- A parabola with equation y² = ax and directrix x = 1 has a = -4 -/
theorem parabola_directrix_a_value :
  ∀ (a : ℝ),
  (∀ (x y : ℝ), y^2 = a*x → (∃ (p : ℝ), x = -p ∧ x = 1)) →
  a = -4 :=
by sorry

end parabola_directrix_a_value_l3987_398760


namespace problem_solution_l3987_398721

theorem problem_solution (x y : ℝ) : 
  x = 201 → x^3 * y - 2 * x^2 * y + x * y = 804000 → y = 1/10 := by
  sorry

end problem_solution_l3987_398721


namespace amy_video_files_l3987_398757

/-- Proves that Amy had 21 video files initially -/
theorem amy_video_files :
  ∀ (initial_music_files deleted_files remaining_files : ℕ),
    initial_music_files = 4 →
    deleted_files = 23 →
    remaining_files = 2 →
    initial_music_files + (deleted_files + remaining_files) - initial_music_files = 21 :=
by
  sorry

end amy_video_files_l3987_398757


namespace largest_pot_cost_l3987_398750

/-- The cost of the largest pot in a set of 6 pots with specific pricing rules -/
theorem largest_pot_cost (total_cost : ℚ) (num_pots : ℕ) (price_diff : ℚ) :
  total_cost = 33/4 ∧ num_pots = 6 ∧ price_diff = 1/10 →
  ∃ (smallest_cost : ℚ),
    smallest_cost > 0 ∧
    (smallest_cost + (num_pots - 1) * price_diff) = 13/8 := by
  sorry

end largest_pot_cost_l3987_398750


namespace inscribed_cylinder_radius_l3987_398785

/-- Represents a right circular cone -/
structure Cone :=
  (diameter : ℝ)
  (altitude : ℝ)

/-- Represents a right circular cylinder -/
structure Cylinder :=
  (radius : ℝ)

/-- 
Theorem: The radius of a cylinder inscribed in a cone
Given:
  - The cylinder's diameter is equal to its height
  - The cone has a diameter of 12 and an altitude of 15
  - The axes of the cylinder and cone coincide
Prove: The radius of the cylinder is 10/3
-/
theorem inscribed_cylinder_radius (cone : Cone) (cyl : Cylinder) :
  cone.diameter = 12 →
  cone.altitude = 15 →
  cyl.radius * 2 = cyl.radius * 2 →  -- cylinder's diameter equals its height
  cyl.radius = 10 / 3 := by
  sorry

end inscribed_cylinder_radius_l3987_398785


namespace candy_distribution_l3987_398742

/-- Given 200 candies distributed among A, B, and C, where A has more than twice as many candies as B,
    and B has more than three times as many candies as C, prove that the minimum number of candies A
    can have is 121, and the maximum number of candies C can have is 19. -/
theorem candy_distribution (a b c : ℕ) : 
  a + b + c = 200 →
  a > 2 * b →
  b > 3 * c →
  (∀ a' b' c' : ℕ, a' + b' + c' = 200 → a' > 2 * b' → b' > 3 * c' → a' ≥ a) →
  (∀ a' b' c' : ℕ, a' + b' + c' = 200 → a' > 2 * b' → b' > 3 * c' → c' ≤ c) →
  a = 121 ∧ c = 19 := by
  sorry

end candy_distribution_l3987_398742


namespace folding_theorem_l3987_398759

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a line segment -/
structure Segment where
  length : ℝ

/-- Represents the folding problem -/
def FoldingProblem (rect : Rectangle) : Prop :=
  ∃ (CC' EF : Segment),
    rect.width = 240 ∧
    rect.height = 288 ∧
    CC'.length = 312 ∧
    EF.length = 260

/-- The main theorem -/
theorem folding_theorem (rect : Rectangle) :
  FoldingProblem rect :=
sorry

end folding_theorem_l3987_398759


namespace f_intersects_y_axis_at_zero_one_l3987_398789

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- Theorem statement
theorem f_intersects_y_axis_at_zero_one : f 0 = 1 := by
  sorry

end f_intersects_y_axis_at_zero_one_l3987_398789


namespace only_lottery_is_random_l3987_398775

-- Define the events
def event_A := "No moisture, seed germination"
def event_B := "At least 2 people out of 367 have the same birthday"
def event_C := "Melting of ice at -1°C under standard pressure"
def event_D := "Xiao Ying bought a lottery ticket and won a 5 million prize"

-- Define a predicate for random events
def is_random_event (e : String) : Prop := sorry

-- Theorem stating that only event_D is a random event
theorem only_lottery_is_random :
  ¬(is_random_event event_A) ∧
  ¬(is_random_event event_B) ∧
  ¬(is_random_event event_C) ∧
  is_random_event event_D :=
by sorry

end only_lottery_is_random_l3987_398775


namespace car_travel_time_difference_l3987_398703

/-- Proves that the time difference between two cars traveling 150 miles is 2 hours,
    given their speeds differ by 10 mph and one car's speed is 22.83882181415011 mph. -/
theorem car_travel_time_difference 
  (distance : ℝ) 
  (speed_R : ℝ) 
  (speed_P : ℝ) : 
  distance = 150 →
  speed_R = 22.83882181415011 →
  speed_P = speed_R + 10 →
  distance / speed_R - distance / speed_P = 2 := by
sorry

end car_travel_time_difference_l3987_398703


namespace compound_interest_existence_l3987_398711

/-- Proves the existence of a principal amount and interest rate satisfying the compound interest conditions --/
theorem compound_interest_existence : ∃ (P r : ℝ), 
  P * (1 + r)^2 = 8840 ∧ P * (1 + r)^3 = 9261 := by
  sorry

end compound_interest_existence_l3987_398711


namespace solution_to_equation_l3987_398733

theorem solution_to_equation : ∃ x : ℚ, (1/3 - 1/2) * x = 1 ∧ x = -6 := by sorry

end solution_to_equation_l3987_398733


namespace smallest_cuboid_face_area_l3987_398723

/-- Given a cuboid with integer volume and face areas 7, 27, and L, 
    prove that the smallest possible integer value for L is 21 -/
theorem smallest_cuboid_face_area (a b c : ℕ+) (L : ℕ) : 
  (a * b : ℕ) = 7 →
  (a * c : ℕ) = 27 →
  (b * c : ℕ) = L →
  (∃ (v : ℕ), v = a * b * c) →
  L ≥ 21 ∧ 
  (∀ L' : ℕ, L' ≥ 21 → ∃ (a' b' c' : ℕ+), 
    (a' * b' : ℕ) = 7 ∧ 
    (a' * c' : ℕ) = 27 ∧ 
    (b' * c' : ℕ) = L') :=
by sorry

#check smallest_cuboid_face_area

end smallest_cuboid_face_area_l3987_398723


namespace ellipse_intersection_theorem_l3987_398790

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the focal length -/
def focal_length (c : ℝ) : Prop :=
  c = 2

/-- Definition of a point on the ellipse -/
def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse 2 (-Real.sqrt 2) a b

/-- Definition of the line intersecting the ellipse -/
def intersecting_line (x y m : ℝ) : Prop :=
  y = x + m

/-- Definition of the circle where the midpoint lies -/
def midpoint_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Main theorem -/
theorem ellipse_intersection_theorem (a b c m : ℝ) : 
  a > b ∧ b > 0 ∧
  focal_length c ∧
  point_on_ellipse a b →
  (∀ x y, ellipse x y a b ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    ellipse A.1 A.2 a b ∧
    ellipse B.1 B.2 a b ∧
    intersecting_line A.1 A.2 m ∧
    intersecting_line B.1 B.2 m ∧
    midpoint_circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) →
    m = 3 * Real.sqrt 5 / 5 ∨ m = -3 * Real.sqrt 5 / 5) :=
sorry

end ellipse_intersection_theorem_l3987_398790


namespace expression_simplification_l3987_398772

variable (a b x : ℝ)

theorem expression_simplification (h : x ≥ a) :
  (Real.sqrt (b^2 + a^2 + x^2) - (x^3 - a^3) / Real.sqrt (b^2 + a^2 + x^2)) / (b^2 + a^2 + x^2) = 
  (b^2 + a^2 + a^3) / (b^2 + a^2 + x^2)^(3/2) := by sorry

end expression_simplification_l3987_398772


namespace complex_abs_sum_l3987_398718

theorem complex_abs_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 8*I) = Real.sqrt 34 + Real.sqrt 73 := by
  sorry

end complex_abs_sum_l3987_398718


namespace total_card_value_is_244_l3987_398756

def jenny_initial_cards : ℕ := 6
def jenny_rare_percentage : ℚ := 1/2
def orlando_extra_cards : ℕ := 2
def orlando_rare_percentage : ℚ := 2/5
def richard_card_multiplier : ℕ := 3
def richard_rare_percentage : ℚ := 1/4
def jenny_additional_cards : ℕ := 4
def holographic_card_value : ℕ := 15
def first_edition_card_value : ℕ := 8
def rare_card_value : ℕ := 10
def non_rare_card_value : ℕ := 3

def total_card_value : ℕ := sorry

theorem total_card_value_is_244 : total_card_value = 244 := by sorry

end total_card_value_is_244_l3987_398756


namespace a_minus_b_value_l3987_398710

theorem a_minus_b_value (a b : ℝ) 
  (h1 : a^2 * b - a * b^2 = -6) 
  (h2 : a * b = 3) : 
  a - b = -2 := by
sorry

end a_minus_b_value_l3987_398710


namespace sum_interior_angles_regular_polygon_l3987_398728

/-- 
Given a regular polygon where each exterior angle measures 40°, 
the sum of its interior angles is 1260°.
-/
theorem sum_interior_angles_regular_polygon (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 40) : 
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end sum_interior_angles_regular_polygon_l3987_398728


namespace tree_height_difference_l3987_398717

theorem tree_height_difference :
  let pine_height : ℚ := 53/4
  let maple_height : ℚ := 41/2
  maple_height - pine_height = 29/4 := by sorry

end tree_height_difference_l3987_398717


namespace train_length_calculation_l3987_398737

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length_calculation (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 → platform_length = 280 → crossing_time = 26 →
  (train_speed * 1000 / 3600) * crossing_time - platform_length = 240 := by
  sorry

#check train_length_calculation

end train_length_calculation_l3987_398737


namespace polynomial_existence_l3987_398761

theorem polynomial_existence : ∃ (f : ℝ → ℝ), 
  (∃ (a b c d e g h : ℝ), ∀ x, f x = a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + g*x + h) ∧ 
  (∀ x, f (Real.sin x) + f (Real.cos x) = 1) := by
  sorry

end polynomial_existence_l3987_398761


namespace smallest_cube_ending_368_l3987_398754

theorem smallest_cube_ending_368 : 
  ∃ (n : ℕ), n > 0 ∧ n^3 ≡ 368 [MOD 1000] ∧ ∀ (m : ℕ), m > 0 ∧ m^3 ≡ 368 [MOD 1000] → n ≤ m :=
by sorry

end smallest_cube_ending_368_l3987_398754


namespace compound_oxygen_atoms_l3987_398752

/-- Proves that a compound with 6 C atoms, 8 H atoms, and a molecular weight of 192
    contains 7 O atoms, given the atomic weights of C, H, and O. -/
theorem compound_oxygen_atoms 
  (atomic_weight_C : ℝ) 
  (atomic_weight_H : ℝ) 
  (atomic_weight_O : ℝ) 
  (h1 : atomic_weight_C = 12.01)
  (h2 : atomic_weight_H = 1.008)
  (h3 : atomic_weight_O = 16.00)
  (h4 : (6 * atomic_weight_C + 8 * atomic_weight_H + 7 * atomic_weight_O) = 192) :
  ∃ n : ℕ, n = 7 ∧ (6 * atomic_weight_C + 8 * atomic_weight_H + n * atomic_weight_O) = 192 :=
by
  sorry


end compound_oxygen_atoms_l3987_398752


namespace direction_vector_l3987_398709

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = (4 * x - 6) / 5

-- Define the parameterization
def parameterization (t : ℝ) (d : ℝ × ℝ) : ℝ × ℝ :=
  (4 + t * d.1, 2 + t * d.2)

-- Define the distance condition
def distance_condition (x y : ℝ) (t : ℝ) : Prop :=
  x ≥ 4 → (x - 4)^2 + (y - 2)^2 = t^2

-- Theorem statement
theorem direction_vector :
  ∃ (d : ℝ × ℝ),
    (∀ x y t, line_eq x y →
      (x, y) = parameterization t d →
      distance_condition x y t) →
    d = (5 / Real.sqrt 41, 4 / Real.sqrt 41) :=
sorry

end direction_vector_l3987_398709


namespace modular_inverse_of_7_mod_800_l3987_398788

theorem modular_inverse_of_7_mod_800 :
  let a : ℕ := 7
  let m : ℕ := 800
  let inv : ℕ := 343
  (Nat.gcd a m = 1) →
  (inv < m) →
  (a * inv) % m = 1 →
  ∃ x : ℕ, x < m ∧ (a * x) % m = 1 ∧ x = inv :=
by sorry

end modular_inverse_of_7_mod_800_l3987_398788


namespace four_half_planes_theorem_l3987_398738

-- Define a half-plane
def HalfPlane : Type := ℝ × ℝ → Prop

-- Define a set of four half-planes
def FourHalfPlanes : Type := Fin 4 → HalfPlane

-- Define the property of covering the entire plane
def CoversPlane (planes : FourHalfPlanes) : Prop :=
  ∀ (x y : ℝ), ∃ (i : Fin 4), planes i (x, y)

-- Define the property of a subset of three half-planes covering the entire plane
def ThreeCoversPlane (planes : FourHalfPlanes) : Prop :=
  ∃ (i j k : Fin 4) (h : i ≠ j ∧ j ≠ k ∧ i ≠ k),
    ∀ (x y : ℝ), planes i (x, y) ∨ planes j (x, y) ∨ planes k (x, y)

-- The theorem to be proved
theorem four_half_planes_theorem (planes : FourHalfPlanes) :
  CoversPlane planes → ThreeCoversPlane planes :=
by
  sorry

end four_half_planes_theorem_l3987_398738


namespace multiply_fractions_result_l3987_398798

theorem multiply_fractions_result : (77 / 4) * (5 / 2) = 48 + 1 / 8 := by
  sorry

end multiply_fractions_result_l3987_398798


namespace even_function_c_value_f_increasing_on_interval_l3987_398735

def f (x : ℝ) : ℝ := x^2 + 4*x + 3

def g (x c : ℝ) : ℝ := f x + c*x

theorem even_function_c_value :
  (∀ x, g x (-4) = g (-x) (-4)) ∧ 
  (∀ c, (∀ x, g x c = g (-x) c) → c = -4) :=
sorry

theorem f_increasing_on_interval :
  ∀ x₁ x₂, -2 ≤ x₁ → x₁ < x₂ → f x₁ < f x₂ :=
sorry

end even_function_c_value_f_increasing_on_interval_l3987_398735


namespace composite_shape_area_l3987_398700

/-- The area of a composite shape consisting of three rectangles --/
def composite_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height

/-- Theorem stating that the area of the given composite shape is 77 square units --/
theorem composite_shape_area : composite_area 10 4 4 7 3 3 = 77 := by
  sorry

end composite_shape_area_l3987_398700


namespace paraboloid_surface_area_l3987_398794

/-- The paraboloid of revolution --/
def paraboloid (x y z : ℝ) : Prop := 3 * y = x^2 + z^2

/-- The bounding plane --/
def bounding_plane (y : ℝ) : Prop := y = 6

/-- The first octant --/
def first_octant (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

/-- The surface area of the part of the paraboloid --/
noncomputable def surface_area : ℝ := sorry

/-- The theorem stating the surface area of the specified part of the paraboloid --/
theorem paraboloid_surface_area :
  surface_area = 39 * Real.pi / 4 := by sorry

end paraboloid_surface_area_l3987_398794


namespace complex_number_quadrant_l3987_398795

theorem complex_number_quadrant : 
  let z : ℂ := (2 - 3*I) / (I^3)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end complex_number_quadrant_l3987_398795


namespace sqrt_expression_equality_l3987_398720

theorem sqrt_expression_equality (t : ℝ) : 
  Real.sqrt (9 * t^4 + 4 * t^2 + 4 * t) = |t| * Real.sqrt ((3 * t^2 + 2 * t) * (3 * t^2 + 2 * t + 2)) := by
  sorry

end sqrt_expression_equality_l3987_398720


namespace symmetry_properties_l3987_398747

def Point := ℝ × ℝ

def symmetricAboutXAxis (p : Point) : Point :=
  (p.1, -p.2)

def symmetricAboutYAxis (p : Point) : Point :=
  (-p.1, p.2)

theorem symmetry_properties (x y : ℝ) :
  let A : Point := (x, y)
  (symmetricAboutXAxis A = (x, -y)) ∧
  (symmetricAboutYAxis A = (-x, y)) := by
  sorry

end symmetry_properties_l3987_398747


namespace largest_of_five_consecutive_even_integers_l3987_398729

/-- The sum of the first n positive even integers -/
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- The sum of five consecutive even integers starting from k -/
def sumFiveConsecutiveEvenIntegers (k : ℕ) : ℕ := 5 * k - 10

theorem largest_of_five_consecutive_even_integers :
  ∃ k : ℕ, 
    sumFirstNEvenIntegers 30 = sumFiveConsecutiveEvenIntegers k ∧ 
    k = 190 := by
  sorry

#eval sumFirstNEvenIntegers 30  -- Should output 930
#eval sumFiveConsecutiveEvenIntegers 190  -- Should also output 930

end largest_of_five_consecutive_even_integers_l3987_398729


namespace honest_person_different_answers_possible_l3987_398746

-- Define a person who always tells the truth
structure HonestPerson where
  name : String
  always_truthful : Bool

-- Define a question with its context
structure Question where
  text : String
  context : String

-- Define an answer
structure Answer where
  text : String

-- Define a function to represent a person answering a question
def answer (person : HonestPerson) (q : Question) : Answer :=
  sorry

-- Theorem: It's possible for an honest person to give different answers to the same question asked twice
theorem honest_person_different_answers_possible 
  (person : HonestPerson) 
  (q : Question) 
  (q_repeated : Question) 
  (different_context : q.context ≠ q_repeated.context) :
  ∃ (a1 a2 : Answer), 
    person.always_truthful = true ∧ 
    q.text = q_repeated.text ∧ 
    answer person q = a1 ∧ 
    answer person q_repeated = a2 ∧ 
    a1 ≠ a2 :=
  sorry

end honest_person_different_answers_possible_l3987_398746


namespace smaller_solution_of_quadratic_l3987_398727

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 - 14*x + 45 = 0 → ∃ y : ℝ, y^2 - 14*y + 45 = 0 ∧ y ≠ x ∧ (∀ z : ℝ, z^2 - 14*z + 45 = 0 → z = x ∨ z = y) ∧ min x y = 5 :=
by sorry

end smaller_solution_of_quadratic_l3987_398727


namespace steelyard_scale_construction_l3987_398777

/-- Represents a steelyard (balance) --/
structure Steelyard where
  l : ℝ  -- length of the steelyard
  Q : ℝ  -- weight of the steelyard
  a : ℝ  -- distance where 1 kg balances the steelyard

/-- Theorem for the steelyard scale construction --/
theorem steelyard_scale_construction (S : Steelyard) (p x : ℝ) 
  (h1 : S.l > 0)
  (h2 : S.Q > 0)
  (h3 : S.a > 0)
  (h4 : S.a < S.l)
  (h5 : x > 0)
  (h6 : x < S.l) :
  p * x / S.a = (S.l - x) / (S.l - S.a) :=
sorry

end steelyard_scale_construction_l3987_398777


namespace tennis_cost_calculation_l3987_398782

/-- Represents the cost of tennis equipment under different purchasing options -/
def TennisCost (x : ℕ) : Prop :=
  let racketPrice : ℕ := 200
  let ballPrice : ℕ := 40
  let racketQuantity : ℕ := 20
  let option1Cost : ℕ := racketPrice * racketQuantity + ballPrice * (x - racketQuantity)
  let option2Cost : ℕ := (racketPrice * racketQuantity + ballPrice * x) * 9 / 10
  x > 20 ∧ option1Cost = 40 * x + 3200 ∧ option2Cost = 3600 + 36 * x

theorem tennis_cost_calculation (x : ℕ) : TennisCost x := by
  sorry

end tennis_cost_calculation_l3987_398782


namespace circle_c_equation_l3987_398726

/-- A circle C satisfying the given conditions -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  tangent_y_axis : center.1 = radius
  chord_length : 4 * radius ^ 2 - center.1 ^ 2 = 12
  center_on_line : center.2 = 1/2 * center.1

/-- The equation of the circle C -/
def circle_equation (c : CircleC) (x y : ℝ) : Prop :=
  (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2

/-- Theorem stating that the circle C has the equation (x-2)² + (y-1)² = 4 -/
theorem circle_c_equation (c : CircleC) :
  ∀ x y, circle_equation c x y ↔ (x - 2) ^ 2 + (y - 1) ^ 2 = 4 :=
by sorry

end circle_c_equation_l3987_398726


namespace distinct_two_mark_grids_l3987_398749

/-- Represents a 4x4 grid --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Represents a rotation of the grid --/
inductive Rotation
| r0 | r90 | r180 | r270

/-- Applies a rotation to a grid --/
def applyRotation (r : Rotation) (g : Grid) : Grid :=
  sorry

/-- Checks if two grids are equivalent under rotation --/
def areEquivalent (g1 g2 : Grid) : Bool :=
  sorry

/-- Counts the number of marked cells in a grid --/
def countMarked (g : Grid) : Nat :=
  sorry

/-- Generates all possible grids with exactly two marked cells --/
def allGridsWithTwoMarked : List Grid :=
  sorry

/-- Counts the number of distinct grids under rotation --/
def countDistinctGrids (grids : List Grid) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem distinct_two_mark_grids :
  countDistinctGrids allGridsWithTwoMarked = 32 :=
sorry

end distinct_two_mark_grids_l3987_398749


namespace five_hour_charge_l3987_398740

/-- Represents the charge structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyCharges where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  moreExpensiveFirst : firstHourCharge = additionalHourCharge + 35
  twoHourTotal : firstHourCharge + additionalHourCharge = 161

/-- Calculates the total charge for a given number of therapy hours. -/
def totalCharge (charges : TherapyCharges) (hours : ℕ) : ℕ :=
  charges.firstHourCharge + (hours - 1) * charges.additionalHourCharge

/-- Theorem stating that the total charge for 5 hours of therapy is $350. -/
theorem five_hour_charge (charges : TherapyCharges) : totalCharge charges 5 = 350 := by
  sorry

end five_hour_charge_l3987_398740


namespace hollow_cube_5x5x5_l3987_398722

/-- The number of cubes needed for a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ :=
  6 * (n^2 - (n-2)^2) - 12 * (n-2)

/-- Theorem: A hollow cube with outer dimensions 5 * 5 * 5 requires 60 cubes -/
theorem hollow_cube_5x5x5 : hollow_cube_cubes 5 = 60 := by
  sorry

end hollow_cube_5x5x5_l3987_398722


namespace total_books_l3987_398734

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
sorry

end total_books_l3987_398734


namespace distance_AB_bounds_l3987_398781

/-- Given six points in space with specific distance relationships, 
    prove that the distance between two of the points lies within a certain range. -/
theorem distance_AB_bounds 
  (A B C D E F : EuclideanSpace ℝ (Fin 3)) 
  (h1 : dist A C = 10 ∧ dist A D = 10 ∧ dist B E = 10 ∧ dist B F = 10)
  (h2 : dist A E = 12 ∧ dist A F = 12 ∧ dist B C = 12 ∧ dist B D = 12)
  (h3 : dist C D = 11 ∧ dist E F = 11)
  (h4 : dist C E = 5 ∧ dist D F = 5) : 
  8.8 < dist A B ∧ dist A B < 19.2 := by
  sorry


end distance_AB_bounds_l3987_398781


namespace quadratic_inequality_range_l3987_398796

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + (a - 1) * x + 1 > 0) ↔ a ∈ Set.Icc 1 5 \ {5} :=
sorry

end quadratic_inequality_range_l3987_398796


namespace square_sum_difference_specific_square_sum_difference_l3987_398767

theorem square_sum_difference (n : ℕ) : 
  (2*n + 1)^2 - (2*n - 1)^2 + (2*n - 1)^2 - (2*n - 3)^2 + 
  (2*n - 3)^2 - (2*n - 5)^2 + (2*n - 5)^2 - (2*n - 7)^2 + 
  (2*n - 7)^2 - (2*n - 9)^2 + (2*n - 9)^2 - (2*n - 11)^2 = 24 * n :=
by
  sorry

theorem specific_square_sum_difference : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by
  sorry

end square_sum_difference_specific_square_sum_difference_l3987_398767


namespace odd_composite_sum_representation_l3987_398732

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

/-- An odd number can be represented as the sum of two composite numbers -/
def CanBeRepresentedAsCompositeSum (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem odd_composite_sum_representation :
  ∀ n : ℕ, n ≥ 13 → Odd n → CanBeRepresentedAsCompositeSum n := by
  sorry

#check odd_composite_sum_representation

end odd_composite_sum_representation_l3987_398732


namespace seven_boys_handshakes_l3987_398716

/-- The number of handshakes between n boys, where each boy shakes hands exactly once with each of the others -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of handshakes between 7 boys is 21 -/
theorem seven_boys_handshakes : num_handshakes 7 = 21 := by
  sorry

end seven_boys_handshakes_l3987_398716


namespace sum_real_imag_parts_of_complex_fraction_l3987_398770

theorem sum_real_imag_parts_of_complex_fraction : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  (z.re + z.im) = 1 := by
  sorry

end sum_real_imag_parts_of_complex_fraction_l3987_398770


namespace complex_equality_l3987_398725

theorem complex_equality (ω : ℂ) :
  Complex.abs (ω - 2) = Complex.abs (ω - 2 * Complex.I) →
  ω.re = ω.im :=
by sorry

end complex_equality_l3987_398725


namespace inverse_cube_relation_l3987_398739

/-- Given that y varies inversely as the cube of x, prove that x = ∛3 when y = 18, 
    given that y = 2 when x = 3. -/
theorem inverse_cube_relation (x y : ℝ) (k : ℝ) (h1 : y * x^3 = k) 
  (h2 : 2 * 3^3 = k) (h3 : 18 * x^3 = k) : x = (3 : ℝ)^(1/3) :=
sorry

end inverse_cube_relation_l3987_398739


namespace escalator_length_is_200_l3987_398704

/-- The length of an escalator given its speed, a person's walking speed, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time_taken : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time_taken

/-- Theorem stating that the length of the escalator is 200 feet -/
theorem escalator_length_is_200 :
  escalator_length 15 5 10 = 200 := by
  sorry

end escalator_length_is_200_l3987_398704


namespace centipede_sock_shoe_arrangements_l3987_398783

/-- Represents the number of legs of the centipede -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the number of valid arrangements for putting on socks and shoes -/
def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_legs)

/-- Theorem stating the number of valid arrangements for a centipede to put on its socks and shoes -/
theorem centipede_sock_shoe_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_legs) := by sorry

end centipede_sock_shoe_arrangements_l3987_398783


namespace line_equation_through_point_with_slope_l3987_398758

/-- The general form equation of a line passing through (1, 1) with slope -3 -/
theorem line_equation_through_point_with_slope :
  ∃ (A B C : ℝ), A ≠ 0 ∨ B ≠ 0 ∧
  (∀ x y : ℝ, A * x + B * y + C = 0 ↔ y - 1 = -3 * (x - 1)) ∧
  A = 3 ∧ B = 1 ∧ C = -4 := by
sorry

end line_equation_through_point_with_slope_l3987_398758


namespace ambiguous_dates_count_l3987_398751

/-- The number of months in a year -/
def num_months : ℕ := 12

/-- The maximum day number that can be confused as a month -/
def max_ambiguous_day : ℕ := 12

/-- The number of ambiguous dates in a year -/
def num_ambiguous_dates : ℕ := num_months * max_ambiguous_day - num_months

theorem ambiguous_dates_count :
  num_ambiguous_dates = 132 :=
sorry

end ambiguous_dates_count_l3987_398751


namespace power_three_plus_four_mod_five_l3987_398764

theorem power_three_plus_four_mod_five : 3^75 + 4 ≡ 1 [ZMOD 5] := by
  sorry

end power_three_plus_four_mod_five_l3987_398764


namespace fraction_addition_l3987_398707

theorem fraction_addition (c : ℝ) : (6 + 5 * c) / 5 + 3 = (21 + 5 * c) / 5 := by
  sorry

end fraction_addition_l3987_398707


namespace probability_of_pair_l3987_398708

/-- Represents a standard deck of cards -/
def StandardDeck := 52

/-- Represents the number of cards of each rank in a standard deck -/
def CardsPerRank := 4

/-- Represents the number of ranks in a standard deck -/
def NumRanks := 13

/-- Represents the number of cards remaining after removing a pair -/
def RemainingCards := StandardDeck - 2

/-- Represents the number of ways to choose 2 cards from the remaining deck -/
def TotalChoices := (RemainingCards.choose 2)

/-- Represents the number of ranks with 4 cards after removing a pair -/
def FullRanks := NumRanks - 1

/-- Represents the number of ways to form a pair from ranks with 4 cards -/
def PairsFromFullRanks := FullRanks * (CardsPerRank.choose 2)

/-- Represents the number of ways to form a pair from the rank with 2 cards -/
def PairsFromReducedRank := 1

/-- Represents the total number of ways to form a pair -/
def TotalPairs := PairsFromFullRanks + PairsFromReducedRank

/-- The main theorem stating the probability of forming a pair -/
theorem probability_of_pair : 
  (TotalPairs : ℚ) / TotalChoices = 73 / 1225 := by sorry

end probability_of_pair_l3987_398708


namespace min_gumballs_for_three_same_color_l3987_398768

/-- Represents the colors of gumballs in the machine -/
inductive GumballColor
| Red
| Blue
| White
| Green

/-- Represents the gumball machine -/
structure GumballMachine where
  red : Nat
  blue : Nat
  white : Nat
  green : Nat

/-- Returns the minimum number of gumballs needed to guarantee 3 of the same color -/
def minGumballsForThreeSameColor (machine : GumballMachine) : Nat :=
  sorry

/-- Theorem stating that for the given gumball machine, 
    the minimum number of gumballs needed to guarantee 3 of the same color is 8 -/
theorem min_gumballs_for_three_same_color :
  let machine : GumballMachine := { red := 13, blue := 5, white := 1, green := 9 }
  minGumballsForThreeSameColor machine = 8 := by
  sorry

end min_gumballs_for_three_same_color_l3987_398768


namespace solve_flower_problem_l3987_398730

def flower_problem (yoojung_flowers namjoon_flowers : ℕ) : Prop :=
  (yoojung_flowers = 32) ∧
  (yoojung_flowers = 4 * namjoon_flowers) ∧
  (yoojung_flowers + namjoon_flowers = 40)

theorem solve_flower_problem :
  ∃ (yoojung_flowers namjoon_flowers : ℕ),
    flower_problem yoojung_flowers namjoon_flowers :=
by
  sorry

end solve_flower_problem_l3987_398730


namespace book_arrangement_problem_l3987_398736

theorem book_arrangement_problem (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 4) :
  Nat.choose n k = 126 := by
  sorry

end book_arrangement_problem_l3987_398736


namespace max_value_trig_expression_l3987_398765

theorem max_value_trig_expression (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x + 1 ≤ Real.sqrt 13 + 1 := by
  sorry

end max_value_trig_expression_l3987_398765


namespace sum_nine_equals_27_l3987_398743

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  is_arithmetic : ∀ n m : ℕ+, a (n + 1) - a n = a (m + 1) - a m
  on_line : ∀ n : ℕ+, ∃ k b : ℝ, a n = k * n + b ∧ 3 = k * 5 + b

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℝ :=
  (n : ℝ) * seq.a n

/-- The main theorem -/
theorem sum_nine_equals_27 (seq : ArithmeticSequence) : sum_n seq 9 = 27 := by
  sorry

end sum_nine_equals_27_l3987_398743


namespace prob_all_same_color_l3987_398778

/-- The probability of picking all same-colored candies from a jar -/
theorem prob_all_same_color (red blue : ℕ) (h_red : red = 15) (h_blue : blue = 5) :
  let total := red + blue
  let prob_terry_red := (red * (red - 1)) / (total * (total - 1))
  let prob_mary_red_given_terry_red := (red - 2) / (total - 2)
  let prob_all_red := prob_terry_red * prob_mary_red_given_terry_red
  let prob_terry_blue := (blue * (blue - 1)) / (total * (total - 1))
  let prob_mary_blue_given_terry_blue := (blue - 2) / (total - 2)
  let prob_all_blue := prob_terry_blue * prob_mary_blue_given_terry_blue
  prob_all_red + prob_all_blue = 31 / 76 := by
sorry

end prob_all_same_color_l3987_398778


namespace bryan_continents_l3987_398753

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := 122

/-- The total number of books Bryan collected from all continents -/
def total_books : ℕ := 488

/-- The number of continents Bryan collected books from -/
def num_continents : ℕ := total_books / books_per_continent

theorem bryan_continents :
  num_continents = 4 := by sorry

end bryan_continents_l3987_398753


namespace joe_cookies_sold_l3987_398741

/-- The number of cookies Joe sold -/
def cookies : ℕ := sorry

/-- The cost to make each cookie in dollars -/
def cost : ℚ := 1

/-- The markup percentage -/
def markup : ℚ := 20 / 100

/-- The selling price of each cookie -/
def selling_price : ℚ := cost * (1 + markup)

/-- The total revenue in dollars -/
def revenue : ℚ := 60

theorem joe_cookies_sold :
  cookies = 50 ∧
  selling_price * cookies = revenue :=
sorry

end joe_cookies_sold_l3987_398741


namespace largest_nineteen_times_digit_sum_l3987_398779

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: 399 is the largest positive integer equal to 19 times the sum of its digits -/
theorem largest_nineteen_times_digit_sum :
  ∀ n : ℕ, n > 0 → n = 19 * sum_of_digits n → n ≤ 399 := by
  sorry

end largest_nineteen_times_digit_sum_l3987_398779


namespace area_of_bcd_l3987_398784

-- Define the right triangular prism
structure RightTriangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  y : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_abc : x = (1/2) * a * b
  h_adc : y = (1/2) * b * c

-- Theorem statement
theorem area_of_bcd (prism : RightTriangularPrism) : 
  (1/2) * prism.b * prism.c = prism.y := by
  sorry

end area_of_bcd_l3987_398784


namespace margarets_mean_score_l3987_398792

def scores : List ℝ := [78, 81, 85, 87, 90, 92]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    margaret_scores.length = 2 ∧ 
    cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    cyprian_scores.sum / cyprian_scores.length = 84) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 2 ∧ 
    margaret_scores.sum / margaret_scores.length = 88.5 := by
sorry

end margarets_mean_score_l3987_398792


namespace florist_roses_problem_l3987_398715

theorem florist_roses_problem (x : ℕ) : 
  x - 2 + 32 = 41 → x = 11 := by
  sorry

end florist_roses_problem_l3987_398715


namespace quadratic_equation_roots_l3987_398702

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧
    x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end quadratic_equation_roots_l3987_398702


namespace quadratic_root_theorem_l3987_398745

theorem quadratic_root_theorem (a b : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x^2 + b*x + a = 0 ∧ x = -a) → b - a = 1 := by
  sorry

end quadratic_root_theorem_l3987_398745


namespace jerrys_action_figures_l3987_398731

theorem jerrys_action_figures (initial_figures : ℕ) : 
  (10 : ℕ) = initial_figures + 4 + 4 → initial_figures = 2 := by
sorry

end jerrys_action_figures_l3987_398731


namespace divisible_by_twelve_l3987_398786

theorem divisible_by_twelve (a b c d : ℤ) : 
  12 ∣ ((b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b)) := by
  sorry

end divisible_by_twelve_l3987_398786


namespace income_data_mean_difference_l3987_398774

/-- The difference between the mean of incorrect data and the mean of actual data -/
theorem income_data_mean_difference (T : ℝ) : 
  (T + 1200000) / 500 - (T + 120000) / 500 = 2160 := by sorry

end income_data_mean_difference_l3987_398774


namespace gcd_30_and_number_l3987_398701

theorem gcd_30_and_number (n : ℕ) : 
  70 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 30 n = 10 → n = 70 ∨ n = 80 ∨ n = 90 := by
  sorry

end gcd_30_and_number_l3987_398701


namespace inverse_function_property_l3987_398771

-- Define the function f
def f : ℕ → ℕ
| 1 => 3
| 2 => 13
| 3 => 8
| 5 => 1
| 8 => 0
| 13 => 5
| _ => 0  -- Default case for other inputs

-- Define the inverse function f_inv
def f_inv : ℕ → ℕ
| 0 => 8
| 1 => 5
| 3 => 1
| 5 => 13
| 8 => 3
| 13 => 2
| _ => 0  -- Default case for other inputs

-- Theorem statement
theorem inverse_function_property :
  f_inv ((f_inv 5 + f_inv 13) / f_inv 1) = 1 :=
by sorry

end inverse_function_property_l3987_398771


namespace sum_of_variables_l3987_398799

theorem sum_of_variables (a b c : ℝ) 
  (eq1 : b + c = 12 - 3*a)
  (eq2 : a + c = -14 - 3*b)
  (eq3 : a + b = 7 - 3*c) :
  2*a + 2*b + 2*c = 2 := by
sorry

end sum_of_variables_l3987_398799


namespace absolute_value_of_w_l3987_398713

theorem absolute_value_of_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : Complex.abs w = Real.sqrt 2 := by
  sorry

end absolute_value_of_w_l3987_398713


namespace custom_op_example_l3987_398780

-- Define the custom operation
def custom_op (a b : Int) : Int := a * (b + 1) + a * b

-- State the theorem
theorem custom_op_example : custom_op (-3) 4 = -27 := by
  sorry

end custom_op_example_l3987_398780


namespace midpoint_coordinate_sum_l3987_398793

/-- Given that N(4,9) is the midpoint of CD and C has coordinates (10,5),
    prove that the sum of the coordinates of D is 11. -/
theorem midpoint_coordinate_sum :
  let N : ℝ × ℝ := (4, 9)
  let C : ℝ × ℝ := (10, 5)
  ∀ D : ℝ × ℝ,
  (N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2) →
  D.1 + D.2 = 11 :=
by
  sorry

end midpoint_coordinate_sum_l3987_398793


namespace factorize_difference_of_squares_factorize_quadratic_l3987_398762

-- Theorem 1
theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

-- Theorem 2
theorem factorize_quadratic (x : ℝ) : 2*x^2 - 20*x + 50 = 2*(x - 5)^2 := by
  sorry

end factorize_difference_of_squares_factorize_quadratic_l3987_398762


namespace A_power_2023_l3987_398755

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, -1, 0;
     1,  0, 0;
     0,  0, 1]

theorem A_power_2023 :
  A ^ 2023 = !![0,  1, 0;
                -1,  0, 0;
                 0,  0, 1] := by sorry

end A_power_2023_l3987_398755


namespace complex_fraction_pure_imaginary_l3987_398712

def complex_is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_fraction_pure_imaginary (a : ℝ) : 
  complex_is_pure_imaginary ((a + 6 * Complex.I) / (3 - Complex.I)) → a = 2 := by
  sorry

end complex_fraction_pure_imaginary_l3987_398712


namespace workshop_workers_l3987_398787

theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (other_salary : ℝ) 
  (num_technicians : ℕ) (h1 : average_salary = 6750) 
  (h2 : technician_salary = 12000) (h3 : other_salary = 6000) 
  (h4 : num_technicians = 7) : 
  ∃ (total_workers : ℕ), total_workers = 56 ∧ 
  average_salary * total_workers = 
    num_technicians * technician_salary + 
    (total_workers - num_technicians) * other_salary :=
by sorry

end workshop_workers_l3987_398787


namespace circle_M_equations_l3987_398776

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of circle M lies in part (I)
def centerLine (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the equation of circle M for part (I)
def circleM1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 25

-- Define the equation of circle M for part (II)
def circleM2 (x y : ℝ) : Prop := (x + 7/2)^2 + (y - 1/2)^2 = 25/2

theorem circle_M_equations :
  (∀ x y : ℝ, (∃ x0 y0 : ℝ, circle1 x0 y0 ∧ circle2 x0 y0 ∧ circleM1 x0 y0) →
    (∃ xc yc : ℝ, centerLine xc yc ∧ circleM1 x y)) ∧
  (∀ x y : ℝ, (∃ x0 y0 : ℝ, circle1 x0 y0 ∧ circle2 x0 y0 ∧ circleM2 x0 y0) →
    circleM2 x y) :=
by sorry

end circle_M_equations_l3987_398776


namespace largest_package_size_l3987_398744

theorem largest_package_size (alex_markers jordan_markers : ℕ) 
  (h_alex : alex_markers = 56) (h_jordan : jordan_markers = 42) :
  Nat.gcd alex_markers jordan_markers = 14 := by
  sorry

end largest_package_size_l3987_398744


namespace problem_statement_l3987_398719

theorem problem_statement (n : ℝ) (h : n + 1/n = 6) : n^2 + 1/n^2 + 9 = 43 := by
  sorry

end problem_statement_l3987_398719


namespace ani_winning_strategy_l3987_398766

/-- Represents the state of the game with three buckets -/
structure GameState :=
  (bucket1 bucket2 bucket3 : ℕ)

/-- Defines a valid game state where each bucket has at least one marble -/
def ValidGameState (state : GameState) : Prop :=
  state.bucket1 > 0 ∧ state.bucket2 > 0 ∧ state.bucket3 > 0

/-- Defines the total number of marbles in the game -/
def TotalMarbles (state : GameState) : ℕ :=
  state.bucket1 + state.bucket2 + state.bucket3

/-- Defines a valid move in the game -/
def ValidMove (marbles : ℕ) : Prop :=
  marbles = 1 ∨ marbles = 2 ∨ marbles = 3

/-- Defines whether a game state is a winning position for the current player -/
def IsWinningPosition (state : GameState) : Prop :=
  sorry

/-- Theorem: Ani has a winning strategy if and only if n is even and n ≥ 6 -/
theorem ani_winning_strategy (n : ℕ) :
  (∃ (initialState : GameState),
    ValidGameState initialState ∧
    TotalMarbles initialState = n ∧
    IsWinningPosition initialState) ↔
  (Even n ∧ n ≥ 6) :=
sorry

end ani_winning_strategy_l3987_398766


namespace journey_time_equation_l3987_398705

theorem journey_time_equation (x : ℝ) (h : x > 0) : 
  let distance : ℝ := 50
  let taxi_speed : ℝ := x + 15
  let bus_speed : ℝ := x
  let taxi_time : ℝ := distance / taxi_speed
  let bus_time : ℝ := distance / bus_speed
  taxi_time = 2/3 * bus_time → distance / taxi_speed = 2/3 * (distance / bus_speed) :=
by sorry

end journey_time_equation_l3987_398705


namespace fraction_simplification_l3987_398706

theorem fraction_simplification (y : ℚ) (h : y = 77) : (7 * y + 77) / 77 = 8 := by
  sorry

end fraction_simplification_l3987_398706


namespace max_value_theorem_l3987_398724

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 2 + 6 * y * z ≤ Real.sqrt 11 :=
sorry

end max_value_theorem_l3987_398724


namespace expression_simplification_and_evaluation_l3987_398769

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 6) :
  (1 + 2 / (x + 1)) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end expression_simplification_and_evaluation_l3987_398769


namespace sum_x_y_equals_negative_two_l3987_398763

theorem sum_x_y_equals_negative_two (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := by
sorry

end sum_x_y_equals_negative_two_l3987_398763
