import Mathlib

namespace mary_lamb_count_l1594_159489

/-- The number of lambs Mary has after a series of events --/
def final_lamb_count (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
                     (lambs_traded : ℕ) (extra_lambs_found : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb - lambs_traded + extra_lambs_found

/-- Theorem stating that Mary ends up with 14 lambs --/
theorem mary_lamb_count : 
  final_lamb_count 6 2 2 3 7 = 14 := by
  sorry

end mary_lamb_count_l1594_159489


namespace optimal_room_configuration_l1594_159461

/-- Represents a configuration of rooms --/
structure RoomConfiguration where
  large_rooms : ℕ
  small_rooms : ℕ

/-- Checks if a given room configuration is valid for the problem --/
def is_valid_configuration (config : RoomConfiguration) : Prop :=
  3 * config.large_rooms + 2 * config.small_rooms = 26

/-- Calculates the total number of rooms in a configuration --/
def total_rooms (config : RoomConfiguration) : ℕ :=
  config.large_rooms + config.small_rooms

/-- Theorem: The optimal room configuration includes exactly one small room --/
theorem optimal_room_configuration :
  ∃ (config : RoomConfiguration),
    is_valid_configuration config ∧
    (∀ (other : RoomConfiguration), is_valid_configuration other →
      total_rooms config ≤ total_rooms other) ∧
    config.small_rooms = 1 :=
sorry

end optimal_room_configuration_l1594_159461


namespace square_rectangle_area_problem_l1594_159497

theorem square_rectangle_area_problem :
  ∃ (x₁ x₂ : ℝ),
    (∀ x : ℝ, (x - 3) * (x + 4) = 2 * (x - 2)^2 → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 9 := by
  sorry

end square_rectangle_area_problem_l1594_159497


namespace odd_integers_between_9_and_39_l1594_159416

theorem odd_integers_between_9_and_39 :
  let first_term := 9
  let last_term := 39
  let sum := 384
  let n := (last_term - first_term) / 2 + 1
  n = 16 ∧ sum = n / 2 * (first_term + last_term) := by
sorry

end odd_integers_between_9_and_39_l1594_159416


namespace angle_cosine_equality_l1594_159477

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side on the ray 3x + 4y = 0 (x ≤ 0), prove that cos(2α + π/6) = (7√3 + 24) / 50 -/
theorem angle_cosine_equality (α : Real) 
    (h1 : ∃ (x y : Real), x ≤ 0 ∧ 3 * x + 4 * y = 0 ∧ 
          x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
          y = Real.sin α * Real.sqrt (x^2 + y^2)) : 
    Real.cos (2 * α + π / 6) = (7 * Real.sqrt 3 + 24) / 50 := by
  sorry

end angle_cosine_equality_l1594_159477


namespace fifteen_percent_of_900_is_135_l1594_159431

theorem fifteen_percent_of_900_is_135 : ∃ x : ℝ, x * 0.15 = 135 ∧ x = 900 := by
  sorry

end fifteen_percent_of_900_is_135_l1594_159431


namespace parabola_properties_l1594_159473

/-- A parabola with equation y = x² - 2mx + m² - 9 where m is a constant -/
def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 9

/-- The x-coordinates of the intersection points of the parabola with the x-axis -/
def roots (m : ℝ) : Set ℝ := {x : ℝ | parabola m x = 0}

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def y_coord (m : ℝ) (x : ℝ) : ℝ := parabola m x

theorem parabola_properties (m : ℝ) :
  (∃ (A B : ℝ), A ∈ roots m ∧ B ∈ roots m ∧ A ≠ B) →
  (∀ x, y_coord m x ≥ -9) ∧
  (∃ (A B : ℝ), A ∈ roots m ∧ B ∈ roots m ∧ |A - B| = 6) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < m - 1 → y_coord m x₁ > y_coord m x₂) ∧
  (y_coord m (m + 1) < y_coord m (m - 3)) :=
by sorry

end parabola_properties_l1594_159473


namespace max_third_altitude_is_six_l1594_159408

/-- A scalene triangle with two known altitudes -/
structure ScaleneTriangle where
  /-- The length of the first known altitude -/
  altitude1 : ℝ
  /-- The length of the second known altitude -/
  altitude2 : ℝ
  /-- The triangle is scalene -/
  scalene : altitude1 ≠ altitude2
  /-- The altitudes are positive -/
  positive1 : altitude1 > 0
  positive2 : altitude2 > 0

/-- The maximum possible integer length of the third altitude -/
def max_third_altitude (t : ScaleneTriangle) : ℕ :=
  6

/-- Theorem stating that the maximum possible integer length of the third altitude is 6 -/
theorem max_third_altitude_is_six (t : ScaleneTriangle) 
  (h1 : t.altitude1 = 6 ∨ t.altitude2 = 6) 
  (h2 : t.altitude1 = 18 ∨ t.altitude2 = 18) : 
  max_third_altitude t = 6 := by
  sorry

#check max_third_altitude_is_six

end max_third_altitude_is_six_l1594_159408


namespace probability_of_observing_change_l1594_159430

/-- Represents the duration of the traffic light cycle in seconds -/
def cycle_duration : ℕ := 63

/-- Represents the points in the cycle where color changes occur -/
def change_points : List ℕ := [30, 33, 63]

/-- Represents the duration of the observation interval in seconds -/
def observation_duration : ℕ := 4

/-- Calculates the total duration of intervals where a change can be observed -/
def total_change_duration (cycle : ℕ) (changes : List ℕ) (obs : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the probability of observing a color change -/
theorem probability_of_observing_change :
  (total_change_duration cycle_duration change_points observation_duration : ℚ) / cycle_duration = 5 / 21 :=
sorry

end probability_of_observing_change_l1594_159430


namespace range_of_a_l1594_159442

-- Define proposition p
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition q
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Theorem statement
theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) : 
  0 ≤ a ∧ a ≤ 1/4 := by sorry

end range_of_a_l1594_159442


namespace other_communities_count_l1594_159400

theorem other_communities_count (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 32 / 100 →
  sikh_percent = 10 / 100 →
  ∃ (other_boys : ℕ), other_boys = 119 ∧ 
    (other_boys : ℚ) / total_boys = 1 - (muslim_percent + hindu_percent + sikh_percent) :=
by sorry

end other_communities_count_l1594_159400


namespace second_polygon_sides_l1594_159487

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times as long as the other, prove that the number
    of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) :
  s > 0 →
  50 * (3 * s) = n * s →
  n = 150 := by
  sorry

end second_polygon_sides_l1594_159487


namespace logarithm_sum_equation_l1594_159413

theorem logarithm_sum_equation (x : ℝ) (h : x > 0) :
  (1 / Real.log x / Real.log 3) + (1 / Real.log x / Real.log 4) + (1 / Real.log x / Real.log 5) = 1 →
  x = 60 := by
  sorry

end logarithm_sum_equation_l1594_159413


namespace west_distance_calculation_l1594_159478

-- Define the given constants
def total_distance : ℝ := 150
def north_distance : ℝ := 55

-- Theorem statement
theorem west_distance_calculation :
  total_distance - north_distance = 95 := by sorry

end west_distance_calculation_l1594_159478


namespace frustum_slant_height_l1594_159458

/-- 
Given a cone cut by a plane parallel to its base forming a frustum:
- r: radius of the upper base of the frustum
- 4r: radius of the lower base of the frustum
- 3: slant height of the removed cone
- h: slant height of the frustum

Prove that h = 9
-/
theorem frustum_slant_height (r : ℝ) (h : ℝ) : h / (4 * r) = (h + 3) / (5 * r) → h = 9 := by
  sorry

end frustum_slant_height_l1594_159458


namespace all_subtracting_not_purple_not_all_happy_are_purple_some_happy_cant_subtract_l1594_159492

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (purple happy can_add can_subtract : Snake → Prop)

-- Define the number of snakes
variable (total_snakes : ℕ)
variable (purple_snakes : ℕ)
variable (happy_snakes : ℕ)

-- State the given conditions
variable (h1 : total_snakes = 20)
variable (h2 : purple_snakes = 6)
variable (h3 : happy_snakes = 8)
variable (h4 : ∃ s, happy s ∧ can_add s)
variable (h5 : ∀ s, purple s → ¬can_subtract s)
variable (h6 : ∀ s, ¬can_subtract s → ¬can_add s)

-- State the theorems to be proved
theorem all_subtracting_not_purple : ∀ s, can_subtract s → ¬purple s := by sorry

theorem not_all_happy_are_purple : ¬(∀ s, happy s → purple s) := by sorry

theorem some_happy_cant_subtract : ∃ s, happy s ∧ ¬can_subtract s := by sorry

end all_subtracting_not_purple_not_all_happy_are_purple_some_happy_cant_subtract_l1594_159492


namespace hyperbola_and_line_equations_l1594_159494

/-- Given a hyperbola with specified properties, prove its equation and the equation of a line intersecting it. -/
theorem hyperbola_and_line_equations
  (a b : ℝ)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_asymptote : ∀ x y : ℝ, y = 2 * x → (∃ t : ℝ, y = t * x ∧ y^2 / a^2 - x^2 / b^2 = 1))
  (h_focus_distance : ∃ F : ℝ × ℝ, ∀ x y : ℝ, y = 2 * x → Real.sqrt ((F.1 - x)^2 + (F.2 - y)^2) = 1)
  (h_midpoint : ∃ A B : ℝ × ℝ, A.1 ≠ B.1 ∧ A.2 ≠ B.2 ∧
    (A.2^2 / a^2 - A.1^2 / b^2 = 1) ∧
    (B.2^2 / a^2 - B.1^2 / b^2 = 1) ∧
    ((A.1 + B.1) / 2 = 1) ∧
    ((A.2 + B.2) / 2 = 4)) :
  (a = 2 ∧ b = 1) ∧
  (∀ x y : ℝ, y^2 / 4 - x^2 = 1 ↔ y^2 / a^2 - x^2 / b^2 = 1) ∧
  (∃ k m : ℝ, k = 1 ∧ m = 3 ∧ ∀ x y : ℝ, y^2 / 4 - x^2 = 1 → (x - y + m = 0 ↔ ∃ t : ℝ, x = 1 + t ∧ y = 4 + t)) :=
by sorry

end hyperbola_and_line_equations_l1594_159494


namespace mary_minus_robert_eq_two_l1594_159450

/-- Represents the candy distribution problem -/
structure CandyDistribution where
  total : Nat
  kate : Nat
  robert : Nat
  bill : Nat
  mary : Nat
  kate_pieces : kate = 4
  robert_more_than_kate : robert = kate + 2
  bill_less_than_mary : bill + 6 = mary
  kate_more_than_bill : kate = bill + 2
  mary_more_than_robert : mary > robert

/-- Proves that Mary gets 2 more pieces of candy than Robert -/
theorem mary_minus_robert_eq_two (cd : CandyDistribution) : cd.mary - cd.robert = 2 := by
  sorry

end mary_minus_robert_eq_two_l1594_159450


namespace rectangular_field_area_l1594_159459

theorem rectangular_field_area (a b c : ℝ) (h1 : a = 15) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) : a * b = 120 := by
  sorry

end rectangular_field_area_l1594_159459


namespace pasture_rent_l1594_159457

/-- Represents a milkman's grazing details -/
structure MilkmanGrazing where
  cows : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given the grazing details of milkmen -/
def totalRent (milkmen : List MilkmanGrazing) (aShare : ℕ) : ℕ :=
  let totalCowMonths := milkmen.foldl (fun acc m => acc + m.cows * m.months) 0
  let aMonths := (milkmen.head?).map (fun m => m.cows * m.months)
  match aMonths with
  | some months => (totalCowMonths * aShare) / months
  | none => 0

/-- Theorem stating that the total rent of the pasture is 3250 -/
theorem pasture_rent :
  let milkmen := [
    MilkmanGrazing.mk 24 3,  -- A
    MilkmanGrazing.mk 10 5,  -- B
    MilkmanGrazing.mk 35 4,  -- C
    MilkmanGrazing.mk 21 3   -- D
  ]
  totalRent milkmen 720 = 3250 := by
    sorry

end pasture_rent_l1594_159457


namespace only_frustum_has_two_parallel_surfaces_l1594_159470

-- Define the geometric bodies
inductive GeometricBody
  | Pyramid
  | Prism
  | Frustum
  | Cuboid

-- Define a function to count parallel surfaces
def parallelSurfaceCount (body : GeometricBody) : Nat :=
  match body with
  | GeometricBody.Pyramid => 0
  | GeometricBody.Prism => 6
  | GeometricBody.Frustum => 2
  | GeometricBody.Cuboid => 6

-- Theorem statement
theorem only_frustum_has_two_parallel_surfaces :
  ∀ (body : GeometricBody),
    parallelSurfaceCount body = 2 ↔ body = GeometricBody.Frustum :=
by
  sorry


end only_frustum_has_two_parallel_surfaces_l1594_159470


namespace exponent_monotonicity_l1594_159438

theorem exponent_monotonicity (a x₁ x₂ : ℝ) :
  (a > 1 ∧ x₁ > x₂ → a^x₁ > a^x₂) ∧
  (0 < a ∧ a < 1 ∧ x₁ > x₂ → a^x₁ < a^x₂) := by
  sorry

end exponent_monotonicity_l1594_159438


namespace inequality_proof_l1594_159439

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l1594_159439


namespace lost_sea_creatures_l1594_159453

/-- Represents the count of sea creatures Harry collected --/
structure SeaCreatures where
  seaStars : ℕ
  seashells : ℕ
  snails : ℕ
  crabs : ℕ

/-- Represents the number of each type of sea creature that reproduced --/
structure Reproduction where
  seaStars : ℕ
  seashells : ℕ
  snails : ℕ

def initialCount : SeaCreatures :=
  { seaStars := 34, seashells := 21, snails := 29, crabs := 17 }

def reproductionCount : Reproduction :=
  { seaStars := 5, seashells := 3, snails := 4 }

def finalCount : ℕ := 105

def totalAfterReproduction (initial : SeaCreatures) (reproduction : Reproduction) : ℕ :=
  (initial.seaStars + reproduction.seaStars) +
  (initial.seashells + reproduction.seashells) +
  (initial.snails + reproduction.snails) +
  initial.crabs

theorem lost_sea_creatures : 
  totalAfterReproduction initialCount reproductionCount - finalCount = 8 := by
  sorry

end lost_sea_creatures_l1594_159453


namespace log_relationship_l1594_159418

theorem log_relationship (a b x : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ 0 < b ∧ b ≠ 1 ∧ 0 < x) :
  5 * (Real.log x / Real.log a)^2 + 2 * (Real.log x / Real.log b)^2 = 15 * (Real.log x)^2 / (Real.log a * Real.log b) →
  b = a^((3 + Real.sqrt 37) / 2) ∨ b = a^((3 - Real.sqrt 37) / 2) := by
sorry

end log_relationship_l1594_159418


namespace event_arrangements_l1594_159493

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose n k) * k * k

theorem event_arrangements : number_of_arrangements 6 3 = 180 := by
  sorry

end event_arrangements_l1594_159493


namespace cookout_buns_needed_l1594_159423

/-- Calculates the number of packs of buns needed for a cookout --/
def buns_needed (total_guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ) : ℕ :=
  let guests_eating_burgers := total_guests - no_meat_guests
  let total_burgers := guests_eating_burgers * burgers_per_guest
  let buns_needed := total_burgers - (no_bread_guests * burgers_per_guest)
  (buns_needed + buns_per_pack - 1) / buns_per_pack

theorem cookout_buns_needed :
  buns_needed 10 3 1 1 8 = 3 := by
  sorry

end cookout_buns_needed_l1594_159423


namespace unique_positive_solution_l1594_159422

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x :=
by
  -- The proof goes here
  sorry

end unique_positive_solution_l1594_159422


namespace shorter_diagonal_length_l1594_159404

/-- Given two vectors in a 2D plane satisfying specific conditions, 
    prove that the length of the shorter diagonal of the parallelogram 
    formed by these vectors is √3. -/
theorem shorter_diagonal_length (a b : ℝ × ℝ) : 
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a • b = 1 →  -- This represents cos(π/3) = 1/2, as |a||b|cos(π/3) = 1
  Real.sqrt 3 = min (‖a + b‖) (‖a - b‖) := by
  sorry


end shorter_diagonal_length_l1594_159404


namespace vet_clinic_dog_treatment_cost_l1594_159460

theorem vet_clinic_dog_treatment_cost (cat_cost : ℕ) (num_dogs num_cats total_cost : ℕ) :
  cat_cost = 40 →
  num_dogs = 20 →
  num_cats = 60 →
  total_cost = 3600 →
  ∃ (dog_cost : ℕ), dog_cost * num_dogs + cat_cost * num_cats = total_cost ∧ dog_cost = 60 :=
by sorry

end vet_clinic_dog_treatment_cost_l1594_159460


namespace crackers_per_person_l1594_159454

theorem crackers_per_person 
  (num_friends : ℕ)
  (cracker_ratio cake_ratio : ℕ)
  (initial_crackers initial_cakes : ℕ)
  (h1 : num_friends = 6)
  (h2 : cracker_ratio = 3)
  (h3 : cake_ratio = 5)
  (h4 : initial_crackers = 72)
  (h5 : initial_cakes = 180) :
  initial_crackers / (cracker_ratio * num_friends) = 12 :=
by sorry

end crackers_per_person_l1594_159454


namespace angle_inequality_equivalence_l1594_159421

theorem angle_inequality_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.sin θ - x * (1 - 2*x) + (1 - 3*x)^2 * Real.cos θ > 0) ↔
  (0 < θ ∧ θ < Real.pi / 2) :=
by sorry

end angle_inequality_equivalence_l1594_159421


namespace polynomial_roots_in_arithmetic_progression_l1594_159486

theorem polynomial_roots_in_arithmetic_progression (j k : ℝ) : 
  (∃ (b d : ℝ), d ≠ 0 ∧ 
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 400 = 0 ↔ 
      (x = b ∨ x = b + d ∨ x = b + 2*d ∨ x = b + 3*d)) ∧
    (b ≠ b + d) ∧ (b + d ≠ b + 2*d) ∧ (b + 2*d ≠ b + 3*d))
  → j = -200 := by
sorry

end polynomial_roots_in_arithmetic_progression_l1594_159486


namespace right_triangle_consecutive_sides_l1594_159433

theorem right_triangle_consecutive_sides : 
  ∀ (a b c : ℕ), 
  (a * a + b * b = c * c) →  -- Pythagorean theorem for right-angled triangle
  (b = a + 1 ∧ c = b + 1) →  -- Sides are consecutive natural numbers
  (a = 3 ∧ b = 4) →          -- Two sides are 3 and 4
  c = 5 := by               -- The third side is 5
sorry

end right_triangle_consecutive_sides_l1594_159433


namespace complex_fraction_equals_neg_i_l1594_159425

theorem complex_fraction_equals_neg_i : (1 - I) / (1 + I) = -I := by sorry

end complex_fraction_equals_neg_i_l1594_159425


namespace point_on_angle_terminal_side_l1594_159491

theorem point_on_angle_terminal_side (P : ℝ × ℝ) (θ : ℝ) (h1 : θ = 2 * π / 3) (h2 : P.1 = -1) :
  P.2 = Real.sqrt 3 := by
  sorry

end point_on_angle_terminal_side_l1594_159491


namespace slope_relation_l1594_159401

theorem slope_relation (k : ℝ) : 
  (∃ α : ℝ, α = (2 : ℝ) ∧ (2 : ℝ) * α = k) → k = -(4 : ℝ) / 3 := by
  sorry

end slope_relation_l1594_159401


namespace unique_magnitude_of_quadratic_root_l1594_159481

theorem unique_magnitude_of_quadratic_root : ∃! m : ℝ, ∃ z : ℂ, z^2 - 10*z + 52 = 0 ∧ Complex.abs z = m :=
by sorry

end unique_magnitude_of_quadratic_root_l1594_159481


namespace team_formations_count_l1594_159424

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to form a team of 3 teachers from 4 female and 5 male teachers,
    with the condition that the team must include both male and female teachers -/
def teamFormations : ℕ :=
  choose 5 1 * choose 4 2 + choose 5 2 * choose 4 1

theorem team_formations_count :
  teamFormations = 70 := by sorry

end team_formations_count_l1594_159424


namespace smallest_common_multiple_12_9_l1594_159485

theorem smallest_common_multiple_12_9 : ∃ n : ℕ, n > 0 ∧ 12 ∣ n ∧ 9 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 12 ∣ m ∧ 9 ∣ m) → n ≤ m := by
  sorry

end smallest_common_multiple_12_9_l1594_159485


namespace five_seventeenths_repetend_l1594_159496

/-- The repetend of a rational number a/b is the repeating sequence of digits in its decimal expansion. -/
def repetend (a b : ℕ) : List ℕ := sorry

/-- Returns the first n digits of a list. -/
def firstNDigits (n : ℕ) (l : List ℕ) : List ℕ := sorry

theorem five_seventeenths_repetend :
  firstNDigits 6 (repetend 5 17) = [2, 9, 4, 1, 1, 7] := by sorry

end five_seventeenths_repetend_l1594_159496


namespace bus_problem_l1594_159445

/-- The number of people who got off at the second stop of a bus route -/
def people_off_second_stop (initial : ℕ) (first_off : ℕ) (second_on : ℕ) (third_off : ℕ) (third_on : ℕ) (final : ℕ) : ℕ :=
  initial - first_off - final + second_on - third_off + third_on

theorem bus_problem : people_off_second_stop 50 15 2 4 3 28 = 8 := by
  sorry

end bus_problem_l1594_159445


namespace equation_solution_l1594_159451

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^3/a = b^2 + a^3/b → a = b ∨ a = -b := by
  sorry

end equation_solution_l1594_159451


namespace car_not_sold_probability_l1594_159495

/-- Given the odds of selling a car on a given day are 5:6, 
    the probability that the car is not sold on that day is 6/11 -/
theorem car_not_sold_probability (odds_success : ℚ) (odds_failure : ℚ) :
  odds_success = 5/6 → odds_failure = 6/5 →
  (odds_failure / (odds_success + 1)) = 6/11 := by
  sorry

end car_not_sold_probability_l1594_159495


namespace max_value_theorem_l1594_159499

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  2 * Real.sqrt (a * b * c / 8) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 :=
by sorry

end max_value_theorem_l1594_159499


namespace new_person_weight_new_person_weight_is_87_l1594_159455

/-- The weight of a new person joining a group, given specific conditions -/
theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) : ℝ :=
  let total_increase := initial_count * avg_increase
  leaving_weight + total_increase

/-- Proof that the new person's weight is 87 kg under given conditions -/
theorem new_person_weight_is_87 :
  new_person_weight 8 67 2.5 = 87 := by
  sorry

end new_person_weight_new_person_weight_is_87_l1594_159455


namespace y_divisibility_l1594_159463

def y : ℕ := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem y_divisibility :
  (∃ k : ℕ, y = 5 * k) ∧
  (∃ k : ℕ, y = 10 * k) ∧
  (∃ k : ℕ, y = 20 * k) ∧
  (∃ k : ℕ, y = 40 * k) :=
by sorry

end y_divisibility_l1594_159463


namespace expression_sum_equals_one_l1594_159405

theorem expression_sum_equals_one (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hprod : x * y * z = 1) :
  1 / (1 + x + x * y) + y / (1 + y + y * z) + x * z / (1 + z + x * z) = 1 := by
  sorry

end expression_sum_equals_one_l1594_159405


namespace subset_of_any_set_implies_zero_l1594_159498

theorem subset_of_any_set_implies_zero (a : ℝ) :
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end subset_of_any_set_implies_zero_l1594_159498


namespace swimmer_downstream_distance_l1594_159479

/-- Proves that a swimmer travels 32 km downstream given specific conditions -/
theorem swimmer_downstream_distance 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_distance = 24) 
  (h2 : time = 4) 
  (h3 : still_water_speed = 7) : 
  ∃ (downstream_distance : ℝ), downstream_distance = 32 := by
  sorry

end swimmer_downstream_distance_l1594_159479


namespace min_value_ab_l1594_159420

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 3 ∧ a₀ * b₀ = 9 :=
sorry

end min_value_ab_l1594_159420


namespace polynomial_coefficient_sum_l1594_159432

theorem polynomial_coefficient_sum :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (∀ x : ℝ, x + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                     a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                     a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a + a₂ + a₃ + a₄ + a₅ + a₆ + a₈ = 510 := by
sorry

end polynomial_coefficient_sum_l1594_159432


namespace quadratic_sum_l1594_159484

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (8 * x^2 + 64 * x + 512 = a * (x + b)^2 + c) ∧ (a + b + c = 396) := by
  sorry

end quadratic_sum_l1594_159484


namespace election_probabilities_l1594_159436

theorem election_probabilities 
  (pA pB pC : ℝ)
  (hA : pA = 4/5)
  (hB : pB = 3/5)
  (hC : pC = 7/10) :
  let p_exactly_one := pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC
  let p_at_most_two := 1 - pA * pB * pC
  (p_exactly_one = 47/250) ∧ (p_at_most_two = 83/125) := by
  sorry

end election_probabilities_l1594_159436


namespace inequality_solution_set_l1594_159471

theorem inequality_solution_set : 
  ∀ x : ℝ, (3/20 : ℝ) + |x - 7/40| < (11/40 : ℝ) ↔ x ∈ Set.Ioo (1/20 : ℝ) (3/10 : ℝ) := by sorry

end inequality_solution_set_l1594_159471


namespace subtraction_multiplication_equality_l1594_159410

theorem subtraction_multiplication_equality : ((3.54 - 1.32) * 2) = 4.44 := by
  sorry

end subtraction_multiplication_equality_l1594_159410


namespace cave_door_weight_theorem_l1594_159474

/-- The weight already on the switch, in pounds. -/
def weight_on_switch : ℕ := 234

/-- The total weight needed, in pounds. -/
def total_weight_needed : ℕ := 712

/-- The additional weight needed to open the cave doors, in pounds. -/
def additional_weight_needed : ℕ := total_weight_needed - weight_on_switch

theorem cave_door_weight_theorem : additional_weight_needed = 478 := by
  sorry

end cave_door_weight_theorem_l1594_159474


namespace alcohol_solution_proof_l1594_159402

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution that is 30% alcohol 
    results in a 50% alcohol solution -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_percentage : ℝ := 0.30
  let target_percentage : ℝ := 0.50
  let added_alcohol : ℝ := 2.4
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol_volume : ℝ := initial_volume * initial_percentage + added_alcohol
  final_alcohol_volume / final_volume = target_percentage :=
by sorry

end alcohol_solution_proof_l1594_159402


namespace range_of_T_l1594_159483

-- Define the function T
def T (x : ℝ) : ℝ := |2 * x - 1|

-- State the theorem
theorem range_of_T (x : ℝ) : 
  (∀ a : ℝ, T x ≥ |1 + a| - |2 - a|) → 
  x ∈ Set.Ici 2 ∪ Set.Iic (-1) :=
sorry

end range_of_T_l1594_159483


namespace school_distance_l1594_159434

/-- The distance between a student's house and school, given travel time conditions. -/
theorem school_distance (t : ℝ) : 
  (t + 1/3 = 24/9) → (t - 1/3 = 24/12) → 24 = 24 := by
  sorry

end school_distance_l1594_159434


namespace animal_mortality_probability_l1594_159468

/-- The probability of an animal dying in each of the first 3 months, given survival data -/
theorem animal_mortality_probability (total : ℕ) (survivors : ℝ) (p : ℝ) 
  (h_total : total = 400)
  (h_survivors : survivors = 291.6)
  (h_survival_equation : survivors = total * (1 - p)^3) :
  p = 0.1 := by
sorry

end animal_mortality_probability_l1594_159468


namespace factorization_equality_l1594_159456

theorem factorization_equality (m n : ℝ) : 
  m^2 - n^2 + 2*m - 2*n = (m - n)*(m + n + 2) := by
  sorry

end factorization_equality_l1594_159456


namespace min_value_quadratic_form_l1594_159441

theorem min_value_quadratic_form (x₁ x₂ x₃ x₄ : ℝ) 
  (h : 5*x₁ + 6*x₂ - 7*x₃ + 4*x₄ = 1) : 
  3*x₁^2 + 2*x₂^2 + 5*x₃^2 + x₄^2 ≥ 15/782 := by
  sorry

end min_value_quadratic_form_l1594_159441


namespace jellybean_problem_l1594_159414

theorem jellybean_problem :
  ∃ (n : ℕ), 
    n ≥ 150 ∧ 
    n % 17 = 9 ∧ 
    (∀ m : ℕ, m ≥ 150 ∧ m % 17 = 9 → m ≥ n) ∧
    n = 162 := by
  sorry

end jellybean_problem_l1594_159414


namespace square_plus_linear_plus_one_l1594_159482

theorem square_plus_linear_plus_one (a : ℝ) : 
  a^2 + a - 5 = 0 → a^2 + a + 1 = 6 := by
  sorry

end square_plus_linear_plus_one_l1594_159482


namespace johns_family_members_l1594_159476

/-- The number of family members on John's father's side -/
def fathers_side : ℕ := sorry

/-- The total number of family members -/
def total_members : ℕ := 23

/-- The ratio of mother's side to father's side -/
def mother_ratio : ℚ := 13/10

theorem johns_family_members :
  fathers_side = 10 ∧
  (fathers_side : ℚ) * (1 + mother_ratio - 1) + fathers_side = total_members :=
sorry

end johns_family_members_l1594_159476


namespace no_large_squares_in_H_l1594_159409

/-- The set of points (x,y) with integer coordinates satisfying 2 ≤ |x| ≤ 6 and 2 ≤ |y| ≤ 6 -/
def H : Set (ℤ × ℤ) :=
  {p | 2 ≤ |p.1| ∧ |p.1| ≤ 6 ∧ 2 ≤ |p.2| ∧ |p.2| ≤ 6}

/-- A square with side length at least 8 -/
def IsValidSquare (s : Set (ℤ × ℤ)) : Prop :=
  ∃ (a b c d : ℤ × ℤ), s = {a, b, c, d} ∧
  (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ 64 ∧
  (b.1 - c.1)^2 + (b.2 - c.2)^2 ≥ 64 ∧
  (c.1 - d.1)^2 + (c.2 - d.2)^2 ≥ 64 ∧
  (d.1 - a.1)^2 + (d.2 - a.2)^2 ≥ 64

theorem no_large_squares_in_H :
  ¬∃ s : Set (ℤ × ℤ), (∀ p ∈ s, p ∈ H) ∧ IsValidSquare s := by
  sorry

end no_large_squares_in_H_l1594_159409


namespace frog_jumped_farther_l1594_159465

/-- The distance the grasshopper jumped in inches -/
def grasshopper_jump : ℕ := 36

/-- The distance the frog jumped in inches -/
def frog_jump : ℕ := 53

/-- The difference between the frog's jump and the grasshopper's jump -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

theorem frog_jumped_farther : jump_difference = 17 := by
  sorry

end frog_jumped_farther_l1594_159465


namespace underground_ticket_cost_l1594_159490

/-- The cost of one ticket to the underground. -/
def ticket_cost (tickets_per_minute : ℕ) (total_minutes : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (tickets_per_minute * total_minutes)

/-- Theorem stating that the cost of one ticket is $3. -/
theorem underground_ticket_cost :
  ticket_cost 5 6 90 = 3 := by
  sorry

end underground_ticket_cost_l1594_159490


namespace coins_in_pockets_l1594_159472

/-- The number of ways to place n identical objects into k distinct containers -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem of placing 5 identical coins into 3 different pockets -/
theorem coins_in_pockets : stars_and_bars 5 3 = 21 := by sorry

end coins_in_pockets_l1594_159472


namespace circle_equation_l1594_159449

-- Define the line L
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ ∃ t : ℝ, p.2 = 1 + t}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 3 = 0}

-- Define the center of the circle
def center : ℝ × ℝ := ((-1 : ℝ), (0 : ℝ))

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 2}

theorem circle_equation (h1 : center ∈ L ∩ x_axis) 
  (h2 : ∀ p ∈ C, p ∈ tangent_line → (∃ q ∈ C, q ≠ p ∧ Set.Subset C {r | r = p ∨ r = q})) :
  C = {p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 2} := by
  sorry

end circle_equation_l1594_159449


namespace blue_balls_removed_l1594_159406

theorem blue_balls_removed (initial_total : Nat) (initial_blue : Nat) (final_probability : Rat) :
  initial_total = 25 →
  initial_blue = 9 →
  final_probability = 1/5 →
  ∃ (removed : Nat), 
    removed ≤ initial_blue ∧
    (initial_blue - removed : Rat) / (initial_total - removed : Rat) = final_probability ∧
    removed = 5 :=
by sorry

end blue_balls_removed_l1594_159406


namespace stamp_problem_l1594_159469

/-- Returns true if postage can be formed with given denominations -/
def can_form_postage (d1 d2 d3 amount : ℕ) : Prop :=
  ∃ (x y z : ℕ), d1 * x + d2 * y + d3 * z = amount

/-- Returns true if n satisfies the stamp problem conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n > 0 ∧
  (∀ m : ℕ, m > 70 → can_form_postage 3 n (n+1) m) ∧
  ¬(can_form_postage 3 n (n+1) 70)

theorem stamp_problem :
  ∃! (n : ℕ), satisfies_conditions n ∧ n = 37 :=
sorry

end stamp_problem_l1594_159469


namespace triangle_angle_determination_l1594_159435

theorem triangle_angle_determination (a b c A B C : ℝ) : 
  a = Real.sqrt 3 → 
  b = Real.sqrt 2 → 
  B = π / 4 → 
  (a = 2 * Real.sin (A / 2)) → 
  (b = 2 * Real.sin (B / 2)) → 
  (c = 2 * Real.sin (C / 2)) → 
  A + B + C = π → 
  (A = π / 3 ∨ A = 2 * π / 3) := by
sorry

end triangle_angle_determination_l1594_159435


namespace regular_polygon_area_l1594_159488

theorem regular_polygon_area (n : ℕ) (R : ℝ) : 
  n > 2 → 
  R > 0 → 
  (1 / 2 : ℝ) * n * R^2 * Real.sin ((2 * Real.pi) / n) = 2 * R^2 → 
  n = 12 := by
sorry

end regular_polygon_area_l1594_159488


namespace mod_congruence_unique_solution_l1594_159415

theorem mod_congruence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 48156 ≡ n [ZMOD 17] ∧ n = 14 := by
  sorry

end mod_congruence_unique_solution_l1594_159415


namespace statistics_properties_l1594_159475

def data : List ℝ := [2, 3, 6, 9, 3, 7]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem statistics_properties :
  mode data = 3 ∧
  median data = 4.5 ∧
  mean data = 5 ∧
  range data = 7 := by sorry

end statistics_properties_l1594_159475


namespace complex_magnitude_l1594_159403

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l1594_159403


namespace min_diff_composite_sum_96_l1594_159467

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem min_diff_composite_sum_96 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 96 ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → c + d = 96 → c ≠ d →
  (max c d - min c d) ≥ (max a b - min a b) ∧ (max a b - min a b) = 4 :=
sorry

end min_diff_composite_sum_96_l1594_159467


namespace largest_integer_with_conditions_l1594_159419

theorem largest_integer_with_conditions : 
  ∃ (n : ℕ), n = 243 ∧ 
  (∀ m : ℕ, (200 < m ∧ m < 250 ∧ ∃ k : ℕ, 12 * m = k^2) → m ≤ n) :=
by sorry

end largest_integer_with_conditions_l1594_159419


namespace oil_drilling_probability_l1594_159426

/-- The probability of drilling into an oil layer in a sea area -/
theorem oil_drilling_probability (total_area oil_area : ℝ) (h1 : total_area = 10000) (h2 : oil_area = 40) :
  oil_area / total_area = 0.004 := by
sorry

end oil_drilling_probability_l1594_159426


namespace first_game_score_l1594_159448

def basketball_scores : List ℕ := [68, 70, 61, 74, 62, 65, 74]

theorem first_game_score (mean : ℚ) (h1 : mean = 67.9) :
  ∃ x : ℕ, (x :: basketball_scores).length = 8 ∧ 
  (((x :: basketball_scores).sum : ℚ) / 8 = mean) ∧
  x = 69 := by
  sorry

end first_game_score_l1594_159448


namespace number_of_type_C_is_16_l1594_159429

/-- Represents the types of people in the problem -/
inductive PersonType
| A
| B
| C

/-- The total number of people -/
def total_people : ℕ := 25

/-- The number of people who answered "yes" to "Are you a Type A person?" -/
def yes_to_A : ℕ := 17

/-- The number of people who answered "yes" to "Are you a Type C person?" -/
def yes_to_C : ℕ := 12

/-- The number of people who answered "yes" to "Are you a Type B person?" -/
def yes_to_B : ℕ := 8

/-- Theorem stating that the number of Type C people is 16 -/
theorem number_of_type_C_is_16 :
  ∃ (a b c : ℕ),
    a + b + c = total_people ∧
    a + b + (c / 2) = yes_to_A ∧
    b + (c / 2) = yes_to_C ∧
    c / 2 = yes_to_B ∧
    c = 16 := by
  sorry

end number_of_type_C_is_16_l1594_159429


namespace circular_track_length_circular_track_length_is_280_l1594_159440

/-- The length of a circular track given specific running conditions -/
theorem circular_track_length : ℝ → Prop :=
  fun track_length =>
    ∀ (brenda_speed jim_speed : ℝ),
      brenda_speed > 0 ∧ jim_speed > 0 →
      ∃ (first_meet_time second_meet_time : ℝ),
        first_meet_time > 0 ∧ second_meet_time > first_meet_time ∧
        brenda_speed * first_meet_time = 120 ∧
        jim_speed * second_meet_time = 300 ∧
        (brenda_speed * first_meet_time + jim_speed * first_meet_time = track_length / 2) ∧
        (brenda_speed * second_meet_time + jim_speed * second_meet_time = track_length) →
        track_length = 280

/-- The circular track length is 280 meters -/
theorem circular_track_length_is_280 : circular_track_length 280 := by
  sorry

end circular_track_length_circular_track_length_is_280_l1594_159440


namespace first_row_dots_l1594_159446

def green_dots_sequence (n : ℕ) : ℕ := 3 * n + 3

theorem first_row_dots : green_dots_sequence 0 = 3 := by sorry

end first_row_dots_l1594_159446


namespace cubic_roots_problem_l1594_159443

-- Define the polynomials p and q
def p (c d x : ℝ) : ℝ := x^3 + c*x + d
def q (c d x : ℝ) : ℝ := x^3 + c*x + d + 360

-- State the theorem
theorem cubic_roots_problem (c d r s : ℝ) : 
  (p c d r = 0 ∧ p c d s = 0 ∧ q c d (r+5) = 0 ∧ q c d (s-4) = 0) → 
  (d = 84 ∨ d = 1260) := by
sorry

end cubic_roots_problem_l1594_159443


namespace carpet_coverage_percentage_l1594_159428

/-- The percentage of a living room floor covered by a rectangular carpet -/
theorem carpet_coverage_percentage 
  (carpet_length : ℝ) 
  (carpet_width : ℝ) 
  (room_area : ℝ) 
  (h1 : carpet_length = 4) 
  (h2 : carpet_width = 9) 
  (h3 : room_area = 120) : 
  (carpet_length * carpet_width) / room_area * 100 = 30 := by
sorry

end carpet_coverage_percentage_l1594_159428


namespace sqrt_sum_equals_9_6_l1594_159452

theorem sqrt_sum_equals_9_6 (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (16 - y^2) = 5) : 
  Real.sqrt (64 - y^2) + Real.sqrt (16 - y^2) = 9.6 := by
  sorry

end sqrt_sum_equals_9_6_l1594_159452


namespace problem_solution_l1594_159417

open Real

theorem problem_solution (α β : ℝ) (h1 : tan α = -1/3) (h2 : cos β = sqrt 5 / 5)
  (h3 : 0 < α) (h4 : α < π) (h5 : 0 < β) (h6 : β < π) :
  (tan (α + β) = 1) ∧
  (∃ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) = sqrt 5) ∧
  (∃ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) = -sqrt 5) ∧
  (∀ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) ≤ sqrt 5) ∧
  (∀ (x : ℝ), -sqrt 5 ≤ sqrt 2 * sin (x - α) + cos (x + β)) := by
  sorry


end problem_solution_l1594_159417


namespace blue_butterflies_count_l1594_159437

-- Define the variables
def total_butterflies : ℕ := 11
def black_butterflies : ℕ := 5

-- Define the theorem
theorem blue_butterflies_count :
  ∃ (blue yellow : ℕ),
    blue = 2 * yellow ∧
    blue + yellow + black_butterflies = total_butterflies ∧
    blue = 4 := by
  sorry

end blue_butterflies_count_l1594_159437


namespace union_of_sets_l1594_159412

-- Define the sets A and B
def A (a : ℤ) : Set ℤ := {|a + 1|, 3, 5}
def B (a : ℤ) : Set ℤ := {2 * a + 1, a^2 + 2 * a, a^2 + 2 * a - 1}

-- Define the theorem
theorem union_of_sets :
  ∃ a : ℤ, (A a ∩ B a = {2, 3}) → (A a ∪ B a = {-5, 2, 3, 5}) :=
by
  sorry

end union_of_sets_l1594_159412


namespace relationship_abc_l1594_159427

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sqrt 0.6 → 
  b = Real.rpow 0.6 (1/3) → 
  c = Real.log 3 / Real.log 0.6 → 
  c < a ∧ a < b := by
  sorry

end relationship_abc_l1594_159427


namespace intersection_of_M_and_N_l1594_159447

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = 2 * Real.sin x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {y | 0 < y ∧ y ≤ 2} := by sorry

end intersection_of_M_and_N_l1594_159447


namespace luca_pizza_ingredients_l1594_159466

/-- Calculates the required amount of milk and oil for a given amount of flour in Luca's pizza dough recipe. -/
def pizza_ingredients (flour : ℚ) : ℚ × ℚ :=
  let milk_ratio : ℚ := 70 / 350
  let oil_ratio : ℚ := 30 / 350
  (flour * milk_ratio, flour * oil_ratio)

/-- Proves that for 1050 mL of flour, Luca needs 210 mL of milk and 90 mL of oil. -/
theorem luca_pizza_ingredients : pizza_ingredients 1050 = (210, 90) := by
  sorry

end luca_pizza_ingredients_l1594_159466


namespace remaining_soup_feeds_16_adults_l1594_159407

-- Define the problem parameters
def total_cans : ℕ := 8
def adults_per_can : ℕ := 4
def children_per_can : ℕ := 6
def children_fed : ℕ := 24

-- Theorem statement
theorem remaining_soup_feeds_16_adults :
  ∃ (cans_for_children : ℕ) (remaining_cans : ℕ),
    cans_for_children * children_per_can = children_fed ∧
    remaining_cans = total_cans - cans_for_children ∧
    remaining_cans * adults_per_can = 16 :=
by sorry

end remaining_soup_feeds_16_adults_l1594_159407


namespace almond_croissant_price_l1594_159480

def white_bread_price : ℝ := 3.50
def baguette_price : ℝ := 1.50
def sourdough_price : ℝ := 4.50
def total_spent : ℝ := 78.00
def num_weeks : ℕ := 4

def weekly_bread_cost : ℝ := 2 * white_bread_price + baguette_price + 2 * sourdough_price

theorem almond_croissant_price :
  ∃ (croissant_price : ℝ),
    croissant_price * num_weeks + weekly_bread_cost * num_weeks = total_spent ∧
    croissant_price = 8.00 := by
  sorry

end almond_croissant_price_l1594_159480


namespace impossibility_of_tiling_l1594_159464

/-- Represents a tetromino shape -/
inductive TetrominoShape
  | T
  | L
  | I

/-- Represents a 10x10 chessboard -/
def Chessboard := Fin 10 → Fin 10 → Bool

/-- Checks if a given tetromino shape can tile the chessboard -/
def can_tile (shape : TetrominoShape) (board : Chessboard) : Prop :=
  ∃ (tiling : Nat → Nat → Nat → Nat → Bool),
    ∀ (i j : Fin 10), board i j = true ↔ 
      ∃ (x y : Nat), tiling x y i j = true

theorem impossibility_of_tiling (shape : TetrominoShape) :
  ¬∃ (board : Chessboard), can_tile shape board := by
  sorry

#check impossibility_of_tiling TetrominoShape.T
#check impossibility_of_tiling TetrominoShape.L
#check impossibility_of_tiling TetrominoShape.I

end impossibility_of_tiling_l1594_159464


namespace different_counting_units_for_equal_decimals_l1594_159411

-- Define the concept of a decimal number
structure Decimal where
  value : ℚ
  decimalPlaces : ℕ

-- Define the concept of a counting unit
def countingUnit (d : Decimal) : ℚ := 1 / (10 ^ d.decimalPlaces)

-- Define equality for decimals based on their value
def decimalEqual (d1 d2 : Decimal) : Prop := d1.value = d2.value

-- Theorem statement
theorem different_counting_units_for_equal_decimals :
  ∃ (d1 d2 : Decimal), decimalEqual d1 d2 ∧ countingUnit d1 ≠ countingUnit d2 := by
  sorry

end different_counting_units_for_equal_decimals_l1594_159411


namespace delta_y_value_l1594_159444

/-- The function f(x) = x² + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: For f(x) = x² + 1, when x = 2 and Δx = 0.1, Δy = 0.41 -/
theorem delta_y_value (x : ℝ) (Δx : ℝ) (h1 : x = 2) (h2 : Δx = 0.1) :
  f (x + Δx) - f x = 0.41 := by
  sorry


end delta_y_value_l1594_159444


namespace science_class_end_time_l1594_159462

-- Define the schedule as a list of durations in minutes
def class_schedule : List ℕ := [60, 90, 25, 45, 15, 75]

-- Function to calculate the end time given a start time and a list of durations
def calculate_end_time (start_time : ℕ) (schedule : List ℕ) : ℕ :=
  start_time + schedule.sum

-- Theorem statement
theorem science_class_end_time :
  calculate_end_time 720 class_schedule = 1030 := by
  sorry

-- Note: 720 minutes is 12:00 pm, 1030 minutes is 5:10 pm

end science_class_end_time_l1594_159462
