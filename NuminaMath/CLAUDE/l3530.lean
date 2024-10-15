import Mathlib

namespace NUMINAMATH_CALUDE_triangle_exists_l3530_353033

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_exists : can_form_triangle 8 6 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_exists_l3530_353033


namespace NUMINAMATH_CALUDE_shiela_colors_l3530_353086

theorem shiela_colors (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) :
  total_blocks / blocks_per_color = 7 :=
by sorry

end NUMINAMATH_CALUDE_shiela_colors_l3530_353086


namespace NUMINAMATH_CALUDE_appetizers_needed_l3530_353034

/-- Represents the number of appetizers per guest -/
def appetizers_per_guest : ℕ := 6

/-- Represents the number of guests -/
def number_of_guests : ℕ := 30

/-- Represents the number of dozens of deviled eggs prepared -/
def dozens_deviled_eggs : ℕ := 3

/-- Represents the number of dozens of pigs in a blanket prepared -/
def dozens_pigs_in_blanket : ℕ := 2

/-- Represents the number of dozens of kebabs prepared -/
def dozens_kebabs : ℕ := 2

/-- Represents the number of items in a dozen -/
def items_per_dozen : ℕ := 12

/-- Theorem stating that Patsy needs to make 8 more dozen appetizers -/
theorem appetizers_needed : 
  (appetizers_per_guest * number_of_guests - 
   (dozens_deviled_eggs + dozens_pigs_in_blanket + dozens_kebabs) * items_per_dozen) / 
  items_per_dozen = 8 := by
  sorry

end NUMINAMATH_CALUDE_appetizers_needed_l3530_353034


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3530_353032

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3530_353032


namespace NUMINAMATH_CALUDE_planes_parallel_l3530_353008

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perp : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perp : Line → Plane → Prop)

-- Define the objects
variable (α β γ : Plane) (a b : Line)

-- State the theorem
theorem planes_parallel 
  (h1 : parallel α γ)
  (h2 : parallel β γ)
  (h3 : line_perp a α)
  (h4 : line_perp b β)
  (h5 : line_parallel a b) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l3530_353008


namespace NUMINAMATH_CALUDE_max_sine_cosine_function_l3530_353017

theorem max_sine_cosine_function (a b : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x ≤ 4) ∧
  (a * Real.sin (π/3) + b * Real.cos (π/3) = 4) →
  a / b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_sine_cosine_function_l3530_353017


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3530_353011

theorem shaded_area_calculation (π : Real) :
  let semicircle_area := π * 2^2 / 2
  let quarter_circle_area := π * 1^2 / 4
  semicircle_area - 2 * quarter_circle_area = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3530_353011


namespace NUMINAMATH_CALUDE_no_solution_condition_l3530_353062

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (x / (x - 3) + (3 * a) / (3 - x) ≠ 2 * a)) ↔ (a = 1 ∨ a = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3530_353062


namespace NUMINAMATH_CALUDE_stating_escalator_steps_l3530_353002

/-- Represents the total number of steps on an escalator -/
def total_steps : ℕ := 40

/-- Represents the number of steps I ascend on the moving escalator -/
def my_steps : ℕ := 20

/-- Represents the time I take to ascend the escalator in seconds -/
def my_time : ℕ := 60

/-- Represents the number of steps my wife ascends on the moving escalator -/
def wife_steps : ℕ := 16

/-- Represents the time my wife takes to ascend the escalator in seconds -/
def wife_time : ℕ := 72

/-- 
Theorem stating that the total number of steps on the escalator is 40,
given the conditions about my ascent and my wife's ascent.
-/
theorem escalator_steps : 
  (total_steps - my_steps) / my_time = (total_steps - wife_steps) / wife_time :=
sorry

end NUMINAMATH_CALUDE_stating_escalator_steps_l3530_353002


namespace NUMINAMATH_CALUDE_basketball_team_selection_l3530_353087

theorem basketball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 880 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l3530_353087


namespace NUMINAMATH_CALUDE_unique_solution_l3530_353050

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Given a five-digit number, returns all possible four-digit numbers
    that can be formed by removing one digit -/
def removeSingleDigit (n : FiveDigitNumber) : Set FourDigitNumber :=
  sorry

/-- The property that defines our solution -/
def isSolution (n : FiveDigitNumber) : Prop :=
  ∃ (m : FourDigitNumber), m ∈ removeSingleDigit n ∧ n.val + m.val = 54321

/-- Theorem stating that 49383 is the unique solution -/
theorem unique_solution :
  ∃! (n : FiveDigitNumber), isSolution n ∧ n.val = 49383 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3530_353050


namespace NUMINAMATH_CALUDE_triangle_centroid_inequality_locus_is_circle_l3530_353005

open Real

-- Define a triangle with vertices A, B, C and centroid G
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  is_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define distance squared between two points
def dist_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the theorem
theorem triangle_centroid_inequality (t : Triangle) (M : ℝ × ℝ) :
  dist_sq M t.A + dist_sq M t.B + dist_sq M t.C ≥ 
  dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C ∧
  (dist_sq M t.A + dist_sq M t.B + dist_sq M t.C = 
   dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C ↔ M = t.G) :=
sorry

-- Define the locus of points
def locus (t : Triangle) (k : ℝ) : Set (ℝ × ℝ) :=
  {M | dist_sq M t.A + dist_sq M t.B + dist_sq M t.C = k}

-- Define the theorem for the locus
theorem locus_is_circle (t : Triangle) (k : ℝ) 
  (h : k > dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C) :
  ∃ (r : ℝ), r > 0 ∧ locus t k = {M | dist_sq M t.G = r^2} ∧
  r^2 = (k - (dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C)) / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_centroid_inequality_locus_is_circle_l3530_353005


namespace NUMINAMATH_CALUDE_remainder_problem_l3530_353047

theorem remainder_problem : (8 * 7^19 + 1^19) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3530_353047


namespace NUMINAMATH_CALUDE_water_left_for_fourth_neighborhood_l3530_353045

-- Define the total capacity of the water tower
def total_capacity : ℕ := 1200

-- Define the water usage of the first neighborhood
def first_neighborhood_usage : ℕ := 150

-- Define the water usage of the second neighborhood
def second_neighborhood_usage : ℕ := 2 * first_neighborhood_usage

-- Define the water usage of the third neighborhood
def third_neighborhood_usage : ℕ := second_neighborhood_usage + 100

-- Define the total usage of the first three neighborhoods
def total_usage : ℕ := first_neighborhood_usage + second_neighborhood_usage + third_neighborhood_usage

-- Theorem to prove
theorem water_left_for_fourth_neighborhood :
  total_capacity - total_usage = 350 := by sorry

end NUMINAMATH_CALUDE_water_left_for_fourth_neighborhood_l3530_353045


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3530_353036

theorem min_value_of_expression (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  1 / m + 2 / n ≥ 4 ∧ (1 / m + 2 / n = 4 ↔ n = 2 * m ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3530_353036


namespace NUMINAMATH_CALUDE_largest_square_with_four_lattice_points_l3530_353003

/-- A point (x, y) is a lattice point if both x and y are integers. -/
def isLatticePoint (p : ℝ × ℝ) : Prop :=
  Int.floor p.1 = p.1 ∧ Int.floor p.2 = p.2

/-- A square contains exactly four lattice points in its interior. -/
def squareContainsFourLatticePoints (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ), p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  isLatticePoint p₁ ∧ isLatticePoint p₂ ∧ isLatticePoint p₃ ∧ isLatticePoint p₄ ∧
  (∀ p ∈ s, isLatticePoint p → p = p₁ ∨ p = p₂ ∨ p = p₃ ∨ p = p₄)

/-- The theorem statement -/
theorem largest_square_with_four_lattice_points :
  ∃ (s : Set (ℝ × ℝ)), squareContainsFourLatticePoints s ∧
  (∀ (t : Set (ℝ × ℝ)), squareContainsFourLatticePoints t → MeasureTheory.volume s ≥ MeasureTheory.volume t) ∧
  MeasureTheory.volume s = 8 :=
sorry

end NUMINAMATH_CALUDE_largest_square_with_four_lattice_points_l3530_353003


namespace NUMINAMATH_CALUDE_hyperbola_focal_property_l3530_353038

/-- The hyperbola with equation x^2 - y^2/9 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2/9 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_property (P : ℝ × ℝ) (h_P : P ∈ Hyperbola) 
    (h_dist : distance P F₁ = 5) : 
  distance P F₂ = 3 ∨ distance P F₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_property_l3530_353038


namespace NUMINAMATH_CALUDE_min_value_theorem_l3530_353000

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 ∧ 
  ∃ y : ℝ, (y^2 + 9) / Real.sqrt (y^2 + 5) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3530_353000


namespace NUMINAMATH_CALUDE_candy_distribution_l3530_353021

theorem candy_distribution (total_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) :
  total_candies = 36 →
  num_friends = 9 →
  candies_per_friend = 4 →
  total_candies = num_friends * candies_per_friend :=
by
  sorry

#check candy_distribution

end NUMINAMATH_CALUDE_candy_distribution_l3530_353021


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3530_353010

/-- A parabola with equation y = x^2 - 8x + m has its vertex on the x-axis if and only if m = 16 -/
theorem parabola_vertex_on_x_axis (m : ℝ) :
  (∃ x, x^2 - 8*x + m = 0 ∧ ∀ y, y^2 - 8*y + m ≥ x^2 - 8*x + m) ↔ m = 16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3530_353010


namespace NUMINAMATH_CALUDE_ellipse_quadrant_area_diff_zero_l3530_353096

/-- Definition of an ellipse with center (h, k) and parameters a, b, c -/
def Ellipse (h k a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - h)^2 / a + (p.2 - k)^2 / b = c}

/-- Areas of the ellipse in each quadrant -/
def QuadrantAreas (e : Set (ℝ × ℝ)) : ℝ × ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The difference of areas in alternating quadrants is zero -/
theorem ellipse_quadrant_area_diff_zero
  (h k a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let e := Ellipse h k a b c
  let (R1, R2, R3, R4) := QuadrantAreas e
  R1 - R2 + R3 - R4 = 0 := by sorry


end NUMINAMATH_CALUDE_ellipse_quadrant_area_diff_zero_l3530_353096


namespace NUMINAMATH_CALUDE_menelaus_theorem_l3530_353066

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the intersection points
def F (t : Triangle) (l : Line) : ℝ × ℝ := sorry
def D (t : Triangle) (l : Line) : ℝ × ℝ := sorry
def E (t : Triangle) (l : Line) : ℝ × ℝ := sorry

-- Define the ratio function
def ratio (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem menelaus_theorem (t : Triangle) (l : Line) :
  ratio (F t l) t.A t.B * ratio (D t l) t.B t.C * ratio (E t l) t.C t.A = -1 := by sorry

end NUMINAMATH_CALUDE_menelaus_theorem_l3530_353066


namespace NUMINAMATH_CALUDE_ny_mets_fans_l3530_353052

/-- The number of NY Mets fans in a town with specific fan ratios --/
theorem ny_mets_fans (total : ℕ) (y m r d : ℚ) : 
  total = 780 →
  y / m = 3 / 2 →
  m / r = 4 / 5 →
  r / d = 7 / (3/2) →
  y + m + r + d = total →
  ⌊m⌋ = 178 := by
  sorry

end NUMINAMATH_CALUDE_ny_mets_fans_l3530_353052


namespace NUMINAMATH_CALUDE_pop_spending_proof_l3530_353099

/-- The amount of money Pop spent on cereal -/
def pop_spending : ℝ := 15

/-- The amount of money Crackle spent on cereal -/
def crackle_spending : ℝ := 3 * pop_spending

/-- The amount of money Snap spent on cereal -/
def snap_spending : ℝ := 2 * crackle_spending

/-- The total amount spent on cereal -/
def total_spending : ℝ := 150

theorem pop_spending_proof :
  pop_spending + crackle_spending + snap_spending = total_spending ∧
  pop_spending = 15 := by
  sorry

end NUMINAMATH_CALUDE_pop_spending_proof_l3530_353099


namespace NUMINAMATH_CALUDE_combination_98_96_l3530_353053

theorem combination_98_96 : Nat.choose 98 96 = 4753 := by
  sorry

end NUMINAMATH_CALUDE_combination_98_96_l3530_353053


namespace NUMINAMATH_CALUDE_average_marks_l3530_353030

theorem average_marks (total_subjects : ℕ) (subjects_avg : ℕ) (last_subject_mark : ℕ) :
  total_subjects = 6 →
  subjects_avg = 74 →
  last_subject_mark = 110 →
  (subjects_avg * (total_subjects - 1) + last_subject_mark) / total_subjects = 80 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_l3530_353030


namespace NUMINAMATH_CALUDE_asian_games_mascot_sales_l3530_353064

/-- Represents the sales situation of Asian Games mascots -/
theorem asian_games_mascot_sales 
  (initial_sales : ℕ) 
  (total_sales_next_two_days : ℕ) 
  (growth_rate : ℝ) :
  initial_sales = 5000 →
  total_sales_next_two_days = 30000 →
  (initial_sales : ℝ) * (1 + growth_rate) + (initial_sales : ℝ) * (1 + growth_rate)^2 = total_sales_next_two_days :=
by sorry

end NUMINAMATH_CALUDE_asian_games_mascot_sales_l3530_353064


namespace NUMINAMATH_CALUDE_factorization_equality_l3530_353098

theorem factorization_equality (x : ℝ) : x * (x - 3) + (3 - x) = (x - 3) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3530_353098


namespace NUMINAMATH_CALUDE_triangle_area_l3530_353093

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  C = π / 6 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3530_353093


namespace NUMINAMATH_CALUDE_not_necessarily_divisible_by_48_l3530_353028

theorem not_necessarily_divisible_by_48 (k : ℤ) :
  let n := k * (k + 1) * (k + 2) * (k + 3)
  ∃ (n : ℤ), (8 ∣ n) ∧ ¬(48 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_divisible_by_48_l3530_353028


namespace NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l3530_353079

theorem one_fourth_in_one_eighth : (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_in_one_eighth_l3530_353079


namespace NUMINAMATH_CALUDE_inverse_relationship_scenarios_l3530_353058

/-- Represents a scenario with two variables that may have an inverse relationship -/
structure Scenario where
  x : ℝ
  y : ℝ
  k : ℝ
  h_k_nonzero : k ≠ 0

/-- Checks if a scenario satisfies the inverse relationship y = k/x -/
def has_inverse_relationship (s : Scenario) : Prop :=
  s.y = s.k / s.x

/-- Rectangle scenario with fixed area -/
def rectangle_scenario (area x y : ℝ) (h : area ≠ 0) : Scenario where
  x := x
  y := y
  k := area
  h_k_nonzero := h

/-- Village land scenario with fixed total arable land -/
def village_land_scenario (total_land n S : ℝ) (h : total_land ≠ 0) : Scenario where
  x := n
  y := S
  k := total_land
  h_k_nonzero := h

/-- Car travel scenario with fixed speed -/
def car_travel_scenario (speed s t : ℝ) (h : speed ≠ 0) : Scenario where
  x := t
  y := s
  k := speed
  h_k_nonzero := h

theorem inverse_relationship_scenarios 
  (rect : Scenario) 
  (village : Scenario) 
  (car : Scenario) : 
  has_inverse_relationship rect ∧ 
  has_inverse_relationship village ∧ 
  ¬has_inverse_relationship car := by
  sorry

end NUMINAMATH_CALUDE_inverse_relationship_scenarios_l3530_353058


namespace NUMINAMATH_CALUDE_parabola_sum_of_coefficients_l3530_353026

/-- A quadratic function with coefficients p, q, and r -/
def quadratic_function (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem parabola_sum_of_coefficients 
  (p q r : ℝ) 
  (h_vertex : quadratic_function p q r 3 = 4)
  (h_symmetry : ∀ (x : ℝ), quadratic_function p q r (3 + x) = quadratic_function p q r (3 - x))
  (h_point1 : quadratic_function p q r 1 = 10)
  (h_point2 : quadratic_function p q r (-1) = 14) :
  p + q + r = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_of_coefficients_l3530_353026


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3530_353049

/-- A boat traveling downstream with the help of a stream. -/
structure BoatTrip where
  boat_speed : ℝ      -- Speed of the boat in still water (km/hr)
  stream_speed : ℝ    -- Speed of the stream (km/hr)
  time : ℝ            -- Time taken for the trip (hours)
  distance : ℝ        -- Distance traveled (km)

/-- The theorem stating the boat's speed in still water given the conditions. -/
theorem boat_speed_in_still_water (trip : BoatTrip)
  (h1 : trip.stream_speed = 5)
  (h2 : trip.time = 5)
  (h3 : trip.distance = 135) :
  trip.boat_speed = 22 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3530_353049


namespace NUMINAMATH_CALUDE_remainder_97_103_times_7_mod_17_l3530_353023

theorem remainder_97_103_times_7_mod_17 : (97^103 * 7) % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_103_times_7_mod_17_l3530_353023


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3530_353072

theorem polynomial_divisibility (P : ℤ → ℤ) (n : ℤ) 
  (h1 : ∃ k1 : ℤ, P n = 3 * k1)
  (h2 : ∃ k2 : ℤ, P (n + 1) = 3 * k2)
  (h3 : ∃ k3 : ℤ, P (n + 2) = 3 * k3)
  (h_poly : ∀ x y : ℤ, ∃ a b c : ℤ, P (x + y) = P x + a * y + b * y^2 + c * y^3) :
  ∀ m : ℤ, ∃ k : ℤ, P m = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3530_353072


namespace NUMINAMATH_CALUDE_selection_with_both_genders_l3530_353067

/-- The number of ways to select 3 people from a group of 4 male students and 6 female students, 
    such that both male and female students are included. -/
theorem selection_with_both_genders (male_count : Nat) (female_count : Nat) : 
  male_count = 4 → female_count = 6 → 
  (Nat.choose (male_count + female_count) 3 - 
   Nat.choose male_count 3 - 
   Nat.choose female_count 3) = 96 := by
  sorry

end NUMINAMATH_CALUDE_selection_with_both_genders_l3530_353067


namespace NUMINAMATH_CALUDE_range_of_m_l3530_353007

theorem range_of_m (x m : ℝ) : 
  (2 * x - m ≤ 3 ∧ -5 < x ∧ x < 4) ↔ m ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3530_353007


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l3530_353065

theorem cousins_ages_sum (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit ages
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct ages
  (a * b = 20 ∧ c * d = 36) ∨ (a * c = 20 ∧ b * d = 36) ∨ 
  (a * d = 20 ∧ b * c = 36) →  -- product conditions
  a + b + c + d = 21 := by
sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l3530_353065


namespace NUMINAMATH_CALUDE_probability_total_gt_seven_is_five_twelfths_l3530_353081

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of outcomes that result in a total greater than 7 -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing two 6-sided dice -/
def probability_total_gt_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_total_gt_seven_is_five_twelfths :
  probability_total_gt_seven = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_total_gt_seven_is_five_twelfths_l3530_353081


namespace NUMINAMATH_CALUDE_campers_rowing_difference_l3530_353006

theorem campers_rowing_difference (morning afternoon evening : ℕ) 
  (h1 : morning = 33) 
  (h2 : afternoon = 34) 
  (h3 : evening = 10) : 
  afternoon - evening = 24 := by
sorry

end NUMINAMATH_CALUDE_campers_rowing_difference_l3530_353006


namespace NUMINAMATH_CALUDE_inequality_proof_l3530_353039

theorem inequality_proof (x y z a b : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  ((a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) ∨ (a + b = 1 ∧ a > 0 ∧ b > 0)) →
  (1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b)) ≥ 3) ∧
  (x = 1 ∧ y = 1 ∧ z = 1 → 1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b)) = 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3530_353039


namespace NUMINAMATH_CALUDE_problem_solution_l3530_353046

theorem problem_solution :
  (∀ x : ℝ, x^2 = 0 → x = 0) ∧
  (∀ x : ℝ, x^2 < 2*x → x > 0) ∧
  (∀ x : ℝ, x > 2 → x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3530_353046


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l3530_353020

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def is_single_digit (n : ℕ) : Prop := n < 10

theorem largest_digit_divisible_by_6 :
  ∀ M : ℕ, is_single_digit M →
    (is_divisible_by_6 (45670 + M) → M ≤ 8) ∧
    (is_divisible_by_6 (45678)) := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l3530_353020


namespace NUMINAMATH_CALUDE_sandwiches_available_l3530_353040

def initial_sandwiches : ℕ := 23
def sold_out_sandwiches : ℕ := 14

theorem sandwiches_available : initial_sandwiches - sold_out_sandwiches = 9 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_available_l3530_353040


namespace NUMINAMATH_CALUDE_toy_cost_calculation_l3530_353055

theorem toy_cost_calculation (initial_amount : ℕ) (game_cost : ℕ) (num_toys : ℕ) :
  initial_amount = 83 →
  game_cost = 47 →
  num_toys = 9 →
  (initial_amount - game_cost) % num_toys = 0 →
  (initial_amount - game_cost) / num_toys = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_cost_calculation_l3530_353055


namespace NUMINAMATH_CALUDE_tuesday_sales_l3530_353018

theorem tuesday_sales (initial_stock : ℕ) (monday_sales wednesday_sales thursday_sales friday_sales : ℕ)
  (unsold_percentage : ℚ) :
  initial_stock = 700 →
  monday_sales = 50 →
  wednesday_sales = 60 →
  thursday_sales = 48 →
  friday_sales = 40 →
  unsold_percentage = 60 / 100 →
  ∃ (tuesday_sales : ℕ),
    tuesday_sales = 82 ∧
    (initial_stock : ℚ) * (1 - unsold_percentage) =
      monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales :=
by sorry

end NUMINAMATH_CALUDE_tuesday_sales_l3530_353018


namespace NUMINAMATH_CALUDE_calculate_expression_l3530_353089

theorem calculate_expression : -1^4 - 1/4 * (2 - (-3)^2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3530_353089


namespace NUMINAMATH_CALUDE_inverse_f_at_negative_eight_l3530_353097

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - Real.log x / Real.log 3

theorem inverse_f_at_negative_eight (a : ℝ) :
  f a 1 = 1 → f a (3^9) = -8 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_negative_eight_l3530_353097


namespace NUMINAMATH_CALUDE_xw_value_l3530_353059

/-- Triangle XYZ with point W on YZ such that XW is perpendicular to YZ -/
structure TriangleXYZW where
  /-- The length of side XY -/
  XY : ℝ
  /-- The length of side XZ -/
  XZ : ℝ
  /-- The length of XW, where W is on YZ and XW ⟂ YZ -/
  XW : ℝ
  /-- The length of YW -/
  YW : ℝ
  /-- The length of ZW -/
  ZW : ℝ
  /-- XY equals 15 -/
  xy_eq : XY = 15
  /-- XZ equals 26 -/
  xz_eq : XZ = 26
  /-- YW:ZW ratio is 3:4 -/
  yw_zw_ratio : YW / ZW = 3 / 4
  /-- Pythagorean theorem for XYW -/
  pythagoras_xyw : YW ^ 2 = XY ^ 2 - XW ^ 2
  /-- Pythagorean theorem for XZW -/
  pythagoras_xzw : ZW ^ 2 = XZ ^ 2 - XW ^ 2

/-- The main theorem: If the conditions are met, then XW = 42/√7 -/
theorem xw_value (t : TriangleXYZW) : t.XW = 42 / Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_xw_value_l3530_353059


namespace NUMINAMATH_CALUDE_five_digit_numbers_count_correct_l3530_353090

/-- Counts five-digit numbers with specific digit conditions -/
def count_five_digit_numbers : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let all_identical := 9
  let two_different := 1215
  let three_different := 16200
  let four_different := 45360
  let five_different := 27216
  (all_identical, two_different, three_different, four_different, five_different)

/-- The first digit of a five-digit number cannot be zero -/
axiom first_digit_nonzero : ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 → n / 10000 ≠ 0

/-- The sum of all cases equals the total number of five-digit numbers -/
theorem five_digit_numbers_count_correct :
  let (a, b, c, d, e) := count_five_digit_numbers
  a + b + c + d + e = 90000 :=
sorry

end NUMINAMATH_CALUDE_five_digit_numbers_count_correct_l3530_353090


namespace NUMINAMATH_CALUDE_shirley_eggs_l3530_353037

theorem shirley_eggs (initial_eggs : ℕ) (bought_eggs : ℕ) : 
  initial_eggs = 98 → bought_eggs = 8 → initial_eggs + bought_eggs = 106 := by
  sorry

end NUMINAMATH_CALUDE_shirley_eggs_l3530_353037


namespace NUMINAMATH_CALUDE_triangle_ratio_l3530_353025

/-- Given a triangle ABC with angle A = 60°, side b = 1, and area = √3,
    prove that (a+b+c)/(sin A + sin B + sin C) = 2√39/3 -/
theorem triangle_ratio (a b c A B C : ℝ) : 
  A = π/3 → 
  b = 1 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3530_353025


namespace NUMINAMATH_CALUDE_cubic_sum_equals_twenty_l3530_353027

theorem cubic_sum_equals_twenty (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) :
  x^3 + 3*y^2 + 3*z^2 + 3*x*y*z = 20 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_twenty_l3530_353027


namespace NUMINAMATH_CALUDE_omega_squared_plus_omega_plus_one_eq_zero_l3530_353073

theorem omega_squared_plus_omega_plus_one_eq_zero :
  let ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I
  ω^2 + ω + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_omega_squared_plus_omega_plus_one_eq_zero_l3530_353073


namespace NUMINAMATH_CALUDE_inequality_proof_l3530_353057

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_inequality : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3530_353057


namespace NUMINAMATH_CALUDE_not_in_second_quadrant_l3530_353012

/-- A linear function f(x) = x - 1 -/
def f (x : ℝ) : ℝ := x - 1

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The linear function f(x) = x - 1 does not pass through the second quadrant -/
theorem not_in_second_quadrant :
  ∀ x : ℝ, ¬(second_quadrant x (f x)) :=
by sorry

end NUMINAMATH_CALUDE_not_in_second_quadrant_l3530_353012


namespace NUMINAMATH_CALUDE_exists_non_intersecting_circle_l3530_353041

-- Define the circular billiard table
def CircularBilliardTable := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1}

-- Define a trajectory of the ball
def Trajectory := Set (ℝ × ℝ)

-- Define the property of a trajectory following the laws of reflection
def FollowsReflectionLaws (t : Trajectory) : Prop := sorry

-- Define a circle inside the table
def InsideCircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2 ∧ p ∈ CircularBilliardTable}

-- The main theorem
theorem exists_non_intersecting_circle :
  ∀ (start : ℝ × ℝ) (t : Trajectory),
    start ∈ CircularBilliardTable →
    FollowsReflectionLaws t →
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      InsideCircle center radius ⊆ CircularBilliardTable ∧
      (InsideCircle center radius ∩ t = ∅) :=
by
  sorry

end NUMINAMATH_CALUDE_exists_non_intersecting_circle_l3530_353041


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3530_353095

theorem no_positive_integer_solutions :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 3 * x^2 + 2 * x + 2 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3530_353095


namespace NUMINAMATH_CALUDE_f_is_integer_valued_l3530_353074

/-- The polynomial f(x) = (1/5)x^5 + (1/2)x^4 + (1/3)x^3 - (1/30)x -/
def f (x : ℚ) : ℚ := (1/5) * x^5 + (1/2) * x^4 + (1/3) * x^3 - (1/30) * x

/-- Theorem stating that f(x) is an integer-valued polynomial -/
theorem f_is_integer_valued : ∀ (x : ℤ), ∃ (y : ℤ), f x = y := by
  sorry

end NUMINAMATH_CALUDE_f_is_integer_valued_l3530_353074


namespace NUMINAMATH_CALUDE_inequality_proof_l3530_353076

theorem inequality_proof (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3530_353076


namespace NUMINAMATH_CALUDE_max_snack_bars_l3530_353085

/-- Represents the number of snack bars in a pack -/
inductive PackSize
  | single : PackSize
  | twin : PackSize
  | four : PackSize

/-- Represents the price of a pack of snack bars -/
def price (p : PackSize) : ℚ :=
  match p with
  | PackSize.single => 1
  | PackSize.twin => 5/2
  | PackSize.four => 4

/-- Represents the number of snack bars in a pack -/
def bars_in_pack (p : PackSize) : ℕ :=
  match p with
  | PackSize.single => 1
  | PackSize.twin => 2
  | PackSize.four => 4

/-- The budget available for purchasing snack bars -/
def budget : ℚ := 10

/-- A purchase combination is represented as a function from PackSize to ℕ -/
def PurchaseCombination := PackSize → ℕ

/-- The total cost of a purchase combination -/
def total_cost (c : PurchaseCombination) : ℚ :=
  (c PackSize.single) * (price PackSize.single) +
  (c PackSize.twin) * (price PackSize.twin) +
  (c PackSize.four) * (price PackSize.four)

/-- The total number of snack bars in a purchase combination -/
def total_bars (c : PurchaseCombination) : ℕ :=
  (c PackSize.single) * (bars_in_pack PackSize.single) +
  (c PackSize.twin) * (bars_in_pack PackSize.twin) +
  (c PackSize.four) * (bars_in_pack PackSize.four)

/-- A purchase combination is valid if its total cost is within the budget -/
def is_valid_combination (c : PurchaseCombination) : Prop :=
  total_cost c ≤ budget

theorem max_snack_bars :
  ∃ (max : ℕ), 
    (∃ (c : PurchaseCombination), is_valid_combination c ∧ total_bars c = max) ∧
    (∀ (c : PurchaseCombination), is_valid_combination c → total_bars c ≤ max) ∧
    max = 10 :=
  sorry

end NUMINAMATH_CALUDE_max_snack_bars_l3530_353085


namespace NUMINAMATH_CALUDE_arithmetic_sequence_slope_l3530_353016

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2

/-- The slope of the line passing through P(n, a_n) and Q(n+2, a_{n+2}) is 4 -/
theorem arithmetic_sequence_slope (seq : ArithmeticSequence) 
    (h1 : seq.S 2 = 10) (h2 : seq.S 5 = 55) :
    ∀ n : ℕ+, (seq.a (n + 2) - seq.a n) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_slope_l3530_353016


namespace NUMINAMATH_CALUDE_min_distance_tan_intersection_l3530_353001

theorem min_distance_tan_intersection (a : ℝ) : 
  let f (x : ℝ) := Real.tan (2 * x - π / 3)
  let g (x : ℝ) := -a
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    f x₁ = g x₁ ∧ 
    f x₂ = g x₂ ∧
    ∀ (y : ℝ), x₁ < y ∧ y < x₂ → f y ≠ g y ∧
    x₂ - x₁ = π / 2 ∧
    ∀ (z₁ z₂ : ℝ), (f z₁ = g z₁ ∧ f z₂ = g z₂ ∧ z₁ < z₂) → z₂ - z₁ ≥ π / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_tan_intersection_l3530_353001


namespace NUMINAMATH_CALUDE_divisible_by_91_l3530_353091

theorem divisible_by_91 (n : ℕ) : ∃ k : ℤ, 9^(n+2) + 10^(2*n+1) = 91 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_91_l3530_353091


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3530_353044

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.5)) 
  (h2 : P = G * (1 - 0.2)) : 
  R = G * (1 + 0.6) := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3530_353044


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3530_353042

theorem inequality_solution_set (x : ℝ) : 
  (3/20 : ℝ) + |x - 9/40| + |x + 1/8| < (1/2 : ℝ) ↔ -3/40 < x ∧ x < 11/40 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3530_353042


namespace NUMINAMATH_CALUDE_circle_max_cube_root_sum_l3530_353061

theorem circle_max_cube_root_sum (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  ∀ a b : ℝ, a^2 + b^2 = 1 → 
  Real.sqrt (|x|^3 + |y|^3) ≤ Real.sqrt (2 * Real.sqrt 2 + 1) / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_cube_root_sum_l3530_353061


namespace NUMINAMATH_CALUDE_total_tabs_is_sixty_l3530_353024

/-- Calculates the total number of tabs opened across all browsers -/
def totalTabs (numBrowsers : ℕ) (windowsPerBrowser : ℕ) (tabsPerWindow : ℕ) : ℕ :=
  numBrowsers * windowsPerBrowser * tabsPerWindow

/-- Theorem: Given the specified conditions, the total number of tabs is 60 -/
theorem total_tabs_is_sixty :
  totalTabs 2 3 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_tabs_is_sixty_l3530_353024


namespace NUMINAMATH_CALUDE_max_carry_weight_is_1001_l3530_353063

/-- Represents the loader with a waggon and a cart -/
structure Loader :=
  (waggon_capacity : ℕ)
  (cart_capacity : ℕ)

/-- Represents the sand sacks in the storehouse -/
structure Storehouse :=
  (total_weight : ℕ)
  (max_sack_weight : ℕ)

/-- The maximum weight of sand the loader can carry -/
def max_carry_weight (l : Loader) (s : Storehouse) : ℕ :=
  l.waggon_capacity + l.cart_capacity

/-- Theorem stating the maximum weight the loader can carry -/
theorem max_carry_weight_is_1001 (l : Loader) (s : Storehouse) :
  l.waggon_capacity = 1000 →
  l.cart_capacity = 1 →
  s.total_weight > 1001 →
  s.max_sack_weight ≤ 1 →
  max_carry_weight l s = 1001 :=
by sorry

end NUMINAMATH_CALUDE_max_carry_weight_is_1001_l3530_353063


namespace NUMINAMATH_CALUDE_inequality_proof_l3530_353088

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3530_353088


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l3530_353084

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The fixed point through which the line passes -/
def fixedPoint : ℝ × ℝ := (0, 1)

/-- The equation of the trajectory of the midpoint of the chord -/
def trajectoryEquation (x y : ℝ) : Prop := 4*x^2 - y^2 + y = 0

/-- Theorem stating that the trajectory equation is correct for the given conditions -/
theorem trajectory_of_midpoint (x y : ℝ) :
  (∃ (k : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    y₁ = k*x₁ + fixedPoint.2 ∧ y₂ = k*x₂ + fixedPoint.2 ∧
    x = (x₁ + x₂)/2 ∧ y = (y₁ + y₂)/2) →
  trajectoryEquation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l3530_353084


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_l3530_353048

theorem sqrt_2_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_l3530_353048


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3530_353019

/-- Calculates the actual distance between Stockholm and Uppsala based on map measurements and scales. -/
def actual_distance (map_distance : ℝ) (first_part : ℝ) (scale1 : ℝ) (scale2 : ℝ) : ℝ :=
  first_part * scale1 + (map_distance - first_part) * scale2

/-- Theorem stating that the actual distance between Stockholm and Uppsala is 375 km. -/
theorem stockholm_uppsala_distance :
  let map_distance : ℝ := 45
  let first_part : ℝ := 15
  let scale1 : ℝ := 5
  let scale2 : ℝ := 10
  actual_distance map_distance first_part scale1 scale2 = 375 := by
  sorry


end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3530_353019


namespace NUMINAMATH_CALUDE_function_range_l3530_353071

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem function_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_function_range_l3530_353071


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3530_353070

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.0000023 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.3 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3530_353070


namespace NUMINAMATH_CALUDE_percentage_problem_l3530_353080

theorem percentage_problem (P : ℝ) : 
  0.15 * 0.30 * (P / 100) * 4000 = 90 → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3530_353080


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3530_353004

theorem roots_of_polynomial (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c ↔ x = -1 ∨ x = 1) →
  a = -6 ∧ b = -1 ∧ c = -4 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3530_353004


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_four_l3530_353009

theorem sum_of_x_and_y_is_four (x y : ℝ) 
  (eq1 : 4 * x - y = 3) 
  (eq2 : x + 6 * y = 17) : 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_four_l3530_353009


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3530_353083

theorem no_prime_roots_for_quadratic : 
  ¬ ∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (p : ℤ) + q = 59 ∧
    (p : ℤ) * q = k ∧
    ∀ (x : ℤ), x^2 - 59*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3530_353083


namespace NUMINAMATH_CALUDE_space_diagonals_count_l3530_353092

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Definition of a space diagonal in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem stating the number of space diagonals in the given polyhedron -/
theorem space_diagonals_count (Q : ConvexPolyhedron) 
  (h1 : Q.vertices = 30)
  (h2 : Q.edges = 58)
  (h3 : Q.faces = 36)
  (h4 : Q.triangular_faces = 26)
  (h5 : Q.quadrilateral_faces = 10)
  (h6 : Q.triangular_faces + Q.quadrilateral_faces = Q.faces) :
  space_diagonals Q = 357 := by
  sorry


end NUMINAMATH_CALUDE_space_diagonals_count_l3530_353092


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3530_353031

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔
  (a < 2 ∧ a ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3530_353031


namespace NUMINAMATH_CALUDE_chosen_number_proof_l3530_353014

theorem chosen_number_proof : ∃ x : ℝ, (x / 5) - 154 = 6 ∧ x = 800 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l3530_353014


namespace NUMINAMATH_CALUDE_third_number_proof_l3530_353043

def mean (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem third_number_proof (x : ℕ) (y : ℕ) :
  mean [28, x, y, 78, 104] = 90 →
  mean [128, 255, 511, 1023, x] = 423 →
  y = 42 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l3530_353043


namespace NUMINAMATH_CALUDE_smallest_base_for_78_l3530_353056

theorem smallest_base_for_78 :
  ∃ (b : ℕ), b > 0 ∧ b^2 ≤ 78 ∧ 78 < b^3 ∧ ∀ (x : ℕ), x > 0 ∧ x^2 ≤ 78 ∧ 78 < x^3 → b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_78_l3530_353056


namespace NUMINAMATH_CALUDE_john_volunteer_frequency_l3530_353069

/-- The number of hours John volunteers per year -/
def annual_hours : ℕ := 72

/-- The number of hours per volunteering session -/
def hours_per_session : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The number of times John volunteers per month -/
def volunteer_times_per_month : ℚ :=
  (annual_hours / hours_per_session : ℚ) / months_per_year

theorem john_volunteer_frequency :
  volunteer_times_per_month = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_volunteer_frequency_l3530_353069


namespace NUMINAMATH_CALUDE_restaurant_bill_division_l3530_353035

theorem restaurant_bill_division (total_bill : ℝ) (num_people : ℕ) (individual_share : ℝ) :
  total_bill = 135 →
  num_people = 3 →
  individual_share = total_bill / num_people →
  individual_share = 45 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_division_l3530_353035


namespace NUMINAMATH_CALUDE_cookie_count_l3530_353015

theorem cookie_count (x y : ℕ) (hx : x = 137) (hy : y = 251) : x * y = 34387 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l3530_353015


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3530_353029

theorem multiply_mixed_number : 7 * (9 + 2/5) = 329/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3530_353029


namespace NUMINAMATH_CALUDE_bowling_tournament_sequences_l3530_353094

/-- A tournament with 6 players and 5 matches -/
structure Tournament :=
  (num_players : Nat)
  (num_matches : Nat)
  (outcomes_per_match : Nat)

/-- The number of possible prize distribution sequences in the tournament -/
def prize_sequences (t : Tournament) : Nat :=
  t.outcomes_per_match ^ t.num_matches

/-- Theorem stating that for a tournament with 6 players, 5 matches, and 2 possible outcomes per match,
    the number of possible prize distribution sequences is 32 -/
theorem bowling_tournament_sequences :
  ∀ t : Tournament, t.num_players = 6 → t.num_matches = 5 → t.outcomes_per_match = 2 →
  prize_sequences t = 32 := by
  sorry

end NUMINAMATH_CALUDE_bowling_tournament_sequences_l3530_353094


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l3530_353077

-- Define the expression
def expression (a b : ℝ) : ℝ := -4 * a^2 + b^2

-- Theorem: The expression can be factored using the difference of squares formula
theorem difference_of_squares_factorization (a b : ℝ) :
  ∃ (x y : ℝ), expression a b = (x + y) * (x - y) :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l3530_353077


namespace NUMINAMATH_CALUDE_min_roots_in_interval_l3530_353051

/-- A function satisfying the given symmetry conditions -/
def SymmetricFunction (g : ℝ → ℝ) : Prop :=
  (∀ x, g (3 + x) = g (3 - x)) ∧ (∀ x, g (5 + x) = g (5 - x))

/-- The theorem stating the minimum number of roots in the given interval -/
theorem min_roots_in_interval
  (g : ℝ → ℝ)
  (h_symmetric : SymmetricFunction g)
  (h_g1_zero : g 1 = 0) :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, g x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧
    roots.card ≥ 250 :=
  sorry

end NUMINAMATH_CALUDE_min_roots_in_interval_l3530_353051


namespace NUMINAMATH_CALUDE_kate_money_left_l3530_353075

def march_savings : ℕ := 27
def april_savings : ℕ := 13
def may_savings : ℕ := 28
def keyboard_cost : ℕ := 49
def mouse_cost : ℕ := 5

def total_savings : ℕ := march_savings + april_savings + may_savings
def total_spent : ℕ := keyboard_cost + mouse_cost
def money_left : ℕ := total_savings - total_spent

theorem kate_money_left : money_left = 14 := by
  sorry

end NUMINAMATH_CALUDE_kate_money_left_l3530_353075


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l3530_353054

theorem factorial_prime_factorization :
  ∃ (i k m p : ℕ+),
    (8 : ℕ).factorial = 2^(i.val) * 3^(k.val) * 5^(m.val) * 7^(p.val) ∧
    i.val + k.val + m.val + p.val = 11 := by
  sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l3530_353054


namespace NUMINAMATH_CALUDE_louis_fabric_purchase_l3530_353060

theorem louis_fabric_purchase (fabric_cost_per_yard : ℝ) (pattern_cost : ℝ) (thread_cost_per_spool : ℝ) (total_spent : ℝ) : 
  fabric_cost_per_yard = 24 →
  pattern_cost = 15 →
  thread_cost_per_spool = 3 →
  total_spent = 141 →
  (total_spent - pattern_cost - 2 * thread_cost_per_spool) / fabric_cost_per_yard = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_louis_fabric_purchase_l3530_353060


namespace NUMINAMATH_CALUDE_trapezoid_height_is_four_l3530_353082

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The length of the midline
  midline : ℝ
  -- The lengths of the bases
  base1 : ℝ
  base2 : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- Condition: The trapezoid has an inscribed circle
  has_inscribed_circle : Prop
  -- Condition: The midline is the average of the bases
  midline_avg : midline = (base1 + base2) / 2
  -- Condition: The area ratio of the parts divided by the midline
  area_ratio : (base1 - midline) / (base2 - midline) = 7 / 13

/-- The main theorem about the height of the trapezoid -/
theorem trapezoid_height_is_four (t : IsoscelesTrapezoid) 
  (h_midline : t.midline = 5) : t.height = 4 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_height_is_four_l3530_353082


namespace NUMINAMATH_CALUDE_function_inequality_implies_positive_c_l3530_353078

/-- Given a function f(x) = x^2 + x + c, if f(f(x)) > x for all real x, then c > 0 -/
theorem function_inequality_implies_positive_c (c : ℝ) : 
  (∀ x : ℝ, (x^2 + x + c)^2 + (x^2 + x + c) + c > x) → c > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_positive_c_l3530_353078


namespace NUMINAMATH_CALUDE_solution_is_axes_l3530_353068

def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

def x_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

def y_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0}

theorem solution_is_axes : solution_set = x_axis ∪ y_axis := by
  sorry

end NUMINAMATH_CALUDE_solution_is_axes_l3530_353068


namespace NUMINAMATH_CALUDE_polly_cooking_time_l3530_353013

/-- Represents the cooking times for Polly in a week -/
structure CookingTimes where
  breakfast_time : ℕ  -- Time spent cooking breakfast daily
  lunch_time : ℕ      -- Time spent cooking lunch daily
  dinner_time_short : ℕ  -- Time spent cooking dinner on short days
  dinner_time_long : ℕ   -- Time spent cooking dinner on long days
  short_dinner_days : ℕ  -- Number of days with short dinner cooking time
  long_dinner_days : ℕ   -- Number of days with long dinner cooking time

/-- Calculates the total cooking time for a week given the cooking times -/
def total_cooking_time (times : CookingTimes) : ℕ :=
  7 * (times.breakfast_time + times.lunch_time) +
  times.short_dinner_days * times.dinner_time_short +
  times.long_dinner_days * times.dinner_time_long

/-- Theorem stating that Polly's total cooking time for the week is 305 minutes -/
theorem polly_cooking_time :
  ∀ (times : CookingTimes),
  times.breakfast_time = 20 ∧
  times.lunch_time = 5 ∧
  times.dinner_time_short = 10 ∧
  times.dinner_time_long = 30 ∧
  times.short_dinner_days = 4 ∧
  times.long_dinner_days = 3 →
  total_cooking_time times = 305 := by
  sorry


end NUMINAMATH_CALUDE_polly_cooking_time_l3530_353013


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3530_353022

theorem polynomial_division_theorem :
  let dividend : Polynomial ℚ := X^4 * 6 + X^3 * 9 - X^2 * 5 + X * 2 - 8
  let divisor : Polynomial ℚ := X * 3 + 4
  let quotient : Polynomial ℚ := X^3 * 2 - X^2 * 1 + X * 1 - 2
  let remainder : Polynomial ℚ := -8/3
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3530_353022
