import Mathlib

namespace NUMINAMATH_CALUDE_mona_unique_players_l2984_298416

/-- The number of groups Mona joined -/
def num_groups : ℕ := 9

/-- The number of other players in each group -/
def players_per_group : ℕ := 4

/-- The number of repeat players in the first group with repeats -/
def repeat_players_group1 : ℕ := 2

/-- The number of repeat players in the second group with repeats -/
def repeat_players_group2 : ℕ := 1

/-- The total number of unique players Mona grouped with -/
def unique_players : ℕ := num_groups * players_per_group - (repeat_players_group1 + repeat_players_group2)

theorem mona_unique_players : unique_players = 33 := by
  sorry

end NUMINAMATH_CALUDE_mona_unique_players_l2984_298416


namespace NUMINAMATH_CALUDE_minimize_sample_variance_l2984_298411

/-- Given a sample of size 5 with specific conditions, prove that the sample variance is minimized when a₄ = a₅ = 2.5 -/
theorem minimize_sample_variance (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  a₁ = 2.5 → a₂ = 3.5 → a₃ = 4 → a₄ + a₅ = 5 →
  let sample_variance := (1 / 5 : ℝ) * ((a₁ - 3)^2 + (a₂ - 3)^2 + (a₃ - 3)^2 + (a₄ - 3)^2 + (a₅ - 3)^2)
  ∀ b₄ b₅ : ℝ, b₄ + b₅ = 5 → 
  let alt_variance := (1 / 5 : ℝ) * ((a₁ - 3)^2 + (a₂ - 3)^2 + (a₃ - 3)^2 + (b₄ - 3)^2 + (b₅ - 3)^2)
  sample_variance ≤ alt_variance → a₄ = 2.5 ∧ a₅ = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_minimize_sample_variance_l2984_298411


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l2984_298433

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l2984_298433


namespace NUMINAMATH_CALUDE_egg_sales_income_l2984_298407

theorem egg_sales_income (num_hens : ℕ) (eggs_per_hen_per_week : ℕ) (price_per_dozen : ℕ) (num_weeks : ℕ) :
  num_hens = 10 →
  eggs_per_hen_per_week = 12 →
  price_per_dozen = 3 →
  num_weeks = 4 →
  (num_hens * eggs_per_hen_per_week * num_weeks / 12) * price_per_dozen = 120 := by
  sorry

#check egg_sales_income

end NUMINAMATH_CALUDE_egg_sales_income_l2984_298407


namespace NUMINAMATH_CALUDE_workshop_average_salary_l2984_298432

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℚ)
  (non_technician_salary : ℚ)
  (h1 : total_workers = 22)
  (h2 : technicians = 7)
  (h3 : technician_salary = 1000)
  (h4 : non_technician_salary = 780) :
  let non_technicians := total_workers - technicians
  let total_salary := technicians * technician_salary + non_technicians * non_technician_salary
  total_salary / total_workers = 850 := by
sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l2984_298432


namespace NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l2984_298460

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.log 2) / 2
  let b : ℝ := (Real.log 3) / 3
  let c : ℝ := (Real.log 5) / 5
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l2984_298460


namespace NUMINAMATH_CALUDE_compare_f_values_l2984_298402

/-- Given 0 < a < 1, this function satisfies f(log_a x) = (a(x^2 - 1)) / (x(a^2 - 1)) for any x > 0 -/
noncomputable def f (a : ℝ) (t : ℝ) : ℝ := sorry

/-- Theorem: For 0 < a < 1, given function f and m > n > 0, we have f(1/n) > f(1/m) -/
theorem compare_f_values (a m n : ℝ) (ha : 0 < a) (ha' : a < 1) (hmn : m > n) (hn : n > 0) :
  f a (1/n) > f a (1/m) := by sorry

end NUMINAMATH_CALUDE_compare_f_values_l2984_298402


namespace NUMINAMATH_CALUDE_yellow_then_not_yellow_probability_l2984_298470

/-- A deck of cards with 5 suits and 13 ranks. -/
structure Deck :=
  (cards : Finset (Fin 5 × Fin 13))
  (card_count : cards.card = 65)
  (suit_rank_unique : ∀ (s : Fin 5) (r : Fin 13), (s, r) ∈ cards)

/-- The probability of drawing a yellow card followed by a non-yellow card from a shuffled deck. -/
def yellow_then_not_yellow_prob (d : Deck) : ℚ :=
  169 / 1040

/-- Theorem stating the probability of drawing a yellow card followed by a non-yellow card. -/
theorem yellow_then_not_yellow_probability (d : Deck) :
  yellow_then_not_yellow_prob d = 169 / 1040 := by
  sorry

end NUMINAMATH_CALUDE_yellow_then_not_yellow_probability_l2984_298470


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l2984_298450

theorem subtraction_of_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l2984_298450


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2984_298457

/-- The eccentricity of an ellipse with given conditions is between 0 and √2/2 -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let B := (0, b)
  let e := Real.sqrt (a^2 - b^2) / a
  (∀ p ∈ C, Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2) ≤ 2*b) →
  0 < e ∧ e ≤ Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2984_298457


namespace NUMINAMATH_CALUDE_a_equals_negative_one_l2984_298499

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.I * (a - 1) + Complex.I^4 * (a^2 - 1)

/-- If z(a) is a pure imaginary number, then a equals -1 -/
theorem a_equals_negative_one : 
  ∀ a : ℝ, is_pure_imaginary (z a) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_a_equals_negative_one_l2984_298499


namespace NUMINAMATH_CALUDE_characterize_solutions_l2984_298485

/-- Given a system of equations with real parameters a, b, c and variables x, y, z,
    this theorem characterizes all possible solutions. -/
theorem characterize_solutions (a b c x y z : ℝ) :
  x^2 * y^2 + x^2 * z^2 = a * x * y * z ∧
  y^2 * z^2 + y^2 * x^2 = b * x * y * z ∧
  z^2 * x^2 + z^2 * y^2 = c * x * y * z →
  (∃ t : ℝ, (x = t ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = t ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = t)) ∨
  (∃ s : ℝ, s = (a + b + c) / 2 ∧
    ((x^2 = (s - b) * (s - c) ∧ y^2 = (s - a) * (s - c) ∧ z^2 = (s - a) * (s - b)) ∧
     (0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∨
     (0 < -a ∧ 0 < -b ∧ 0 < -c ∧ -a + -b > -c ∧ -a + -c > -b ∧ -b + -c > -a))) :=
by sorry

end NUMINAMATH_CALUDE_characterize_solutions_l2984_298485


namespace NUMINAMATH_CALUDE_inequality_theorem_l2984_298426

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hy₁ : x₁ * y₁ - z₁^2 > 0) (hy₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2984_298426


namespace NUMINAMATH_CALUDE_estimate_product_l2984_298444

def approximate_819 : ℕ := 800
def approximate_32 : ℕ := 30

theorem estimate_product : 
  approximate_819 * approximate_32 = 24000 := by sorry

end NUMINAMATH_CALUDE_estimate_product_l2984_298444


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2984_298468

theorem cyclic_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

#check cyclic_inequality

end NUMINAMATH_CALUDE_cyclic_inequality_l2984_298468


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2984_298427

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
def distPointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Predicate to check if a line intersects a circle -/
def intersects (c : Circle) (l : Line) : Prop :=
  distPointToLine c.center l < c.radius

theorem line_circle_intersection (c : Circle) (l : Line) :
  distPointToLine c.center l < c.radius → intersects c l := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2984_298427


namespace NUMINAMATH_CALUDE_apple_balance_theorem_l2984_298419

variable {α : Type*} [LinearOrderedField α]

def balanced (s t : Finset (α)) : Prop :=
  s.sum id = t.sum id

theorem apple_balance_theorem
  (apples : Finset α)
  (h_count : apples.card = 6)
  (h_tanya : ∃ (s t : Finset α), s ⊆ apples ∧ t ⊆ apples ∧ s ∩ t = ∅ ∧ s ∪ t = apples ∧ s.card = 3 ∧ t.card = 3 ∧ balanced s t)
  (h_sasha : ∃ (u v : Finset α), u ⊆ apples ∧ v ⊆ apples ∧ u ∩ v = ∅ ∧ u ∪ v = apples ∧ u.card = 2 ∧ v.card = 4 ∧ balanced u v) :
  ∃ (x y : Finset α), x ⊆ apples ∧ y ⊆ apples ∧ x ∩ y = ∅ ∧ x ∪ y = apples ∧ x.card = 1 ∧ y.card = 2 ∧ balanced x y :=
by
  sorry

end NUMINAMATH_CALUDE_apple_balance_theorem_l2984_298419


namespace NUMINAMATH_CALUDE_f_satisfies_condition_l2984_298438

-- Define the function f
def f (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem f_satisfies_condition : ∀ x : ℝ, f x + f (2 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_condition_l2984_298438


namespace NUMINAMATH_CALUDE_balanced_digraph_has_valid_coloring_l2984_298401

/-- A directed graph where each vertex has in-degree 2 and out-degree 2 -/
structure BalancedDigraph (V : Type) :=
  (edge : V → V → Prop)
  (in_degree_two : ∀ v, (∃ u w, u ≠ w ∧ edge u v ∧ edge w v) ∧ 
                        (∀ x y z, edge x v → edge y v → edge z v → (x = y ∨ x = z ∨ y = z)))
  (out_degree_two : ∀ v, (∃ u w, u ≠ w ∧ edge v u ∧ edge v w) ∧ 
                         (∀ x y z, edge v x → edge v y → edge v z → (x = y ∨ x = z ∨ y = z)))

/-- A valid coloring of edges in a balanced digraph -/
def ValidColoring (V : Type) (G : BalancedDigraph V) (color : V → V → Bool) : Prop :=
  ∀ v, (∃! u, G.edge v u ∧ color v u = true) ∧
       (∃! u, G.edge v u ∧ color v u = false) ∧
       (∃! u, G.edge u v ∧ color u v = true) ∧
       (∃! u, G.edge u v ∧ color u v = false)

/-- The main theorem: every balanced digraph has a valid coloring -/
theorem balanced_digraph_has_valid_coloring (V : Type) (G : BalancedDigraph V) :
  ∃ color : V → V → Bool, ValidColoring V G color := by
  sorry

end NUMINAMATH_CALUDE_balanced_digraph_has_valid_coloring_l2984_298401


namespace NUMINAMATH_CALUDE_gcd_of_squares_l2984_298422

theorem gcd_of_squares : Nat.gcd (121^2 + 233^2 + 345^2) (120^2 + 232^2 + 346^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l2984_298422


namespace NUMINAMATH_CALUDE_journey_distance_is_20km_l2984_298424

/-- Represents a round trip journey with a horizontal section and a hill. -/
structure Journey where
  total_time : ℝ
  horizontal_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Calculates the total distance covered in a journey. -/
def total_distance (j : Journey) : ℝ :=
  j.total_time * j.horizontal_speed

/-- Theorem stating that for the given journey parameters, the total distance is 20 km. -/
theorem journey_distance_is_20km (j : Journey)
  (h_time : j.total_time = 5)
  (h_horizontal : j.horizontal_speed = 4)
  (h_uphill : j.uphill_speed = 3)
  (h_downhill : j.downhill_speed = 6) :
  total_distance j = 20 := by
  sorry

#eval total_distance { total_time := 5, horizontal_speed := 4, uphill_speed := 3, downhill_speed := 6 }

end NUMINAMATH_CALUDE_journey_distance_is_20km_l2984_298424


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l2984_298486

/-- The set of possible slopes for a line with y-intercept (0, -3) intersecting the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/20) ∨ m ≥ Real.sqrt (1/20)}

/-- The equation of the line with slope m and y-intercept (0, -3) -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) : 
  m ∈ possible_slopes ↔ 
  ∃ x : ℝ, ellipse_equation x (line_equation m x) := by
  sorry

#check line_intersects_ellipse

end NUMINAMATH_CALUDE_line_intersects_ellipse_l2984_298486


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l2984_298474

def vector1 : Fin 2 → ℝ := ![2, 5]
def vector2 (x : ℝ) : Fin 2 → ℝ := ![x, -3]

theorem vectors_orthogonal :
  let x : ℝ := 15/2
  (vector1 0 * vector2 x 0 + vector1 1 * vector2 x 1 = 0) := by sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l2984_298474


namespace NUMINAMATH_CALUDE_solid_surface_area_theorem_l2984_298408

def solid_surface_area (s : ℝ) (h : ℝ) : ℝ :=
  let base_area := s^2
  let upper_area := 3 * s^2
  let trapezoid_area := 2 * (s + 3*s) * h
  base_area + upper_area + trapezoid_area

theorem solid_surface_area_theorem :
  solid_surface_area (4 * Real.sqrt 2) (3 * Real.sqrt 2) = 320 := by
  sorry

end NUMINAMATH_CALUDE_solid_surface_area_theorem_l2984_298408


namespace NUMINAMATH_CALUDE_quadratic_roots_min_value_l2984_298498

theorem quadratic_roots_min_value (m : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - 2*m*x₁ + m + 6 = 0 →
  x₂^2 - 2*m*x₂ + m + 6 = 0 →
  x₁ ≠ x₂ →
  ∀ y : ℝ, y = (x₁ - 1)^2 + (x₂ - 1)^2 → y ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_min_value_l2984_298498


namespace NUMINAMATH_CALUDE_keith_digimon_pack_price_l2984_298431

/-- The price of each pack of Digimon cards -/
def digimon_pack_price (total_spent : ℚ) (baseball_deck_price : ℚ) (num_digimon_packs : ℕ) : ℚ :=
  (total_spent - baseball_deck_price) / num_digimon_packs

theorem keith_digimon_pack_price :
  digimon_pack_price 23.86 6.06 4 = 4.45 := by
  sorry

end NUMINAMATH_CALUDE_keith_digimon_pack_price_l2984_298431


namespace NUMINAMATH_CALUDE_rose_ratio_l2984_298435

theorem rose_ratio (total : ℕ) (red : ℕ) (yellow : ℕ) (white : ℕ) : 
  total = 80 →
  yellow = (total - red) / 4 →
  red + white = 75 →
  total = red + yellow + white →
  (red : ℚ) / total = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rose_ratio_l2984_298435


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_length_l2984_298481

theorem line_ellipse_intersection_length : ∃ A B : ℝ × ℝ,
  (∀ x y : ℝ, y = x - 1 → x^2 / 4 + y^2 / 3 = 1 → (x, y) = A ∨ (x, y) = B) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24 / 7 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_length_l2984_298481


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l2984_298494

theorem sin_cos_sum_equals_one :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l2984_298494


namespace NUMINAMATH_CALUDE_distance_inequality_l2984_298429

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define the vertices of the quadrilateral
variable (A B C D : V)

-- Define the condition that all sides are equal
variable (h : ‖A - B‖ = ‖B - C‖ ∧ ‖B - C‖ = ‖C - D‖ ∧ ‖C - D‖ = ‖D - A‖)

-- State the theorem
theorem distance_inequality (P : V) : 
  ‖P - A‖ < ‖P - B‖ + ‖P - C‖ + ‖P - D‖ := by sorry

end NUMINAMATH_CALUDE_distance_inequality_l2984_298429


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l2984_298455

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beads_percent : ℝ
  marbles_percent : ℝ
  silver_coins_percent : ℝ
  gold_coins_percent : ℝ

/-- Theorem stating the percentage of gold coins in the urn --/
theorem gold_coins_percentage (u : UrnComposition) 
  (beads_cond : u.beads_percent = 0.3)
  (marbles_cond : u.marbles_percent = 0.1)
  (silver_coins_cond : u.silver_coins_percent = 0.45 * (1 - u.beads_percent - u.marbles_percent))
  (total_cond : u.beads_percent + u.marbles_percent + u.silver_coins_percent + u.gold_coins_percent = 1) :
  u.gold_coins_percent = 0.33 := by
  sorry

#check gold_coins_percentage

end NUMINAMATH_CALUDE_gold_coins_percentage_l2984_298455


namespace NUMINAMATH_CALUDE_arrangement_count_l2984_298439

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of people in the Jia-Bing-Yi group
def group_size : ℕ := 3

-- Define the number of units (group + other individuals)
def num_units : ℕ := total_people - group_size + 1

-- Theorem statement
theorem arrangement_count : 
  (num_units.factorial * group_size.factorial * 2) - (num_units.factorial * 2) = 240 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l2984_298439


namespace NUMINAMATH_CALUDE_honda_production_l2984_298412

/-- Honda car production problem -/
theorem honda_production (day_shift second_shift total : ℕ) : 
  day_shift = 4 * second_shift → 
  second_shift = 1100 → 
  total = day_shift + second_shift → 
  total = 5500 := by
  sorry

end NUMINAMATH_CALUDE_honda_production_l2984_298412


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l2984_298443

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l2984_298443


namespace NUMINAMATH_CALUDE_constant_value_l2984_298488

theorem constant_value (x : ℝ) (some_constant a k n : ℝ) :
  (3 * x + some_constant) * (2 * x - 7) = a * x^2 + k * x + n →
  a - n + k = 3 →
  some_constant = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l2984_298488


namespace NUMINAMATH_CALUDE_container_max_volume_l2984_298445

/-- The volume function for the container --/
def volume (x : ℝ) : ℝ := (90 - 2*x) * (48 - 2*x) * x

/-- The derivative of the volume function --/
def volume_derivative (x : ℝ) : ℝ := 12 * (x^2 - 46*x + 360)

theorem container_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 24 ∧
  ∀ (y : ℝ), y > 0 → y < 24 → volume y ≤ volume x ∧
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_container_max_volume_l2984_298445


namespace NUMINAMATH_CALUDE_systematic_sampling_methods_l2984_298413

/-- Represents a sampling method -/
inductive SamplingMethod
  | BallSelection
  | ProductInspection
  | MarketSurvey
  | CinemaAudienceSurvey

/-- Defines the characteristics of systematic sampling -/
def is_systematic_sampling (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.BallSelection => true
  | SamplingMethod.ProductInspection => true
  | SamplingMethod.MarketSurvey => false
  | SamplingMethod.CinemaAudienceSurvey => true

/-- Theorem stating which sampling methods are systematic -/
theorem systematic_sampling_methods :
  (is_systematic_sampling SamplingMethod.BallSelection) ∧
  (is_systematic_sampling SamplingMethod.ProductInspection) ∧
  (¬is_systematic_sampling SamplingMethod.MarketSurvey) ∧
  (is_systematic_sampling SamplingMethod.CinemaAudienceSurvey) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_methods_l2984_298413


namespace NUMINAMATH_CALUDE_direction_vector_of_line_l_l2984_298421

/-- The line l is defined by the equation (x-1)/3 = (y+1)/4 -/
def line_l (x y : ℝ) : Prop := (x - 1) / 3 = (y + 1) / 4

/-- A direction vector of a line is a vector parallel to the line -/
def is_direction_vector (v : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (t : ℝ) (x y : ℝ), l x y → l (x + t * v.1) (y + t * v.2)

/-- Prove that (3,4) is a direction vector of the line l -/
theorem direction_vector_of_line_l : is_direction_vector (3, 4) line_l := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_of_line_l_l2984_298421


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_l2984_298478

/-- In a triangle ABC where sides a, b, c form a geometric sequence and satisfy a² - c² = ac - bc, 
    the ratio (b * sin B) / c is equal to √3/2. -/
theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b ^ 2 = a * c) →  -- geometric sequence condition
  (a ^ 2 - c ^ 2 = a * c - b * c) →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →  -- cosine rule
  (b * Real.sin B = a * Real.sin A) →  -- sine rule
  (b * Real.sin B) / c = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_geometric_sequence_l2984_298478


namespace NUMINAMATH_CALUDE_quadratic_range_l2984_298452

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem quadratic_range :
  ∀ x : ℝ, (2 ≤ quadratic_function x ∧ quadratic_function x < 5) ↔
  (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) := by sorry

end NUMINAMATH_CALUDE_quadratic_range_l2984_298452


namespace NUMINAMATH_CALUDE_percent_of_x_is_z_l2984_298453

theorem percent_of_x_is_z (x y z : ℝ) 
  (h1 : 0.45 * z = 1.20 * y) 
  (h2 : y = 0.75 * x) : 
  z = 2 * x := by
sorry

end NUMINAMATH_CALUDE_percent_of_x_is_z_l2984_298453


namespace NUMINAMATH_CALUDE_cone_slant_height_l2984_298459

/-- The slant height of a cone with base radius 6 cm and lateral surface sector angle 240° is 9 cm. -/
theorem cone_slant_height (base_radius : ℝ) (sector_angle : ℝ) (slant_height : ℝ) : 
  base_radius = 6 →
  sector_angle = 240 →
  slant_height = (360 / sector_angle) * base_radius →
  slant_height = 9 := by
sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2984_298459


namespace NUMINAMATH_CALUDE_budget_allocation_home_electronics_l2984_298482

theorem budget_allocation_home_electronics : 
  ∀ (total_budget : ℝ) (microphotonics food_additives gmo industrial_lubricants astrophysics home_electronics : ℝ),
  total_budget > 0 →
  microphotonics = 0.14 * total_budget →
  food_additives = 0.15 * total_budget →
  gmo = 0.19 * total_budget →
  industrial_lubricants = 0.08 * total_budget →
  astrophysics = (72 / 360) * total_budget →
  home_electronics + microphotonics + food_additives + gmo + industrial_lubricants + astrophysics = total_budget →
  home_electronics = 0.24 * total_budget :=
by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_home_electronics_l2984_298482


namespace NUMINAMATH_CALUDE_hundredth_digit_of_13_over_90_l2984_298423

theorem hundredth_digit_of_13_over_90 : 
  ∃ (d : ℕ), d = 4 ∧ 
  (∃ (a b : ℕ), (13 : ℚ) / 90 = a + (d : ℚ) / 10^100 + b / 10^101 ∧ 
                0 ≤ b ∧ b < 10) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_digit_of_13_over_90_l2984_298423


namespace NUMINAMATH_CALUDE_electric_sharpener_time_l2984_298440

/-- Proves that an electric pencil sharpener takes 20 seconds to sharpen one pencil -/
theorem electric_sharpener_time : ∀ (hand_crank_time electric_time : ℕ),
  hand_crank_time = 45 →
  (360 / hand_crank_time : ℚ) + 10 = 360 / electric_time →
  electric_time = 20 :=
by sorry

end NUMINAMATH_CALUDE_electric_sharpener_time_l2984_298440


namespace NUMINAMATH_CALUDE_set_equality_implies_a_plus_minus_one_l2984_298490

theorem set_equality_implies_a_plus_minus_one (a : ℝ) :
  ({0, -1, 2*a} : Set ℝ) = {a-1, -abs a, a+1} →
  (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_plus_minus_one_l2984_298490


namespace NUMINAMATH_CALUDE_negative_one_third_squared_l2984_298495

theorem negative_one_third_squared : (-1/3 : ℚ)^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_third_squared_l2984_298495


namespace NUMINAMATH_CALUDE_parcel_weight_sum_l2984_298466

/-- Given three parcels with weights x, y, and z, prove that their total weight is 209 pounds. -/
theorem parcel_weight_sum (x y z : ℝ) 
  (h1 : x + y = 132)
  (h2 : y + z = 146)
  (h3 : z + x = 140) : 
  x + y + z = 209 := by
  sorry

end NUMINAMATH_CALUDE_parcel_weight_sum_l2984_298466


namespace NUMINAMATH_CALUDE_road_length_l2984_298418

/-- Given 10 trees planted on one side of a road at intervals of 10 meters,
    with trees at both ends, prove that the length of the road is 90 meters. -/
theorem road_length (num_trees : ℕ) (interval : ℕ) : 
  num_trees = 10 → interval = 10 → (num_trees - 1) * interval = 90 := by
  sorry

end NUMINAMATH_CALUDE_road_length_l2984_298418


namespace NUMINAMATH_CALUDE_x_difference_l2984_298471

theorem x_difference (x₁ x₂ : ℝ) : 
  ((x₁ + 3)^2 / (2*x₁ + 15) = 3) →
  ((x₂ + 3)^2 / (2*x₂ + 15) = 3) →
  x₁ ≠ x₂ →
  |x₁ - x₂| = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_x_difference_l2984_298471


namespace NUMINAMATH_CALUDE_bruce_grapes_purchase_l2984_298458

theorem bruce_grapes_purchase (grape_price mango_price mango_quantity total_paid : ℕ) 
  (h1 : grape_price = 70)
  (h2 : mango_price = 55)
  (h3 : mango_quantity = 9)
  (h4 : total_paid = 1055) :
  ∃ grape_quantity : ℕ, grape_quantity * grape_price + mango_quantity * mango_price = total_paid ∧ grape_quantity = 8 := by
  sorry

end NUMINAMATH_CALUDE_bruce_grapes_purchase_l2984_298458


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l2984_298464

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (h1 : length = 90)
  (h2 : num_poles = 70)
  (h3 : pole_distance = 4)
  : ∃ width : ℝ, width = 48 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l2984_298464


namespace NUMINAMATH_CALUDE_problem_statement_l2984_298448

-- Define sets A, B, and C
def A : Set ℝ := {x | |3*x - 4| > 2}
def B : Set ℝ := {x | x^2 - x - 2 > 0}
def C (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 1) ≥ 0}

-- Define predicates p, q, and r
def p : Set ℝ := {x | 2/3 ≤ x ∧ x ≤ 2}
def q : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def r (a : ℝ) : Set ℝ := {x | x ≤ a ∨ x ≥ a + 1}

theorem problem_statement :
  (∀ x : ℝ, x ∈ p → x ∈ q) ∧
  (∃ x : ℝ, x ∈ q ∧ x ∉ p) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ p → x ∈ r a) ∧ (∃ x : ℝ, x ∈ r a ∧ x ∉ p) ↔ a ≥ 2 ∨ a ≤ -1/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2984_298448


namespace NUMINAMATH_CALUDE_percentage_calculation_l2984_298462

theorem percentage_calculation (a : ℝ) (x : ℝ) (h1 : a = 140) (h2 : (x / 100) * a = 70) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2984_298462


namespace NUMINAMATH_CALUDE_sum_of_coefficients_10_11_l2984_298476

/-- Given that (x-1)^21 = a + a₁x + a₂x² + ... + a₂₁x²¹, prove that a₁₀ + a₁₁ = 0 -/
theorem sum_of_coefficients_10_11 (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ a₁₅ a₁₆ a₁₇ a₁₈ a₁₉ a₂₀ a₂₁ : ℝ) :
  (∀ x : ℝ, (x - 1)^21 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + 
             a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14 + a₁₅*x^15 + a₁₆*x^16 + a₁₇*x^17 + a₁₈*x^18 + 
             a₁₉*x^19 + a₂₀*x^20 + a₂₁*x^21) →
  a₁₀ + a₁₁ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_10_11_l2984_298476


namespace NUMINAMATH_CALUDE_part_I_part_II_part_III_l2984_298436

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^2 + (1 - 3*m) * x + 2*m - 1

-- Part I
theorem part_I (a : ℝ) :
  (a > 0 ∧ {x : ℝ | f 2 x ≤ 0} ⊆ Set.Ioo a (2*a + 1)) ↔ (1/4 ≤ a ∧ a < 1) :=
sorry

-- Part II
def solution_set (m : ℝ) : Set ℝ :=
  if m < 0 then Set.Iic 1 ∪ Set.Ici (2 - 1/m)
  else if m = 0 then Set.Iic 1
  else if 0 < m ∧ m < 1 then Set.Icc (2 - 1/m) 1
  else if m = 1 then {1}
  else Set.Icc 1 (2 - 1/m)

theorem part_II (m : ℝ) :
  {x : ℝ | f m x ≤ 0} = solution_set m :=
sorry

-- Part III
theorem part_III (m : ℝ) :
  (∃ x > 0, f m x > -3*m*x + m - 1) ↔ m > -1/2 :=
sorry

end NUMINAMATH_CALUDE_part_I_part_II_part_III_l2984_298436


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2984_298447

/-- A regular polygon with perimeter 150 cm and side length 10 cm has 15 sides -/
theorem regular_polygon_sides (perimeter : ℝ) (side_length : ℝ) (num_sides : ℕ) :
  perimeter = 150 ∧ side_length = 10 ∧ perimeter = num_sides * side_length → num_sides = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2984_298447


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l2984_298473

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| + 3 = 47 ∧ |5 * x₂| + 3 = 47) ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -1936/25) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l2984_298473


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2984_298415

theorem trigonometric_simplification (x z : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + z) ^ 2 - 2 * Real.sin x * Real.sin z * Real.sin (x + z) = Real.sin z ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2984_298415


namespace NUMINAMATH_CALUDE_square_two_minus_sqrt_three_l2984_298492

theorem square_two_minus_sqrt_three (a b : ℚ) :
  (2 - Real.sqrt 3)^2 = a + b * Real.sqrt 3 → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_two_minus_sqrt_three_l2984_298492


namespace NUMINAMATH_CALUDE_picture_processing_time_l2984_298409

theorem picture_processing_time (num_pictures : ℕ) (processing_time_per_picture : ℕ) : 
  num_pictures = 960 → 
  processing_time_per_picture = 2 → 
  (num_pictures * processing_time_per_picture) / 60 = 32 := by
sorry

end NUMINAMATH_CALUDE_picture_processing_time_l2984_298409


namespace NUMINAMATH_CALUDE_unique_solution_l2984_298414

/-- Represents the intersection point of two lines --/
structure IntersectionPoint where
  x : ℤ
  y : ℤ

/-- Checks if a given point satisfies both line equations --/
def is_valid_intersection (m : ℕ) (p : IntersectionPoint) : Prop :=
  13 * p.x + 11 * p.y = 700 ∧ p.y = m * p.x - 1

/-- Main theorem: m = 6 is the only solution --/
theorem unique_solution : 
  ∃! (m : ℕ), ∃ (p : IntersectionPoint), is_valid_intersection m p :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2984_298414


namespace NUMINAMATH_CALUDE_race_result_l2984_298465

/-- Represents a participant in the race -/
structure Participant where
  position : ℝ
  speed : ℝ

/-- The race setup -/
structure Race where
  distance : ℝ
  a : Participant
  b : Participant
  c : Participant

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.distance = 60 ∧
  r.a.position = r.distance ∧
  r.b.position = r.distance - 10 ∧
  r.c.position = r.distance - 20

/-- Theorem stating the result of the race -/
theorem race_result (r : Race) :
  race_conditions r →
  (r.distance / r.b.speed - r.distance / r.c.speed) * r.c.speed = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_race_result_l2984_298465


namespace NUMINAMATH_CALUDE_drop_notation_l2984_298405

/-- Represents a temperature change in Celsius -/
structure TempChange where
  value : ℤ

/-- Notation for temperature changes -/
def temp_notation (change : TempChange) : ℤ :=
  change.value

/-- Given condition: A temperature rise of 3℃ is denoted as +3℃ -/
axiom rise_notation : temp_notation ⟨3⟩ = 3

/-- Theorem: A temperature drop of 8℃ is denoted as -8℃ -/
theorem drop_notation : temp_notation ⟨-8⟩ = -8 := by
  sorry

end NUMINAMATH_CALUDE_drop_notation_l2984_298405


namespace NUMINAMATH_CALUDE_indeterminate_teachers_per_department_l2984_298437

/-- Represents a school with departments and teachers -/
structure School where
  departments : ℕ
  total_teachers : ℕ

/-- Defines a function to check if it's possible to determine exact number of teachers per department -/
def can_determine_teachers_per_department (s : School) : Prop :=
  ∃ (teachers_per_dept : ℕ), s.total_teachers = s.departments * teachers_per_dept

/-- Theorem stating that for a school with 7 departments and 140 teachers, 
    it's not always possible to determine the exact number of teachers in each department -/
theorem indeterminate_teachers_per_department :
  ¬ ∀ (s : School), s.departments = 7 ∧ s.total_teachers = 140 → can_determine_teachers_per_department s :=
by
  sorry


end NUMINAMATH_CALUDE_indeterminate_teachers_per_department_l2984_298437


namespace NUMINAMATH_CALUDE_probability_of_exact_successes_l2984_298467

def probability_of_success : ℚ := 1/3

def number_of_trials : ℕ := 3

def number_of_successes : ℕ := 2

theorem probability_of_exact_successes :
  (number_of_trials.choose number_of_successes) *
  probability_of_success ^ number_of_successes *
  (1 - probability_of_success) ^ (number_of_trials - number_of_successes) =
  2/9 :=
sorry

end NUMINAMATH_CALUDE_probability_of_exact_successes_l2984_298467


namespace NUMINAMATH_CALUDE_sum_base3_equals_11000_l2984_298493

/-- Represents a number in base 3 --/
def Base3 : Type := List Nat

/-- Converts a base 3 number to its decimal representation --/
def to_decimal (n : Base3) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- Addition of two base 3 numbers --/
def add_base3 (a b : Base3) : Base3 :=
  sorry

/-- Theorem: The sum of 2₃, 22₃, 202₃, and 2022₃ is 11000₃ in base 3 --/
theorem sum_base3_equals_11000 :
  let a := [2]
  let b := [2, 2]
  let c := [2, 0, 2]
  let d := [2, 2, 0, 2]
  let result := [1, 1, 0, 0, 0]
  add_base3 (add_base3 (add_base3 a b) c) d = result :=
sorry

end NUMINAMATH_CALUDE_sum_base3_equals_11000_l2984_298493


namespace NUMINAMATH_CALUDE_prime_remainder_theorem_l2984_298472

theorem prime_remainder_theorem (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (p^3 + 17) % 24 = 0 ∨ (p^3 + 17) % 24 = 16 := by
  sorry

end NUMINAMATH_CALUDE_prime_remainder_theorem_l2984_298472


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l2984_298410

theorem newberg_airport_passengers :
  let on_time : ℕ := 14507
  let late : ℕ := 213
  on_time + late = 14720 :=
by sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l2984_298410


namespace NUMINAMATH_CALUDE_triangle_side_angle_relation_l2984_298484

theorem triangle_side_angle_relation (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  2 * c^2 - 2 * a^2 = b^2 →
  2 * c * Real.cos A - 2 * a * Real.cos C = b :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_angle_relation_l2984_298484


namespace NUMINAMATH_CALUDE_multiplication_results_l2984_298434

theorem multiplication_results (h : 25 * 4 = 100) : 
  (25 * 8 = 200) ∧ 
  (25 * 12 = 300) ∧ 
  (250 * 40 = 10000) ∧ 
  (25 * 24 = 600) := by
sorry

end NUMINAMATH_CALUDE_multiplication_results_l2984_298434


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2984_298406

/-- Given a hyperbola with real axis length 16 and imaginary axis length 12, its eccentricity is 5/4 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a = 8) (h2 : b = 6) : 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5/4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2984_298406


namespace NUMINAMATH_CALUDE_ball_probability_l2984_298491

/-- Given a bag of balls with the specified conditions, prove the probability of choosing a ball that is neither red nor purple -/
theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h_total : total = 60)
  (h_red : red = 5)
  (h_purple : purple = 7) :
  (total - (red + purple)) / total = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l2984_298491


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2984_298463

theorem triangle_perimeter (a b c : ℝ) (h1 : a = 4000) (h2 : b = 3500) 
  (h3 : c^2 = a^2 - b^2) : a + b + c = 9437 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2984_298463


namespace NUMINAMATH_CALUDE_angles_on_axes_l2984_298489

def TerminalSideOnAxes (α : Real) : Prop :=
  ∃ k : ℤ, α = k * (Real.pi / 2)

theorem angles_on_axes :
  {α : Real | TerminalSideOnAxes α} = {α : Real | ∃ k : ℤ, α = k * (Real.pi / 2)} := by
  sorry

end NUMINAMATH_CALUDE_angles_on_axes_l2984_298489


namespace NUMINAMATH_CALUDE_min_total_cost_l2984_298425

-- Define the probabilities and costs
def prob_event : ℝ := 0.3
def loss : ℝ := 400 -- in ten thousand yuan
def cost_A : ℝ := 45 -- in ten thousand yuan
def cost_B : ℝ := 30 -- in ten thousand yuan
def prob_no_event_A : ℝ := 0.9
def prob_no_event_B : ℝ := 0.85

-- Define the total cost function for each scenario
def total_cost_none : ℝ := prob_event * loss
def total_cost_A : ℝ := cost_A + (1 - prob_no_event_A) * loss
def total_cost_B : ℝ := cost_B + (1 - prob_no_event_B) * loss
def total_cost_both : ℝ := cost_A + cost_B + (1 - prob_no_event_A * prob_no_event_B) * loss

-- Theorem: Implementing measure A results in the minimum total cost
theorem min_total_cost :
  total_cost_A = 85 ∧ 
  total_cost_A ≤ total_cost_none ∧ 
  total_cost_A ≤ total_cost_B ∧ 
  total_cost_A ≤ total_cost_both :=
sorry

end NUMINAMATH_CALUDE_min_total_cost_l2984_298425


namespace NUMINAMATH_CALUDE_k_satisfies_conditions_l2984_298461

/-- The number of digits in the second factor of (9)(999...9) -/
def k : ℕ := 55

/-- The resulting integer from the multiplication (9)(999...9) -/
def result (n : ℕ) : ℕ := 9 * (10^n - 1)

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- The theorem stating that k satisfies the given conditions -/
theorem k_satisfies_conditions : digit_sum (result k) = 500 := by sorry

end NUMINAMATH_CALUDE_k_satisfies_conditions_l2984_298461


namespace NUMINAMATH_CALUDE_division_remainder_l2984_298420

theorem division_remainder (N : ℕ) : N = 7 * 5 + 0 → N % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2984_298420


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2984_298400

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 = 5 →
  a 8 = 6 →
  a 2 * a 10 = 30 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2984_298400


namespace NUMINAMATH_CALUDE_prob_second_day_restaurant_A_l2984_298480

/-- Represents the restaurants in the Olympic Village -/
inductive Restaurant
| A  -- Smart restaurant
| B  -- Manual restaurant

/-- The probability of choosing restaurant A on the second day -/
def prob_second_day_A (first_day_choice : Restaurant) : ℝ :=
  match first_day_choice with
  | Restaurant.A => 0.6
  | Restaurant.B => 0.5

/-- The probability of choosing a restaurant on the first day -/
def prob_first_day (r : Restaurant) : ℝ := 0.5

/-- The theorem stating the probability of going to restaurant A on the second day -/
theorem prob_second_day_restaurant_A :
  (prob_first_day Restaurant.A * prob_second_day_A Restaurant.A +
   prob_first_day Restaurant.B * prob_second_day_A Restaurant.B) = 0.55 := by
  sorry


end NUMINAMATH_CALUDE_prob_second_day_restaurant_A_l2984_298480


namespace NUMINAMATH_CALUDE_product_of_scientific_notation_l2984_298477

theorem product_of_scientific_notation :
  (-2 * (10 ^ 4)) * (4 * (10 ^ 5)) = -8 * (10 ^ 9) := by
  sorry

end NUMINAMATH_CALUDE_product_of_scientific_notation_l2984_298477


namespace NUMINAMATH_CALUDE_at_least_one_basketball_l2984_298403

/-- Represents the total number of balls -/
def totalBalls : ℕ := 8

/-- Represents the number of basketballs -/
def numBasketballs : ℕ := 6

/-- Represents the number of volleyballs -/
def numVolleyballs : ℕ := 2

/-- Represents the number of balls to be chosen -/
def chosenBalls : ℕ := 3

/-- Theorem stating that at least one basketball is always chosen -/
theorem at_least_one_basketball : 
  ∀ (selection : Finset (Fin totalBalls)), 
  selection.card = chosenBalls → 
  ∃ (i : Fin totalBalls), i ∈ selection ∧ i.val < numBasketballs :=
sorry

end NUMINAMATH_CALUDE_at_least_one_basketball_l2984_298403


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l2984_298449

theorem sum_of_reciprocals_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 4) :
  1/a + 1/b ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l2984_298449


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2984_298475

theorem perfect_square_trinomial (a : ℚ) : 
  (∃ r s : ℚ, a * x^2 + 20 * x + 9 = (r * x + s)^2) → a = 100 / 9 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2984_298475


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_wire_length_is_15840_l2984_298479

/-- The length of wire required to go 15 times around a square field with area 69696 m² -/
theorem wire_length_around_square_field : ℝ :=
  let field_area : ℝ := 69696
  let side_length : ℝ := Real.sqrt field_area
  let perimeter : ℝ := 4 * side_length
  let num_rounds : ℝ := 15
  num_rounds * perimeter

/-- Proof that the wire length is 15840 m -/
theorem wire_length_is_15840 : wire_length_around_square_field = 15840 := by
  sorry


end NUMINAMATH_CALUDE_wire_length_around_square_field_wire_length_is_15840_l2984_298479


namespace NUMINAMATH_CALUDE_equation_equivalence_l2984_298446

theorem equation_equivalence (x Q : ℝ) (h : 5 * (5 * x + 7 * Real.pi) = Q) :
  10 * (10 * x + 14 * Real.pi + 2) = 4 * Q + 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2984_298446


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2984_298454

theorem min_value_of_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4*x + y = x*y) :
  x + y ≥ 13 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2984_298454


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2984_298456

theorem arithmetic_sequence_ratio (x y d₁ d₂ : ℝ) (h₁ : d₁ ≠ 0) (h₂ : d₂ ≠ 0) : 
  (x + 4 * d₁ = y) → (x + 5 * d₂ = y) → d₁ / d₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2984_298456


namespace NUMINAMATH_CALUDE_length_of_DE_l2984_298404

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the angles
variable (angle_BAE angle_CBE angle_DCE : ℝ)

-- Define the side lengths
variable (AE AB BC CD : ℝ)

-- Define t
variable (t : ℝ)

-- State the theorem
theorem length_of_DE (h1 : angle_BAE = 90) (h2 : angle_CBE = 90) (h3 : angle_DCE = 90)
                     (h4 : AE = Real.sqrt 5) (h5 : AB = Real.sqrt 4) (h6 : BC = Real.sqrt 3)
                     (h7 : CD = Real.sqrt t) (h8 : t = 4) :
  Real.sqrt ((CD^2) + (Real.sqrt ((BC^2) + (Real.sqrt (AB^2 + AE^2))^2))^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_length_of_DE_l2984_298404


namespace NUMINAMATH_CALUDE_larger_cube_volume_l2984_298442

/-- Proves that a cube containing 64 smaller cubes of 1 cubic inch each, with a surface area
    difference of 288 square inches between the sum of the smaller cubes' surface areas and
    the larger cube's surface area, has a volume of 64 cubic inches. -/
theorem larger_cube_volume (s : ℝ) (h1 : s > 0) :
  (s^3 : ℝ) = 64 ∧
  64 * (6 : ℝ) - 6 * s^2 = 288 →
  (s^3 : ℝ) = 64 := by sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l2984_298442


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2984_298496

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (∃ c : ℝ, (x - 2/x)^4 = c*x^2 + (terms_without_x_squared : ℝ)) → 
  (∃ c : ℝ, (x - 2/x)^4 = 8*x^2 + (terms_without_x_squared : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2984_298496


namespace NUMINAMATH_CALUDE_smallest_zero_201_l2984_298487

/-- A sequence defined by the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 134
  | 1 => 150
  | (k + 2) => a k - (k + 1) / a (k + 1)

/-- The property that a_n = 0 -/
def sequence_zero (n : ℕ) : Prop := a n = 0

/-- Theorem stating that 201 is the smallest positive integer n for which a_n = 0 -/
theorem smallest_zero_201 : 
  (∀ m : ℕ, m < 201 → ¬ sequence_zero m) ∧ sequence_zero 201 := by sorry

end NUMINAMATH_CALUDE_smallest_zero_201_l2984_298487


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l2984_298469

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l2984_298469


namespace NUMINAMATH_CALUDE_six_circles_l2984_298441

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (a : Point) (b : Point) (c : Point)

/-- Two identical equilateral triangles sharing one vertex -/
structure TwoTriangles :=
  (t1 : EquilateralTriangle)
  (t2 : EquilateralTriangle)
  (shared_vertex : Point)
  (h1 : t1.c = shared_vertex)
  (h2 : t2.a = shared_vertex)

/-- A function that returns all circles satisfying the conditions -/
def circles_through_vertices (triangles : TwoTriangles) : Finset Circle := sorry

/-- The main theorem -/
theorem six_circles (triangles : TwoTriangles) :
  (circles_through_vertices triangles).card = 6 := by sorry

end NUMINAMATH_CALUDE_six_circles_l2984_298441


namespace NUMINAMATH_CALUDE_sufficient_not_imply_necessary_l2984_298428

-- Define the propositions A and B
variable (A B : Prop)

-- Define what it means for B to be a sufficient condition for A
def sufficient (B A : Prop) : Prop := B → A

-- Define what it means for A to be a necessary condition for B
def necessary (A B : Prop) : Prop := B → A

-- Theorem: If B is sufficient for A, it doesn't necessarily mean A is necessary for B
theorem sufficient_not_imply_necessary (h : sufficient B A) : 
  ¬ (∀ A B, sufficient B A → necessary A B) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_imply_necessary_l2984_298428


namespace NUMINAMATH_CALUDE_mixture_alcohol_percentage_l2984_298430

/-- The percentage of alcohol in solution X -/
def alcohol_percent_X : ℝ := 15

/-- The percentage of alcohol in solution Y -/
def alcohol_percent_Y : ℝ := 45

/-- The initial volume of solution X in milliliters -/
def initial_volume_X : ℝ := 300

/-- The volume of solution Y to be added in milliliters -/
def volume_Y : ℝ := 150

/-- The desired percentage of alcohol in the final solution -/
def target_alcohol_percent : ℝ := 25

/-- Theorem stating that adding 150 mL of solution Y to 300 mL of solution X
    results in a solution with 25% alcohol by volume -/
theorem mixture_alcohol_percentage :
  let total_volume := initial_volume_X + volume_Y
  let total_alcohol := (alcohol_percent_X / 100) * initial_volume_X + (alcohol_percent_Y / 100) * volume_Y
  (total_alcohol / total_volume) * 100 = target_alcohol_percent := by
  sorry

end NUMINAMATH_CALUDE_mixture_alcohol_percentage_l2984_298430


namespace NUMINAMATH_CALUDE_not_decreasing_on_interval_l2984_298497

-- Define a real-valued function on the real line
variable (f : ℝ → ℝ)

-- State the theorem
theorem not_decreasing_on_interval (h : f (-1) < f 1) :
  ¬(∀ x y : ℝ, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → x ≤ y → f x ≥ f y) :=
by sorry

end NUMINAMATH_CALUDE_not_decreasing_on_interval_l2984_298497


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2984_298417

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 105 →
  a * d + b * c = 225 →
  c * d = 144 →
  a^2 + b^2 + c^2 + d^2 ≤ 150 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2984_298417


namespace NUMINAMATH_CALUDE_phone_number_prime_factorization_l2984_298483

theorem phone_number_prime_factorization :
  ∃ (p q r s : ℕ), 
    (Nat.Prime p) ∧ 
    (Nat.Prime q) ∧ 
    (Nat.Prime r) ∧ 
    (Nat.Prime s) ∧
    (q = p + 2) ∧ 
    (r = q + 2) ∧ 
    (s = r + 2) ∧
    (p * q * r * s = 27433619) ∧
    (p + q + r + s = 290) := by
  sorry

end NUMINAMATH_CALUDE_phone_number_prime_factorization_l2984_298483


namespace NUMINAMATH_CALUDE_complex_magnitude_l2984_298451

theorem complex_magnitude (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (h1 : (1 + a * i) * i = 2 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2984_298451
