import Mathlib

namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3786_378694

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p < 15 → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  is_composite 289 ∧
  has_no_small_prime_factors 289 ∧
  ∀ m, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3786_378694


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l3786_378673

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 10) 
  (h2 : current_speed = 2.5) : 
  speed_against_current + 2 * current_speed = 15 :=
by
  sorry

#check mans_speed_with_current

end NUMINAMATH_CALUDE_mans_speed_with_current_l3786_378673


namespace NUMINAMATH_CALUDE_wiper_generates_sector_l3786_378621

/-- Represents a car wiper -/
structure CarWiper :=
  (length : ℝ)

/-- Represents a windshield -/
structure Windshield :=
  (width : ℝ)
  (height : ℝ)

/-- Represents a sector on a windshield -/
structure Sector :=
  (angle : ℝ)
  (radius : ℝ)

/-- The action of a car wiper on a windshield -/
def wiper_action (w : CarWiper) (s : Windshield) : Sector :=
  sorry

/-- States that a line (represented by a car wiper) generates a surface (represented by a sector) -/
theorem wiper_generates_sector (w : CarWiper) (s : Windshield) :
  ∃ (sector : Sector), wiper_action w s = sector :=
sorry

end NUMINAMATH_CALUDE_wiper_generates_sector_l3786_378621


namespace NUMINAMATH_CALUDE_new_average_age_l3786_378677

/-- Calculates the new average age of a class after a student leaves and the teacher's age is included -/
theorem new_average_age 
  (initial_students : Nat) 
  (initial_average_age : ℝ) 
  (leaving_student_age : ℝ) 
  (teacher_age : ℝ) 
  (h1 : initial_students = 30)
  (h2 : initial_average_age = 10)
  (h3 : leaving_student_age = 11)
  (h4 : teacher_age = 41) : 
  let total_initial_age : ℝ := initial_students * initial_average_age
  let remaining_age : ℝ := total_initial_age - leaving_student_age
  let new_total_age : ℝ := remaining_age + teacher_age
  let new_count : Nat := initial_students
  new_total_age / new_count = 11 := by
  sorry


end NUMINAMATH_CALUDE_new_average_age_l3786_378677


namespace NUMINAMATH_CALUDE_smallest_two_digit_number_with_conditions_l3786_378662

theorem smallest_two_digit_number_with_conditions : ∃ n : ℕ,
  (n ≥ 10 ∧ n < 100) ∧  -- two-digit number
  (n % 3 = 0) ∧         -- divisible by 3
  (n % 4 = 0) ∧         -- divisible by 4
  (n % 5 = 4) ∧         -- remainder 4 when divided by 5
  (∀ m : ℕ, (m ≥ 10 ∧ m < 100) ∧ (m % 3 = 0) ∧ (m % 4 = 0) ∧ (m % 5 = 4) → n ≤ m) ∧
  n = 24 :=
by
  sorry

#check smallest_two_digit_number_with_conditions

end NUMINAMATH_CALUDE_smallest_two_digit_number_with_conditions_l3786_378662


namespace NUMINAMATH_CALUDE_notebook_duration_example_l3786_378685

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and daily page usage. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, used at a rate of 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebook_duration_example_l3786_378685


namespace NUMINAMATH_CALUDE_smallest_square_box_for_cards_l3786_378697

/-- Represents the dimensions of a business card -/
structure BusinessCard where
  width : ℕ
  length : ℕ

/-- Represents a square box -/
structure SquareBox where
  side : ℕ

/-- Checks if a square box can fit a whole number of business cards without overlapping -/
def canFitCards (box : SquareBox) (card : BusinessCard) : Prop :=
  (box.side % card.width = 0) ∧ (box.side % card.length = 0)

/-- Theorem: The smallest square box that can fit business cards of 5x7 cm has sides of 35 cm -/
theorem smallest_square_box_for_cards :
  let card := BusinessCard.mk 5 7
  let box := SquareBox.mk 35
  (canFitCards box card) ∧
  (∀ (smallerBox : SquareBox), smallerBox.side < box.side → ¬(canFitCards smallerBox card)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_box_for_cards_l3786_378697


namespace NUMINAMATH_CALUDE_remainder_of_product_l3786_378683

theorem remainder_of_product (a b c : ℕ) (hc : c ≥ 3) 
  (ha : a % c = 1) (hb : b % c = 2) : (a * b) % c = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_l3786_378683


namespace NUMINAMATH_CALUDE_product_inequality_l3786_378607

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3786_378607


namespace NUMINAMATH_CALUDE_existence_of_unequal_positive_numbers_l3786_378629

theorem existence_of_unequal_positive_numbers : ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≠ b ∧ a + b = a * b := by
  sorry

end NUMINAMATH_CALUDE_existence_of_unequal_positive_numbers_l3786_378629


namespace NUMINAMATH_CALUDE_julias_running_time_l3786_378681

/-- Julia's running time problem -/
theorem julias_running_time 
  (normal_mile_time : ℝ) 
  (extra_time_for_five_miles : ℝ) 
  (h1 : normal_mile_time = 10) 
  (h2 : extra_time_for_five_miles = 15) : 
  (5 * normal_mile_time + extra_time_for_five_miles) / 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_julias_running_time_l3786_378681


namespace NUMINAMATH_CALUDE_monotonicity_and_extrema_l3786_378690

def f (x : ℝ) : ℝ := 2*x^3 + 3*x^2 - 12*x + 1

theorem monotonicity_and_extrema :
  (∀ x < -2, (deriv f) x > 0) ∧
  (∀ x ∈ Set.Ioo (-2 : ℝ) 1, (deriv f) x < 0) ∧
  (∀ x > 1, (deriv f) x > 0) ∧
  IsLocalMax f (-2) ∧
  IsLocalMin f 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≤ f 5) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≥ f 1) ∧
  f 5 = 266 ∧
  f 1 = -6 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_extrema_l3786_378690


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3786_378604

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℚ),
  (x₁ = 3/2 ∧ x₂ = 1/2 ∧ x₃ = -5/2) ∧
  (8 * x₁^3 + 4 * x₁^2 - 34 * x₁ + 15 = 0) ∧
  (8 * x₂^3 + 4 * x₂^2 - 34 * x₂ + 15 = 0) ∧
  (8 * x₃^3 + 4 * x₃^2 - 34 * x₃ + 15 = 0) ∧
  (2 * x₁ - 4 * x₂ = 1) := by
  sorry

#check cubic_equation_roots

end NUMINAMATH_CALUDE_cubic_equation_roots_l3786_378604


namespace NUMINAMATH_CALUDE_speech_contest_probability_l3786_378670

/-- Represents the number of participants in the speech contest -/
def total_participants : ℕ := 10

/-- Represents the number of participants from Class 1 -/
def class1_participants : ℕ := 3

/-- Represents the number of participants from Class 2 -/
def class2_participants : ℕ := 2

/-- Represents the number of participants from other classes -/
def other_participants : ℕ := 5

/-- Calculates the probability of Class 1 students being consecutive and Class 2 students not being consecutive -/
def probability_class1_consecutive_class2_not : ℚ :=
  1 / 20

/-- Theorem stating the probability of the given event -/
theorem speech_contest_probability :
  probability_class1_consecutive_class2_not = 1 / 20 :=
by
  sorry

end NUMINAMATH_CALUDE_speech_contest_probability_l3786_378670


namespace NUMINAMATH_CALUDE_triangle_area_product_l3786_378639

/-- Given positive real numbers a and b, and a triangle in the first quadrant
    bounded by the coordinate axes and the line ax + by = 6 with area 6,
    prove that ab = 3. -/
theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ a * x + b * y = 6) →
  ((1 / 2) * (6 / a) * (6 / b) = 6) →
  a * b = 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_product_l3786_378639


namespace NUMINAMATH_CALUDE_angle_DAB_depends_on_triangle_l3786_378617

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the rectangle BCDE
structure Rectangle :=
  (B C D E : ℝ × ℝ)

-- Define the angle β (DAB)
def angle_DAB (tri : Triangle) (rect : Rectangle) : ℝ := sorry

-- State the theorem
theorem angle_DAB_depends_on_triangle (tri : Triangle) (rect : Rectangle) :
  tri.A ≠ tri.B ∧ tri.B ≠ tri.C ∧ tri.C ≠ tri.A →  -- Triangle inequality
  (tri.A.1 - tri.C.1)^2 + (tri.A.2 - tri.C.2)^2 = (tri.B.1 - tri.C.1)^2 + (tri.B.2 - tri.C.2)^2 →  -- CA = CB
  (rect.B = tri.B ∧ rect.C = tri.C) →  -- Rectangle is constructed on CB
  (rect.B.1 - rect.C.1)^2 + (rect.B.2 - rect.C.2)^2 > (rect.C.1 - rect.D.1)^2 + (rect.C.2 - rect.D.2)^2 →  -- BC > CD
  ∃ (f : Triangle → ℝ), angle_DAB tri rect = f tri :=
sorry

end NUMINAMATH_CALUDE_angle_DAB_depends_on_triangle_l3786_378617


namespace NUMINAMATH_CALUDE_hexadecagon_triangles_l3786_378680

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- A function to calculate the number of triangles in a regular polygon with n vertices -/
def num_triangles (n : ℕ) : ℕ := n.choose 3

/-- Theorem: The number of triangles in a regular hexadecagon is 560 -/
theorem hexadecagon_triangles : num_triangles n = 560 := by
  sorry

#eval num_triangles n

end NUMINAMATH_CALUDE_hexadecagon_triangles_l3786_378680


namespace NUMINAMATH_CALUDE_number_problem_l3786_378693

theorem number_problem : 
  ∃ x : ℝ, (1345 - (x / 20.04) = 1295) ∧ (x = 1002) := by sorry

end NUMINAMATH_CALUDE_number_problem_l3786_378693


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3786_378669

theorem solution_set_of_inequality (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) (-1/3 : ℝ)).Nonempty ∧ 
  (∀ y ∈ Set.Ioo (-2 : ℝ) (-1/3 : ℝ), (2*y - 1) / (3*y + 1) > 1) ∧
  (∀ z : ℝ, z ∉ Set.Ioo (-2 : ℝ) (-1/3 : ℝ) → (2*z - 1) / (3*z + 1) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3786_378669


namespace NUMINAMATH_CALUDE_ark5_ensures_metabolic_energy_needs_l3786_378602

-- Define the enzyme Ark5
structure Ark5 where
  activity : Bool

-- Define cancer cells
structure CancerCell where
  energy_balanced : Bool
  proliferating : Bool
  alive : Bool

-- Define the effect of Ark5 on cancer cells
def ark5_effect (a : Ark5) (c : CancerCell) : CancerCell :=
  { energy_balanced := a.activity
  , proliferating := true
  , alive := a.activity }

-- Theorem statement
theorem ark5_ensures_metabolic_energy_needs :
  ∀ (a : Ark5) (c : CancerCell),
    (¬a.activity → ¬c.energy_balanced) ∧
    (¬a.activity → c.proliferating) ∧
    (¬a.activity → ¬c.alive) →
    (a.activity → c.energy_balanced) :=
sorry

end NUMINAMATH_CALUDE_ark5_ensures_metabolic_energy_needs_l3786_378602


namespace NUMINAMATH_CALUDE_tournament_prize_orderings_l3786_378665

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := 5

/-- Represents whether the special reassignment rule is applied -/
def special_rule_applied : Bool := false

/-- Calculates the number of possible outcomes for a single match -/
def outcomes_per_match : ℕ := 2

/-- Theorem stating the number of different prize orderings in the tournament -/
theorem tournament_prize_orderings :
  (outcomes_per_match ^ num_matches : ℕ) = 32 :=
sorry

end NUMINAMATH_CALUDE_tournament_prize_orderings_l3786_378665


namespace NUMINAMATH_CALUDE_ship_distance_theorem_l3786_378699

/-- A function representing the square of the distance of a ship from an island over time. -/
def distance_squared (t : ℝ) : ℝ := 36 * t^2 - 84 * t + 49

/-- The theorem stating the distances at specific times given the initial conditions. -/
theorem ship_distance_theorem :
  (distance_squared 0 = 49) ∧
  (distance_squared 2 = 25) ∧
  (distance_squared 3 = 121) →
  (Real.sqrt (distance_squared 1) = 1) ∧
  (Real.sqrt (distance_squared 4) = 17) := by
  sorry

#check ship_distance_theorem

end NUMINAMATH_CALUDE_ship_distance_theorem_l3786_378699


namespace NUMINAMATH_CALUDE_determine_c_l3786_378667

theorem determine_c (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_determine_c_l3786_378667


namespace NUMINAMATH_CALUDE_distance_O_to_MN_l3786_378695

/-- The hyperbola C₁: 2x² - y² = 1 -/
def C₁ (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

/-- The ellipse C₂: 4x² + y² = 1 -/
def C₂ (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

/-- M is a point on C₁ -/
def M : ℝ × ℝ := sorry

/-- N is a point on C₂ -/
def N : ℝ × ℝ := sorry

/-- O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- OM is perpendicular to ON -/
def OM_perp_ON : Prop := sorry

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- The line MN -/
def lineMN : Set (ℝ × ℝ) := sorry

/-- Main theorem: The distance from O to MN is √3/3 -/
theorem distance_O_to_MN :
  C₁ M.1 M.2 → C₂ N.1 N.2 → OM_perp_ON →
  distancePointToLine O lineMN = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_distance_O_to_MN_l3786_378695


namespace NUMINAMATH_CALUDE_parabola_translation_l3786_378661

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in vertex form -/
def Parabola.equation (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * (x - p.h)^2 + p.k

/-- Vertical translation of a parabola -/
def verticalTranslate (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 3 0 0
  let p_translated := verticalTranslate p 3
  Parabola.equation p_translated x y ↔ y = 3 * x^2 + 3 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3786_378661


namespace NUMINAMATH_CALUDE_min_a_value_l3786_378675

open Real

-- Define the function f(x) = x/ln(x) - 1/(4x)
noncomputable def f (x : ℝ) : ℝ := x / log x - 1 / (4 * x)

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∃ x ∈ Set.Icc (exp 1) (exp 2), x / log x ≤ 1/4 + a*x) ↔ 
  a ≥ 1/2 - 1/(4 * (exp 2)^2) :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l3786_378675


namespace NUMINAMATH_CALUDE_smallest_sum_of_five_relatively_prime_numbers_l3786_378679

/-- A function that checks if two natural numbers are relatively prime -/
def isRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- A function that checks if a list of natural numbers are pairwise relatively prime -/
def arePairwiseRelativelyPrime (list : List ℕ) : Prop :=
  ∀ (i j : Fin list.length), i.val < j.val → isRelativelyPrime (list.get i) (list.get j)

/-- The main theorem statement -/
theorem smallest_sum_of_five_relatively_prime_numbers :
  ∃ (list : List ℕ),
    list.length = 5 ∧
    arePairwiseRelativelyPrime list ∧
    (∀ (sum : ℕ),
      (∃ (other_list : List ℕ),
        other_list.length = 5 ∧
        arePairwiseRelativelyPrime other_list ∧
        sum = other_list.sum) →
      list.sum ≤ sum) ∧
    list.sum = 4 :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_five_relatively_prime_numbers_l3786_378679


namespace NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l3786_378632

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l3786_378632


namespace NUMINAMATH_CALUDE_rectangle_area_l3786_378644

theorem rectangle_area (p : ℝ) (h : p > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧
  l / w = 5 / 2 ∧
  2 * (l + w) = p ∧
  l * w = (5 / 98) * p^2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3786_378644


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3786_378653

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -8
def c : ℝ := -7

-- Define the condition for m (not divisible by the square of any prime)
def is_square_free (m : ℕ) : Prop := ∀ p : ℕ, Prime p → (p^2 ∣ m) → False

-- Define the theorem
theorem quadratic_root_difference (m n : ℕ) (h1 : is_square_free m) (h2 : n > 0) :
  (((b^2 - 4*a*c).sqrt / (2*a)) = (m.sqrt / n)) → m + n = 56 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3786_378653


namespace NUMINAMATH_CALUDE_triangle_array_properties_l3786_378625

-- Define what it means to be a triangle array
def is_triangle_array (a b c : ℝ) : Prop :=
  0 < a ∧ a ≤ b ∧ b ≤ c ∧ a + b > c

-- Theorem statement
theorem triangle_array_properties 
  (p q r : ℝ) 
  (h : is_triangle_array p q r) : 
  (is_triangle_array (Real.sqrt p) (Real.sqrt q) (Real.sqrt r)) ∧ 
  (∃ p q r : ℝ, is_triangle_array p q r ∧ ¬is_triangle_array (p^2) (q^2) (r^2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_array_properties_l3786_378625


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3786_378619

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Product of the first n terms of a sequence -/
def ProductOfTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (fun acc i => acc * a (i + 1)) 1

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) 
  (h_geo : GeometricSequence a)
  (h_prop : a (m - 1) * a (m + 1) - 2 * a m = 0)
  (h_product : ProductOfTerms a (2 * m - 1) = 128) :
  m = 4 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l3786_378619


namespace NUMINAMATH_CALUDE_floor_abs_sum_l3786_378672

theorem floor_abs_sum : ⌊|(-5.7 : ℝ)|⌋ + |⌊(-5.7 : ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l3786_378672


namespace NUMINAMATH_CALUDE_ratio_part_to_whole_l3786_378646

theorem ratio_part_to_whole (N : ℝ) (x : ℝ) 
  (h1 : (1/4) * x * (2/5) * N = 14)
  (h2 : (2/5) * N = 168) : 
  x / N = 2/5 := by
sorry

end NUMINAMATH_CALUDE_ratio_part_to_whole_l3786_378646


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3786_378628

/-- For a quadratic equation x^2 + 4x + k = 0 to have real roots, k must be less than or equal to 4 -/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, x^2 + 4*x + k = 0) ↔ k ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l3786_378628


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3786_378682

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, ax^2 + bx - 2 = 0 ↔ x = -2 ∨ x = -1/4) → 
  a + b = -13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3786_378682


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l3786_378606

theorem solution_implies_k_value (k : ℝ) : 
  (k * (-3 + 4) - 2 * k - (-3) = 5) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l3786_378606


namespace NUMINAMATH_CALUDE_headphone_price_reduction_l3786_378611

theorem headphone_price_reduction (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ) :
  original_price = 120 →
  first_discount_rate = 0.25 →
  second_discount_rate = 0.1 →
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = 81 := by
sorry

end NUMINAMATH_CALUDE_headphone_price_reduction_l3786_378611


namespace NUMINAMATH_CALUDE_remaining_clothing_l3786_378620

theorem remaining_clothing (initial : ℕ) (donated_first : ℕ) (thrown_away : ℕ) : 
  initial = 100 →
  donated_first = 5 →
  thrown_away = 15 →
  initial - (donated_first + 3 * donated_first + thrown_away) = 65 := by
  sorry

end NUMINAMATH_CALUDE_remaining_clothing_l3786_378620


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3786_378659

theorem quadratic_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = a * x^2 + b * x + c) 
  (h_cond : f 0 = f 4 ∧ f 0 > f 1) :
  a > 0 ∧ 4 * a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3786_378659


namespace NUMINAMATH_CALUDE_linear_function_characterization_l3786_378601

/-- A function satisfying the Cauchy functional equation -/
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A monotonic function -/
def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- A function bounded between 0 and 1 -/
def is_bounded_01 (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ f x ∧ f x ≤ 1

/-- Main theorem: If f satisfies the given conditions, then it is linear -/
theorem linear_function_characterization (f : ℝ → ℝ)
  (h_additive : is_additive f)
  (h_monotonic : is_monotonic f)
  (h_bounded : is_bounded_01 f) :
  ∀ x, f x = x * f 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l3786_378601


namespace NUMINAMATH_CALUDE_inequality_iff_p_in_unit_interval_l3786_378687

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The proposition that pf(x) + qf(y) ≥ f(px + qy) for all real x, y -/
def inequality_holds (a b p q : ℝ) : Prop :=
  ∀ x y : ℝ, p * f a b x + q * f a b y ≥ f a b (p*x + q*y)

theorem inequality_iff_p_in_unit_interval (a b : ℝ) :
  ∀ p q : ℝ, p + q = 1 →
    (inequality_holds a b p q ↔ 0 ≤ p ∧ p ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_iff_p_in_unit_interval_l3786_378687


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l3786_378630

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 14 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l3786_378630


namespace NUMINAMATH_CALUDE_train_length_calculation_l3786_378657

/-- Calculates the length of a train given its speed, bridge length, and time to cross the bridge. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (crossing_time : Real) :
  let train_speed_ms : Real := train_speed * (1000 / 3600)
  let total_distance : Real := train_speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_speed = 45 ∧ bridge_length = 219.03 ∧ crossing_time = 30 →
  train_length = 155.97 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3786_378657


namespace NUMINAMATH_CALUDE_wedding_rsvp_theorem_l3786_378678

def total_guests : ℕ := 200
def yes_percent : ℚ := 83 / 100
def no_percent : ℚ := 9 / 100

theorem wedding_rsvp_theorem :
  (total_guests : ℚ) - (yes_percent * total_guests + no_percent * total_guests) = 16 := by
  sorry

end NUMINAMATH_CALUDE_wedding_rsvp_theorem_l3786_378678


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3786_378609

theorem rationalize_denominator : 
  7 / Real.sqrt 75 = (7 * Real.sqrt 3) / 15 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3786_378609


namespace NUMINAMATH_CALUDE_range_of_a_l3786_378608

/-- Given two statements p and q, where p: x^2 - 8x - 20 < 0 and q: x^2 - 2x + 1 - a^2 ≤ 0 with a > 0,
    and ¬p is a necessary but not sufficient condition for ¬q,
    prove that the range of values for the real number a is [9, +∞). -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) 
    (hp : ∀ x, p x ↔ x^2 - 8*x - 20 < 0)
    (hq : ∀ x, q x ↔ x^2 - 2*x + 1 - a^2 ≤ 0)
    (ha : a > 0)
    (hnec : ∀ x, ¬(p x) → ¬(q x))
    (hnsuff : ∃ x, ¬(q x) ∧ p x) :
  a ≥ 9 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l3786_378608


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l3786_378636

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg actual_num : ℝ) :
  n = 10 ∧ 
  initial_avg = 14 ∧ 
  correct_avg = 15 ∧ 
  actual_num = 36 →
  ∃ wrong_num : ℝ, 
    n * correct_avg - n * initial_avg = actual_num - wrong_num ∧ 
    wrong_num = 26 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l3786_378636


namespace NUMINAMATH_CALUDE_third_cat_weight_calculation_l3786_378651

/-- The weight of the third cat given the weights of the other cats and their average -/
def third_cat_weight (cat1 : ℝ) (cat2 : ℝ) (cat4 : ℝ) (avg_weight : ℝ) : ℝ :=
  4 * avg_weight - (cat1 + cat2 + cat4)

theorem third_cat_weight_calculation :
  third_cat_weight 12 12 9.3 12 = 14.7 := by
  sorry

end NUMINAMATH_CALUDE_third_cat_weight_calculation_l3786_378651


namespace NUMINAMATH_CALUDE_zilla_savings_theorem_l3786_378603

/-- Represents Zilla's monthly financial breakdown -/
structure ZillaFinances where
  total_earnings : ℝ
  rent_percentage : ℝ
  rent_amount : ℝ
  other_expenses_percentage : ℝ

/-- Calculates Zilla's savings based on her financial breakdown -/
def calculate_savings (z : ZillaFinances) : ℝ :=
  z.total_earnings - (z.rent_amount + z.total_earnings * z.other_expenses_percentage)

/-- Theorem stating Zilla's savings amount to $817 -/
theorem zilla_savings_theorem (z : ZillaFinances) 
  (h1 : z.rent_percentage = 0.07)
  (h2 : z.other_expenses_percentage = 0.5)
  (h3 : z.rent_amount = 133)
  (h4 : z.rent_amount = z.total_earnings * z.rent_percentage) :
  calculate_savings z = 817 := by
  sorry

#eval calculate_savings { total_earnings := 1900, rent_percentage := 0.07, rent_amount := 133, other_expenses_percentage := 0.5 }

end NUMINAMATH_CALUDE_zilla_savings_theorem_l3786_378603


namespace NUMINAMATH_CALUDE_bears_permutations_l3786_378622

theorem bears_permutations :
  Finset.card (Finset.univ.image (fun σ : Equiv.Perm (Fin 5) => σ)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_bears_permutations_l3786_378622


namespace NUMINAMATH_CALUDE_natural_number_representation_l3786_378623

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem natural_number_representation (n : ℕ) :
  ∃ x y z : ℕ, n = choose x 1 + choose y 2 + choose z 3 ∧
    ((0 ≤ x ∧ x < y ∧ y < z) ∨ (0 = x ∧ x = y ∧ y < z)) :=
  sorry

end NUMINAMATH_CALUDE_natural_number_representation_l3786_378623


namespace NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l3786_378612

/-- The number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

theorem students_play_both_football_and_cricket :
  students_play_both 450 325 175 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l3786_378612


namespace NUMINAMATH_CALUDE_right_prism_circumscribed_sphere_radius_l3786_378642

/-- A right prism with a square base -/
structure RightPrism where
  baseEdgeLength : ℝ
  sideEdgeLength : ℝ

/-- The sphere that circumscribes the right prism -/
structure CircumscribedSphere (p : RightPrism) where
  radius : ℝ
  contains_vertices : Prop  -- This represents the condition that all vertices lie on the sphere

/-- Theorem stating that for a right prism with base edge length 1 and side edge length 2,
    if all its vertices lie on a sphere, then the radius of that sphere is √6/2 -/
theorem right_prism_circumscribed_sphere_radius 
  (p : RightPrism) 
  (s : CircumscribedSphere p) 
  (h1 : p.baseEdgeLength = 1) 
  (h2 : p.sideEdgeLength = 2) 
  (h3 : s.contains_vertices) : 
  s.radius = Real.sqrt 6 / 2 := by
sorry

end NUMINAMATH_CALUDE_right_prism_circumscribed_sphere_radius_l3786_378642


namespace NUMINAMATH_CALUDE_lagrange_mean_value_theorem_l3786_378652

theorem lagrange_mean_value_theorem {f : ℝ → ℝ} {a b : ℝ} (hf : Differentiable ℝ f) (hab : a < b) :
  ∃ x₀ ∈ Set.Ioo a b, deriv f x₀ = (f a - f b) / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_lagrange_mean_value_theorem_l3786_378652


namespace NUMINAMATH_CALUDE_camila_hikes_per_week_l3786_378633

theorem camila_hikes_per_week (camila_hikes : ℕ) (amanda_factor : ℕ) (steven_extra : ℕ) (weeks : ℕ) : 
  camila_hikes = 7 →
  amanda_factor = 8 →
  steven_extra = 15 →
  weeks = 16 →
  ((amanda_factor * camila_hikes + steven_extra - camila_hikes) / weeks : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_camila_hikes_per_week_l3786_378633


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3786_378647

theorem geometric_series_ratio (r : ℝ) (h : r ≠ 1) :
  (∀ a : ℝ, a ≠ 0 → a / (1 - r) = 81 * (a * r^4) / (1 - r)) →
  r = 1/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3786_378647


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3786_378616

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) + sequence_a n

theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ sequence_a n ↔ 2^k ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3786_378616


namespace NUMINAMATH_CALUDE_abs_difference_given_sum_and_product_l3786_378643

theorem abs_difference_given_sum_and_product (a b : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a + b = 6) : 
  |a - b| = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_given_sum_and_product_l3786_378643


namespace NUMINAMATH_CALUDE_abs_fraction_sum_not_one_l3786_378689

theorem abs_fraction_sum_not_one (a b : ℝ) (h : a * b ≠ 0) :
  |a| / a + |b| / b ≠ 1 := by sorry

end NUMINAMATH_CALUDE_abs_fraction_sum_not_one_l3786_378689


namespace NUMINAMATH_CALUDE_isosceles_triangles_12_similar_l3786_378686

/-- An isosceles triangle with side ratio 1:2 -/
structure IsoscelesTriangle12 where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of another side
  h : a = 2 * b ∨ b = 2 * a  -- Condition for 1:2 ratio

/-- Similarity of isosceles triangles with 1:2 side ratio -/
theorem isosceles_triangles_12_similar (t1 t2 : IsoscelesTriangle12) :
  ∃ (k : ℝ), k > 0 ∧ 
    (t1.a = k * t2.a ∧ t1.b = k * t2.b) ∨
    (t1.a = k * t2.b ∧ t1.b = k * t2.a) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_12_similar_l3786_378686


namespace NUMINAMATH_CALUDE_product_sqrt_minus_square_eq_1988_l3786_378648

theorem product_sqrt_minus_square_eq_1988 :
  Real.sqrt (1988 * 1989 * 1990 * 1991 + 1) + (-1989^2) = 1988 := by
  sorry

end NUMINAMATH_CALUDE_product_sqrt_minus_square_eq_1988_l3786_378648


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3786_378638

theorem complex_number_quadrant (m : ℝ) (z : ℂ) 
  (h1 : 2/3 < m) (h2 : m < 1) (h3 : z = Complex.mk (3*m - 2) (m - 1)) : 
  0 < z.re ∧ z.re < 1 ∧ -1/3 < z.im ∧ z.im < 0 :=
by sorry

#check complex_number_quadrant

end NUMINAMATH_CALUDE_complex_number_quadrant_l3786_378638


namespace NUMINAMATH_CALUDE_square_sum_eight_l3786_378614

theorem square_sum_eight (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 16 = 10 * a * b) : 
  a^2 + b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_sum_eight_l3786_378614


namespace NUMINAMATH_CALUDE_hundred_billion_scientific_notation_l3786_378696

theorem hundred_billion_scientific_notation :
  (100000000000 : ℕ) = 1 * 10^11 :=
sorry

end NUMINAMATH_CALUDE_hundred_billion_scientific_notation_l3786_378696


namespace NUMINAMATH_CALUDE_kaydence_age_is_twelve_l3786_378658

/-- Represents the ages of family members and the total family age -/
structure Family where
  total_age : ℕ
  father_age : ℕ
  mother_age : ℕ
  brother_age : ℕ
  sister_age : ℕ

/-- Calculates Kaydence's age based on the family's ages -/
def kaydence_age (f : Family) : ℕ :=
  f.total_age - (f.father_age + f.mother_age + f.brother_age + f.sister_age)

/-- Theorem stating that Kaydence's age is 12 given the family conditions -/
theorem kaydence_age_is_twelve :
  ∀ (f : Family),
    f.total_age = 200 →
    f.father_age = 60 →
    f.mother_age = f.father_age - 2 →
    f.brother_age = f.father_age / 2 →
    f.sister_age = 40 →
    kaydence_age f = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaydence_age_is_twelve_l3786_378658


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3786_378668

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3786_378668


namespace NUMINAMATH_CALUDE_assignment_validity_l3786_378684

-- Define what constitutes a valid assignment statement
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String),
    stmt = var ++ " = " ++ expr ∧
    var.length > 0 ∧
    expr.length > 0 ∧
    var.all Char.isAlpha

theorem assignment_validity :
  is_valid_assignment "x = x + 1" ∧
  ¬is_valid_assignment "b =" ∧
  ¬is_valid_assignment "x = y = 10" ∧
  ¬is_valid_assignment "x + y = 10" :=
by sorry


end NUMINAMATH_CALUDE_assignment_validity_l3786_378684


namespace NUMINAMATH_CALUDE_cubic_roots_product_l3786_378654

theorem cubic_roots_product (r s t : ℝ) : 
  (r^3 - 20*r^2 + 18*r - 7 = 0) ∧ 
  (s^3 - 20*s^2 + 18*s - 7 = 0) ∧ 
  (t^3 - 20*t^2 + 18*t - 7 = 0) →
  (1 + r) * (1 + s) * (1 + t) = 46 := by
sorry


end NUMINAMATH_CALUDE_cubic_roots_product_l3786_378654


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3786_378676

theorem triangle_angle_problem (A B C : ℝ) : 
  A = 32 ∧ 
  C = 2 * A - 12 ∧ 
  B = 3 * A ∧ 
  A + B + C = 180 → 
  B = 96 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3786_378676


namespace NUMINAMATH_CALUDE_investment_B_is_72000_l3786_378650

/-- Represents the investment and profit distribution in a partnership. -/
structure Partnership where
  investA : ℕ
  investC : ℕ
  profitC : ℕ
  totalProfit : ℕ

/-- Calculates the investment of partner B given the partnership details. -/
def calculateInvestmentB (p : Partnership) : ℕ :=
  p.totalProfit * p.investC / p.profitC - p.investA - p.investC

/-- Theorem stating that given the specified partnership conditions, B's investment is 72000. -/
theorem investment_B_is_72000 (p : Partnership) 
  (h1 : p.investA = 27000)
  (h2 : p.investC = 81000)
  (h3 : p.profitC = 36000)
  (h4 : p.totalProfit = 80000) :
  calculateInvestmentB p = 72000 := by
  sorry

#eval calculateInvestmentB ⟨27000, 81000, 36000, 80000⟩

end NUMINAMATH_CALUDE_investment_B_is_72000_l3786_378650


namespace NUMINAMATH_CALUDE_train_length_l3786_378640

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 60 → time = 18 → 
  ∃ length : ℝ, abs (length - 300.06) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3786_378640


namespace NUMINAMATH_CALUDE_smallest_d_for_3150_l3786_378626

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem smallest_d_for_3150 : 
  (∃ d : ℕ+, (d.val > 0 ∧ is_perfect_square (3150 * d.val) ∧ 
    ∀ k : ℕ+, k.val > 0 → k.val < d.val → ¬ is_perfect_square (3150 * k.val))) → 
  (∃ d : ℕ+, d.val = 14 ∧ is_perfect_square (3150 * d.val) ∧ 
    ∀ k : ℕ+, k.val > 0 → k.val < d.val → ¬ is_perfect_square (3150 * k.val)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_for_3150_l3786_378626


namespace NUMINAMATH_CALUDE_expected_replanted_seeds_l3786_378635

/-- The expected number of replanted seeds when sowing 1000 seeds with a 0.9 germination probability -/
theorem expected_replanted_seeds :
  let germination_prob : ℝ := 0.9
  let total_seeds : ℕ := 1000
  let replant_per_fail : ℕ := 2
  let expected_non_germinating : ℝ := total_seeds * (1 - germination_prob)
  expected_non_germinating * replant_per_fail = 200 := by sorry

end NUMINAMATH_CALUDE_expected_replanted_seeds_l3786_378635


namespace NUMINAMATH_CALUDE_distinct_polynomials_differ_l3786_378610

-- Define the set X inductively
inductive X : (ℝ → ℝ) → Prop
  | base : X (λ x => x)
  | mul {r} : X r → X (λ x => x * r x)
  | add {r} : X r → X (λ x => x + (1 - x) * r x)

-- Define the theorem
theorem distinct_polynomials_differ (r s : ℝ → ℝ) (hr : X r) (hs : X s) (h_distinct : r ≠ s) :
  ∀ x, 0 < x → x < 1 → r x ≠ s x :=
sorry

end NUMINAMATH_CALUDE_distinct_polynomials_differ_l3786_378610


namespace NUMINAMATH_CALUDE_turkeys_to_ducks_ratio_l3786_378664

/-- Represents the number of birds on Mr. Valentino's farm -/
def total_birds : ℕ := 1800

/-- Represents the number of chickens on Mr. Valentino's farm -/
def chickens : ℕ := 200

/-- Represents the number of ducks on Mr. Valentino's farm -/
def ducks : ℕ := 2 * chickens

/-- Represents the number of turkeys on Mr. Valentino's farm -/
def turkeys : ℕ := total_birds - chickens - ducks

/-- Theorem stating the ratio of turkeys to ducks is 3:1 -/
theorem turkeys_to_ducks_ratio : 
  (turkeys : ℚ) / (ducks : ℚ) = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_turkeys_to_ducks_ratio_l3786_378664


namespace NUMINAMATH_CALUDE_derivative_of_f_l3786_378600

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 10) / x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = (1 - Real.log 10 * (Real.log x / Real.log 10)) / (x^2 * Real.log 10) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3786_378600


namespace NUMINAMATH_CALUDE_parabola_intersection_value_l3786_378698

theorem parabola_intersection_value (m : ℝ) : 
  (m^2 - m - 1 = 0) → (m^2 - m + 2008 = 2009) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_value_l3786_378698


namespace NUMINAMATH_CALUDE_root_transformation_l3786_378634

theorem root_transformation (p q r s : ℂ) : 
  (p^4 - 5*p^2 + 6 = 0) ∧ 
  (q^4 - 5*q^2 + 6 = 0) ∧ 
  (r^4 - 5*r^2 + 6 = 0) ∧ 
  (s^4 - 5*s^2 + 6 = 0) →
  ((p+q)/(r+s))^4 + 4*((p+q)/(r+s))^3 + 6*((p+q)/(r+s))^2 + 4*((p+q)/(r+s)) + 1 = 0 ∧
  ((p+r)/(q+s))^4 + 4*((p+r)/(q+s))^3 + 6*((p+r)/(q+s))^2 + 4*((p+r)/(q+s)) + 1 = 0 ∧
  ((p+s)/(q+r))^4 + 4*((p+s)/(q+r))^3 + 6*((p+s)/(q+r))^2 + 4*((p+s)/(q+r)) + 1 = 0 ∧
  ((q+r)/(p+s))^4 + 4*((q+r)/(p+s))^3 + 6*((q+r)/(p+s))^2 + 4*((q+r)/(p+s)) + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l3786_378634


namespace NUMINAMATH_CALUDE_min_value_inequality_l3786_378692

theorem min_value_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) : 
  (1 / (x + y)) + ((x + y) / z) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3786_378692


namespace NUMINAMATH_CALUDE_coin_distribution_l3786_378605

theorem coin_distribution (a d : ℤ) : 
  (a - 3*d) + (a - 2*d) = 58 ∧ 
  (a + d) + (a + 2*d) + (a + 3*d) = 60 →
  (a - 2*d = 28 ∧ a = 24) := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_l3786_378605


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3786_378660

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 19 * X + 53 = (X - 3) * q + 23 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3786_378660


namespace NUMINAMATH_CALUDE_paint_canvas_cost_ratio_l3786_378688

theorem paint_canvas_cost_ratio 
  (canvas_original : ℝ) 
  (paint_original : ℝ) 
  (canvas_decrease : ℝ) 
  (paint_decrease : ℝ) 
  (total_decrease : ℝ)
  (h1 : canvas_decrease = 0.4)
  (h2 : paint_decrease = 0.6)
  (h3 : total_decrease = 0.5599999999999999)
  (h4 : canvas_original > 0)
  (h5 : paint_original > 0)
  (h6 : (1 - paint_decrease) * paint_original + (1 - canvas_decrease) * canvas_original 
      = (1 - total_decrease) * (paint_original + canvas_original)) :
  paint_original / canvas_original = 4 := by
sorry

end NUMINAMATH_CALUDE_paint_canvas_cost_ratio_l3786_378688


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3786_378618

def M : Set ℝ := {x | 1 + x > 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3786_378618


namespace NUMINAMATH_CALUDE_unique_m_value_l3786_378655

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem unique_m_value : ∃! m : ℝ, 2 ∈ A m ∧ (∀ x ∈ A m, ∀ y ∈ A m, x = y → x = 0 ∨ x = m ∨ x = m^2 - 3*m + 2) :=
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l3786_378655


namespace NUMINAMATH_CALUDE_balloon_cost_theorem_l3786_378666

/-- Represents the cost of balloons for a person -/
structure BalloonCost where
  count : ℕ
  price : ℚ

/-- Calculates the total cost for a person's balloons -/
def totalCost (bc : BalloonCost) : ℚ :=
  bc.count * bc.price

theorem balloon_cost_theorem (fred sam dan : BalloonCost)
  (h_fred : fred = ⟨10, 1⟩)
  (h_sam : sam = ⟨46, (3/2)⟩)
  (h_dan : dan = ⟨16, (3/4)⟩) :
  totalCost fred + totalCost sam + totalCost dan = 91 := by
  sorry

end NUMINAMATH_CALUDE_balloon_cost_theorem_l3786_378666


namespace NUMINAMATH_CALUDE_curve_properties_l3786_378656

-- Define the curve C
def C (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Define the condition k ≠ -1
def k_not_neg_one (k : ℝ) : Prop := k ≠ -1

theorem curve_properties (k : ℝ) (h : k_not_neg_one k) :
  -- 1. C is always a circle
  (∃ (center_x center_y radius : ℝ), ∀ (x y : ℝ),
    C k x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
  -- The centers of the circles lie on the line y = 2x - 5
  (∃ (center_x center_y : ℝ), 
    (∀ (x y : ℝ), C k x y → (x - center_x)^2 + (y - center_y)^2 = (5*(k+1)^2)) ∧
    center_y = 2*center_x - 5) ∧
  -- 2. C passes through the fixed point (1, -3)
  C k 1 (-3) ∧
  -- 3. When C is tangent to the x-axis, k = 5 ± 3√5
  (∃ (x : ℝ), C k x 0 ∧ 
    (∀ (y : ℝ), y ≠ 0 → ¬(C k x y)) →
    (k = 5 + 3*Real.sqrt 5 ∨ k = 5 - 3*Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l3786_378656


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3786_378671

theorem arithmetic_calculation : 1 + 2 * 3 - 4 + 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3786_378671


namespace NUMINAMATH_CALUDE_card_game_properties_l3786_378641

/-- A card collection game with 3 colors -/
structure CardGame where
  colors : Nat
  cards_per_color : Nat

/-- The probability of not collecting 3 cards of the same color after 4 purchases -/
def prob_not_three_same (game : CardGame) : ℚ :=
  2 / 3

/-- The distribution of X (number of purchases before collecting 3 cards of the same color) -/
def distribution_X (game : CardGame) (x : Nat) : ℚ :=
  match x with
  | 3 => 1 / 9
  | 4 => 2 / 9
  | 5 => 8 / 27
  | 6 => 20 / 81
  | 7 => 10 / 81
  | _ => 0

/-- The expectation of X -/
def expectation_X (game : CardGame) : ℚ :=
  409 / 81

/-- Main theorem about the card collection game -/
theorem card_game_properties (game : CardGame) 
    (h1 : game.colors = 3) 
    (h2 : game.cards_per_color = 3) : 
  prob_not_three_same game = 2 / 3 ∧ 
  (∀ x, distribution_X game x = match x with
                                | 3 => 1 / 9
                                | 4 => 2 / 9
                                | 5 => 8 / 27
                                | 6 => 20 / 81
                                | 7 => 10 / 81
                                | _ => 0) ∧
  expectation_X game = 409 / 81 := by
  sorry

end NUMINAMATH_CALUDE_card_game_properties_l3786_378641


namespace NUMINAMATH_CALUDE_problem_solution_l3786_378691

/-- Given the conditions of the problem, prove that x · z = 4.5 -/
theorem problem_solution :
  ∀ x y z : ℝ,
  (∃ x₀ y₀ z₀ : ℝ, x₀ = 2*y₀ ∧ z₀ = x₀ ∧ x₀*y₀ = y₀*z₀) →  -- Initial condition
  z = x/2 →
  x*y = y^2 →
  y = 3 →
  x*z = 4.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3786_378691


namespace NUMINAMATH_CALUDE_sandys_shopping_l3786_378613

/-- Sandy's shopping problem -/
theorem sandys_shopping (total : ℝ) (spent_percentage : ℝ) (remaining : ℝ) : 
  total = 300 → 
  spent_percentage = 30 → 
  remaining = total * (1 - spent_percentage / 100) → 
  remaining = 210 := by
sorry

end NUMINAMATH_CALUDE_sandys_shopping_l3786_378613


namespace NUMINAMATH_CALUDE_class_size_problem_l3786_378674

/-- Given information about class sizes, prove the size of Class C -/
theorem class_size_problem (size_B : ℕ) (size_A : ℕ) (size_C : ℕ)
  (h1 : size_A = 2 * size_B)
  (h2 : size_A = size_C / 3)
  (h3 : size_B = 20) :
  size_C = 120 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l3786_378674


namespace NUMINAMATH_CALUDE_bracket_removal_l3786_378663

theorem bracket_removal (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_bracket_removal_l3786_378663


namespace NUMINAMATH_CALUDE_miller_rabin_composite_detection_probability_l3786_378627

/-- Miller-Rabin test function that returns true if n is probably prime, false if n is definitely composite -/
def miller_rabin_test (n : ℕ) (a : ℕ) : Bool :=
  sorry

/-- The probability that the Miller-Rabin test correctly identifies a composite number -/
def miller_rabin_probability (n : ℕ) : ℝ :=
  sorry

theorem miller_rabin_composite_detection_probability 
  (n : ℕ) (h : ¬ Nat.Prime n) : 
  miller_rabin_probability n ≥ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_miller_rabin_composite_detection_probability_l3786_378627


namespace NUMINAMATH_CALUDE_zero_clever_numbers_l3786_378631

def is_zero_clever (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = a * 1000 + b * 10 + c ∧
    n = (a * 100 + b * 10 + c) * 9 ∧
    a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9

theorem zero_clever_numbers :
  {n : ℕ | is_zero_clever n} = {2025, 4050, 6075} :=
sorry

end NUMINAMATH_CALUDE_zero_clever_numbers_l3786_378631


namespace NUMINAMATH_CALUDE_probability_reach_top_correct_l3786_378624

/-- The probability of reaching the top floor using only open doors in a building with n floors and two staircases, where half the doors are randomly locked. -/
def probability_reach_top (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (Nat.choose (2 * (n - 1)) (n - 1))

/-- Theorem stating the probability of reaching the top floor using only open doors. -/
theorem probability_reach_top_correct (n : ℕ) (h : n > 1) :
  probability_reach_top n = (2 ^ (n - 1)) / (Nat.choose (2 * (n - 1)) (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_probability_reach_top_correct_l3786_378624


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_equals_6020_l3786_378615

theorem inequality_holds_iff_k_equals_6020 :
  ∃ (k : ℝ), k > 0 ∧
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a / (c + k * b) + b / (a + k * c) + c / (b + k * a) ≥ 1 / 2007)) ∧
  k = 6020 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_equals_6020_l3786_378615


namespace NUMINAMATH_CALUDE_siblings_selection_probability_l3786_378637

theorem siblings_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_ritu : ℚ) 
  (h_ram : p_ram = 3 / 7) 
  (h_ravi : p_ravi = 1 / 5) 
  (h_ritu : p_ritu = 2 / 9) : 
  p_ram * p_ravi * p_ritu = 2 / 105 := by
  sorry

end NUMINAMATH_CALUDE_siblings_selection_probability_l3786_378637


namespace NUMINAMATH_CALUDE_repeating_decimal_value_l3786_378645

def repeating_decimal : ℚ := 33 / 99999

theorem repeating_decimal_value : 
  (10^5 - 10^3 : ℚ) * repeating_decimal = 32.67 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_value_l3786_378645


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_l3786_378649

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def only_zero_and_one (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting only of 0s and 1s that is divisible by 15 -/
def smallest_T : ℕ := sorry

theorem smallest_divisible_by_15 :
  smallest_T > 0 ∧
  only_zero_and_one smallest_T ∧
  smallest_T % 15 = 0 ∧
  smallest_T / 15 = 74 ∧
  ∀ n : ℕ, n > 0 → only_zero_and_one n → n % 15 = 0 → n ≥ smallest_T :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_l3786_378649
