import Mathlib

namespace NUMINAMATH_CALUDE_duck_flight_days_l331_33152

/-- The number of days it takes for a duck to fly south in winter. -/
def days_south : ℕ := sorry

/-- The number of days it takes for a duck to fly north in summer. -/
def days_north : ℕ := 2 * days_south

/-- The number of days it takes for a duck to fly east in spring. -/
def days_east : ℕ := 60

/-- The total number of days the duck flies during winter, summer, and spring. -/
def total_days : ℕ := 180

/-- Theorem stating that the number of days it takes for the duck to fly south in winter is 40. -/
theorem duck_flight_days : days_south = 40 := by
  sorry

end NUMINAMATH_CALUDE_duck_flight_days_l331_33152


namespace NUMINAMATH_CALUDE_pete_backward_speed_l331_33156

/-- Represents the speeds of various activities in miles per hour -/
structure Speeds where
  susan_forward : ℝ
  pete_backward : ℝ
  tracy_cartwheel : ℝ
  pete_hands : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : Speeds) : Prop :=
  s.pete_backward = 3 * s.susan_forward ∧
  s.tracy_cartwheel = 2 * s.susan_forward ∧
  s.pete_hands = 1/4 * s.tracy_cartwheel ∧
  s.pete_hands = 2

/-- The theorem stating Pete's backward walking speed -/
theorem pete_backward_speed (s : Speeds) 
  (h : problem_conditions s) : s.pete_backward = 12 := by
  sorry


end NUMINAMATH_CALUDE_pete_backward_speed_l331_33156


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l331_33183

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  ¬∃ (g h : Polynomial ℤ), (Polynomial.degree g ≥ 1) ∧ (Polynomial.degree h ≥ 1) ∧
  (x^n + 5 * x^(n-1) + 3 : Polynomial ℤ) = g * h :=
by sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l331_33183


namespace NUMINAMATH_CALUDE_courtyard_length_l331_33137

/-- The length of a rectangular courtyard given specific conditions -/
theorem courtyard_length (width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) : 
  width = 20 → 
  num_stones = 100 → 
  stone_length = 4 → 
  stone_width = 2 → 
  (width * (num_stones * stone_length * stone_width / width) : ℝ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l331_33137


namespace NUMINAMATH_CALUDE_radius_of_third_circle_l331_33148

structure Triangle :=
  (a b c : ℝ)

structure Circle :=
  (radius : ℝ)

def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b

def isInscribed (c : Circle) (t : Triangle) : Prop :=
  sorry

def isTangent (c1 c2 : Circle) (t : Triangle) : Prop :=
  sorry

theorem radius_of_third_circle (t : Triangle) (q1 q2 q3 : Circle) :
  t.a = 78 →
  t.b = 78 →
  t.c = 60 →
  isIsosceles t →
  isInscribed q1 t →
  isTangent q2 q1 t →
  isTangent q3 q2 t →
  q3.radius = 320 / 81 :=
sorry

end NUMINAMATH_CALUDE_radius_of_third_circle_l331_33148


namespace NUMINAMATH_CALUDE_distinct_tetrahedra_count_l331_33172

/-- A type representing a thin rod with a length -/
structure Rod where
  length : ℝ
  positive : length > 0

/-- A type representing a set of 6 rods -/
structure SixRods where
  rods : Fin 6 → Rod
  distinct : ∀ i j, i ≠ j → rods i ≠ rods j

/-- A predicate stating that any three rods can form a triangle -/
def can_form_triangle (sr : SixRods) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (sr.rods i).length + (sr.rods j).length > (sr.rods k).length

/-- A type representing a tetrahedral edge framework -/
structure Tetrahedron where
  edges : Fin 6 → Rod

/-- A function to count distinct tetrahedral edge frameworks -/
noncomputable def count_distinct_tetrahedra (sr : SixRods) : ℕ := sorry

/-- The main theorem -/
theorem distinct_tetrahedra_count (sr : SixRods) 
  (h : can_form_triangle sr) : 
  count_distinct_tetrahedra sr = 30 := by sorry

end NUMINAMATH_CALUDE_distinct_tetrahedra_count_l331_33172


namespace NUMINAMATH_CALUDE_camel_cost_l331_33166

/-- Represents the cost of different animals in Rupees -/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ
  giraffe : ℝ
  zebra : ℝ
  llama : ℝ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  3 * costs.elephant = 5 * costs.giraffe ∧
  8 * costs.giraffe = 12 * costs.zebra ∧
  20 * costs.zebra = 7 * costs.llama ∧
  10 * costs.elephant = 120000

theorem camel_cost (costs : AnimalCosts) :
  problem_conditions costs → costs.camel = 4800 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l331_33166


namespace NUMINAMATH_CALUDE_lowest_divisible_by_one_and_two_l331_33138

theorem lowest_divisible_by_one_and_two : 
  ∃ n : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2 → k ∣ n) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2 → k ∣ m) → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_lowest_divisible_by_one_and_two_l331_33138


namespace NUMINAMATH_CALUDE_inequality_proof_l331_33133

theorem inequality_proof (x : ℝ) (h : x ≥ 5) :
  Real.sqrt (x - 2) - Real.sqrt (x - 3) < Real.sqrt (x - 4) - Real.sqrt (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l331_33133


namespace NUMINAMATH_CALUDE_hyperbola_properties_l331_33157

/-- Given a hyperbola with specific properties, prove its equation and a property of its intersection with a line --/
theorem hyperbola_properties (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let e : ℝ := Real.sqrt 3
  let vertex : ℝ × ℝ := (Real.sqrt 3, 0)
  ∀ x y, C x y →
    (∃ c, c > 0 ∧ c^2 = a^2 + b^2 ∧ c / a = e) →
    (C (Real.sqrt 3) 0) →
    (∃ F₂ : ℝ × ℝ, F₂.1 > 0 ∧
      (∀ x y, (y - F₂.2) = Real.sqrt 3 / 3 * (x - F₂.1) →
        C x y →
        ∃ A B : ℝ × ℝ, A ≠ B ∧ C A.1 A.2 ∧ C B.1 B.2 ∧
          Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 * Real.sqrt 3 / 5)) →
  C x y ↔ x^2 / 3 - y^2 / 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l331_33157


namespace NUMINAMATH_CALUDE_locus_of_tangent_circles_l331_33143

/-- The locus of centers of circles externally tangent to a given circle and a line -/
theorem locus_of_tangent_circles (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    ((x - 0)^2 + (y - 3)^2)^(1/2) = r + 1 ∧
    y = r) → 
  ∃ (a b c : ℝ), a ≠ 0 ∧ (y - b)^2 = 4 * a * (x - c) :=
sorry

end NUMINAMATH_CALUDE_locus_of_tangent_circles_l331_33143


namespace NUMINAMATH_CALUDE_expand_product_l331_33191

theorem expand_product (x : ℝ) : (3*x - 4) * (2*x + 9) = 6*x^2 + 19*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l331_33191


namespace NUMINAMATH_CALUDE_subtract_like_terms_l331_33119

theorem subtract_like_terms (a : ℝ) : 2 * a - 3 * a = -a := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l331_33119


namespace NUMINAMATH_CALUDE_trig_identity_proof_l331_33112

theorem trig_identity_proof : 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) - 
  Real.cos (75 * π / 180) * Real.sin (105 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l331_33112


namespace NUMINAMATH_CALUDE_square_sum_equals_five_l331_33186

theorem square_sum_equals_five (a b : ℝ) 
  (h1 : a^3 - 3*a*b^2 = 11) 
  (h2 : b^3 - 3*a^2*b = 2) : 
  a^2 + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_square_sum_equals_five_l331_33186


namespace NUMINAMATH_CALUDE_variable_value_l331_33147

theorem variable_value (x : ℝ) : 5 / (4 + 1 / x) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_variable_value_l331_33147


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l331_33144

theorem percentage_of_male_employees 
  (total_employees : ℕ)
  (males_below_50 : ℕ)
  (h1 : total_employees = 800)
  (h2 : males_below_50 = 120)
  (h3 : (males_below_50 : ℝ) = 0.6 * (total_employees * (percentage_males / 100))) :
  percentage_males = 25 := by
  sorry

#check percentage_of_male_employees

end NUMINAMATH_CALUDE_percentage_of_male_employees_l331_33144


namespace NUMINAMATH_CALUDE_symmetric_function_properties_l331_33127

def f (x m : ℝ) : ℝ := 2 * |x| + |2*x - m|

theorem symmetric_function_properties (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x : ℝ, f x m = f (2 - x) m) :
  (m = 4) ∧ 
  (∀ x : ℝ, f x m ≥ 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = m → 1/a + 4/b ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_properties_l331_33127


namespace NUMINAMATH_CALUDE_max_boats_in_river_l331_33122

theorem max_boats_in_river (river_width : ℝ) (boat_width : ℝ) (min_space : ℝ) :
  river_width = 42 →
  boat_width = 3 →
  min_space = 2 →
  ⌊(river_width - 2 * min_space) / (boat_width + 2 * min_space)⌋ = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_boats_in_river_l331_33122


namespace NUMINAMATH_CALUDE_minimum_a_value_l331_33195

theorem minimum_a_value (a : ℝ) : 
  (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ↔ 
  a ≥ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_a_value_l331_33195


namespace NUMINAMATH_CALUDE_line_properties_l331_33190

/-- The line l₁ with equation (m + 1)x - (m - 3)y - 8 = 0 where m ∈ ℝ --/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x - (m - 3) * y - 8 = 0

/-- The line l₂ parallel to l₁ passing through the origin --/
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x - (m - 3) * y = 0

theorem line_properties :
  (∀ m : ℝ, l₁ m 2 2) ∧ 
  (∀ x y : ℝ, x + y = 0 → (∀ m : ℝ, l₂ m x y) ∧ 
    ∀ a b : ℝ, l₂ m a b → (a^2 + b^2 ≤ x^2 + y^2)) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l331_33190


namespace NUMINAMATH_CALUDE_max_y_value_l331_33106

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) : 
  y ≤ 27 ∧ ∃ (x' y' : ℤ), x' * y' + 7 * x' + 6 * y' = -8 ∧ y' = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l331_33106


namespace NUMINAMATH_CALUDE_first_player_wins_l331_33120

-- Define the game state
structure GameState where
  hour : Nat

-- Define player moves
def firstPlayerMove (state : GameState) : GameState :=
  { hour := (state.hour + 2) % 12 }

def secondPlayerMoveInitial (state : GameState) : GameState :=
  { hour := 5 }

def secondPlayerMoveSubsequent (state : GameState) (move : Nat) : GameState :=
  { hour := (state.hour + move) % 12 }

def firstPlayerMoveSubsequent (state : GameState) (move : Nat) : GameState :=
  { hour := (state.hour + move) % 12 }

-- Define the game sequence
def gameSequence (secondPlayerLastMove : Nat) : GameState :=
  let initial := { hour := 0 }  -- 12 o'clock
  let afterFirstMove := firstPlayerMove initial
  let afterSecondMove := secondPlayerMoveInitial afterFirstMove
  let afterThirdMove := firstPlayerMoveSubsequent afterSecondMove 3
  let afterFourthMove := secondPlayerMoveSubsequent afterThirdMove secondPlayerLastMove
  let finalState := 
    if secondPlayerLastMove = 2 then
      firstPlayerMoveSubsequent afterFourthMove 3
    else
      firstPlayerMoveSubsequent afterFourthMove 2

  finalState

-- Theorem statement
theorem first_player_wins (secondPlayerLastMove : Nat) 
  (h : secondPlayerLastMove = 2 ∨ secondPlayerLastMove = 3) : 
  (gameSequence secondPlayerLastMove).hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l331_33120


namespace NUMINAMATH_CALUDE_shopkeeper_total_cards_l331_33115

/-- The number of cards in a standard deck of playing cards -/
def standard_deck_size : ℕ := 52

/-- The number of complete decks the shopkeeper has -/
def complete_decks : ℕ := 6

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := 7

/-- Theorem: The total number of cards the shopkeeper has is 319 -/
theorem shopkeeper_total_cards : 
  complete_decks * standard_deck_size + additional_cards = 319 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_total_cards_l331_33115


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l331_33161

theorem min_value_expression (a b : ℤ) (h : a > b) :
  (a + 2*b) / (a - b) + (a - b) / (a + 2*b) ≥ 2 :=
sorry

theorem min_value_achievable :
  ∃ (a b : ℤ), a > b ∧ (a + 2*b) / (a - b) + (a - b) / (a + 2*b) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l331_33161


namespace NUMINAMATH_CALUDE_uniform_price_calculation_l331_33111

/-- Represents the price of the uniform in Rupees -/
def uniform_price : ℝ := 200

/-- Represents the full year service pay in Rupees -/
def full_year_pay : ℝ := 800

/-- Represents the actual service duration in months -/
def actual_service : ℝ := 9

/-- Represents the full year service duration in months -/
def full_year : ℝ := 12

/-- Represents the actual payment received in Rupees -/
def actual_payment : ℝ := 400

theorem uniform_price_calculation :
  uniform_price = full_year_pay * (actual_service / full_year) - actual_payment :=
by sorry

end NUMINAMATH_CALUDE_uniform_price_calculation_l331_33111


namespace NUMINAMATH_CALUDE_round_4995000_to_million_l331_33140

/-- Round a natural number to the nearest million -/
def round_to_million (n : ℕ) : ℕ :=
  if n % 1000000 ≥ 500000 then
    ((n + 500000) / 1000000) * 1000000
  else
    (n / 1000000) * 1000000

/-- Theorem: Rounding 4995000 to the nearest million equals 5000000 -/
theorem round_4995000_to_million :
  round_to_million 4995000 = 5000000 := by
  sorry

end NUMINAMATH_CALUDE_round_4995000_to_million_l331_33140


namespace NUMINAMATH_CALUDE_minimal_sum_for_equal_last_digits_l331_33185

theorem minimal_sum_for_equal_last_digits (m n : ℕ) : 
  n > m ∧ m ≥ 1 ∧ 
  (1978^m : ℕ) % 1000 = (1978^n : ℕ) % 1000 ∧
  (∀ m' n' : ℕ, n' > m' ∧ m' ≥ 1 ∧ 
    (1978^m' : ℕ) % 1000 = (1978^n' : ℕ) % 1000 → 
    m + n ≤ m' + n') →
  m = 3 ∧ n = 103 := by
sorry

end NUMINAMATH_CALUDE_minimal_sum_for_equal_last_digits_l331_33185


namespace NUMINAMATH_CALUDE_linda_basketball_scores_l331_33160

theorem linda_basketball_scores (first_seven : List Nat) 
  (h1 : first_seven = [5, 6, 4, 7, 3, 2, 6])
  (h2 : first_seven.length = 7)
  (eighth_game : Nat) (ninth_game : Nat)
  (h3 : eighth_game < 10)
  (h4 : ninth_game < 10)
  (h5 : (first_seven.sum + eighth_game) % 8 = 0)
  (h6 : (first_seven.sum + eighth_game + ninth_game) % 9 = 0) :
  eighth_game * ninth_game = 35 := by
sorry

end NUMINAMATH_CALUDE_linda_basketball_scores_l331_33160


namespace NUMINAMATH_CALUDE_cylinder_painted_area_l331_33168

/-- The total painted area of a cylinder with given dimensions and painting conditions -/
theorem cylinder_painted_area (h r : ℝ) (paint_percent : ℝ) : 
  h = 15 → r = 5 → paint_percent = 0.75 → 
  (2 * π * r^2) + (paint_percent * 2 * π * r * h) = 162.5 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_painted_area_l331_33168


namespace NUMINAMATH_CALUDE_total_caps_production_l331_33153

/-- The total number of caps produced over four weeks, given the production
    of the first three weeks and the fourth week being the average of the first three. -/
theorem total_caps_production
  (week1 : ℕ)
  (week2 : ℕ)
  (week3 : ℕ)
  (h1 : week1 = 320)
  (h2 : week2 = 400)
  (h3 : week3 = 300) :
  week1 + week2 + week3 + (week1 + week2 + week3) / 3 = 1360 := by
  sorry

#eval 320 + 400 + 300 + (320 + 400 + 300) / 3

end NUMINAMATH_CALUDE_total_caps_production_l331_33153


namespace NUMINAMATH_CALUDE_vector_simplification_l331_33158

variable (V : Type*) [AddCommGroup V]
variable (A B C D : V)

theorem vector_simplification (h : A + (B - A) + (C - B) = C) :
  (B - A) + (C - B) - (C - A) - (D - C) = C - D :=
sorry

end NUMINAMATH_CALUDE_vector_simplification_l331_33158


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l331_33125

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * a 2^2 - 10 * a 2 + 3 = 0) →
  (3 * a 6^2 - 10 * a 6 + 3 = 0) →
  (1 / a 2 + 1 / a 6 + a 4^2 = 13/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l331_33125


namespace NUMINAMATH_CALUDE_line_properties_l331_33162

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := ∀ x y, l₁ a x y → l₂ a b x y

-- Define perpendicular lines
def perpendicular (a b : ℝ) : Prop := ∀ x y, l₁ a x y → l₂ a b x y

theorem line_properties (a b : ℝ) :
  (b = -2 ∧ parallel a b → a = 1 ∨ a = -1) ∧
  (perpendicular a b → ∀ c d : ℝ, perpendicular c d → |a * b| ≤ |c * d| ∧ |a * b| = 2) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l331_33162


namespace NUMINAMATH_CALUDE_intersection_empty_union_real_l331_33113

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 1}

-- Theorem 1
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ a > 3 := by sorry

-- Theorem 2
theorem union_real (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_union_real_l331_33113


namespace NUMINAMATH_CALUDE_mike_taller_than_mark_l331_33159

/-- Converts feet and inches to total inches -/
def heightToInches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- The height difference between two people in inches -/
def heightDifference (height1 : ℕ) (height2 : ℕ) : ℕ :=
  max height1 height2 - min height1 height2

theorem mike_taller_than_mark : 
  let markHeight := heightToInches 5 3
  let mikeHeight := heightToInches 6 1
  heightDifference markHeight mikeHeight = 10 := by
sorry

end NUMINAMATH_CALUDE_mike_taller_than_mark_l331_33159


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l331_33117

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l331_33117


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l331_33123

theorem minimum_value_theorem (x y m : ℝ) :
  y ≥ 1 →
  y ≤ 2 * x - 1 →
  x + y ≤ m →
  (∀ x' y' : ℝ, y' ≥ 1 → y' ≤ 2 * x' - 1 → x' + y' ≤ m → x - y ≤ x' - y') →
  x - y = 0 →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l331_33123


namespace NUMINAMATH_CALUDE_linear_regression_estimate_l331_33197

/-- Given a linear regression equation y = 0.50x - 0.81, prove that when x = 25, y = 11.69 -/
theorem linear_regression_estimate (x y : ℝ) : 
  y = 0.50 * x - 0.81 → x = 25 → y = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_estimate_l331_33197


namespace NUMINAMATH_CALUDE_inequality_solution_set_l331_33132

theorem inequality_solution_set : 
  {x : ℝ | 2 ≥ (1 / (x - 1))} = Set.Iic 1 ∪ Set.Ici (3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l331_33132


namespace NUMINAMATH_CALUDE_count_numbers_with_three_in_range_l331_33176

def count_numbers_with_three (lower_bound upper_bound : ℕ) : ℕ :=
  sorry

theorem count_numbers_with_three_in_range : 
  count_numbers_with_three 200 499 = 138 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_three_in_range_l331_33176


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l331_33169

theorem jose_bottle_caps (initial : Real) (given_away : Real) (remaining : Real) : 
  initial = 7.0 → given_away = 2.0 → remaining = initial - given_away → remaining = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l331_33169


namespace NUMINAMATH_CALUDE_inequality_solution_l331_33151

theorem inequality_solution (x : ℝ) : 
  (x - 1) / ((x - 3)^2) < 0 ↔ x < 1 ∧ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l331_33151


namespace NUMINAMATH_CALUDE_inequalities_and_range_l331_33100

theorem inequalities_and_range :
  (∀ x : ℝ, x > 1 → 2 * Real.log x < x - 1/x) ∧
  (∀ a : ℝ, a > 0 → (∀ t : ℝ, t > 0 → (1 + a/t) * Real.log (1 + t) > a) ↔ 0 < a ∧ a ≤ 2) ∧
  ((9/10 : ℝ)^19 < 1/Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_and_range_l331_33100


namespace NUMINAMATH_CALUDE_color_assignment_theorem_l331_33167

def numbers : List ℕ := List.range 13 |>.map (· + 13)

structure ColorAssignment where
  black : ℕ
  red : List ℕ
  blue : List ℕ
  yellow : List ℕ
  green : List ℕ

def isValidAssignment (ca : ColorAssignment) : Prop :=
  ca.black ∈ numbers ∧
  ca.red.length = 3 ∧ ca.red.all (· ∈ numbers) ∧
  ca.blue.length = 3 ∧ ca.blue.all (· ∈ numbers) ∧
  ca.yellow.length = 3 ∧ ca.yellow.all (· ∈ numbers) ∧
  ca.green.length = 3 ∧ ca.green.all (· ∈ numbers) ∧
  ca.red.sum = ca.blue.sum ∧ ca.blue.sum = ca.yellow.sum ∧ ca.yellow.sum = ca.green.sum ∧
  13 ∈ ca.red ∧ 15 ∈ ca.yellow ∧ 23 ∈ ca.blue ∧
  (ca.black :: ca.red ++ ca.blue ++ ca.yellow ++ ca.green).toFinset = numbers.toFinset

theorem color_assignment_theorem (ca : ColorAssignment) 
  (h : isValidAssignment ca) : 
  ca.black = 19 ∧ ca.green = [14, 21, 22] := by
  sorry

#check color_assignment_theorem

end NUMINAMATH_CALUDE_color_assignment_theorem_l331_33167


namespace NUMINAMATH_CALUDE_simplify_polynomial_expression_l331_33193

theorem simplify_polynomial_expression (x : ℝ) :
  6 * x^2 + 4 * x + 9 - (7 - 5 * x - 9 * x^3 + 8 * x^2) = 9 * x^3 - 2 * x^2 + 9 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_expression_l331_33193


namespace NUMINAMATH_CALUDE_A_initial_investment_l331_33175

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's contribution to the capital in rupees -/
def B_contribution : ℝ := 15750

/-- Represents the number of months A was in the business -/
def A_months : ℝ := 12

/-- Represents the number of months B was in the business -/
def B_months : ℝ := 4

/-- Represents the ratio of profit division for A -/
def A_profit_ratio : ℝ := 2

/-- Represents the ratio of profit division for B -/
def B_profit_ratio : ℝ := 3

/-- Theorem stating that A's initial investment is 1750 rupees -/
theorem A_initial_investment : 
  A_investment = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_A_initial_investment_l331_33175


namespace NUMINAMATH_CALUDE_betty_morning_flies_l331_33118

/-- The number of flies Betty caught in the morning -/
def morning_flies : ℕ := 5

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty caught in the afternoon -/
def afternoon_flies : ℕ := 6

/-- The number of flies that escaped -/
def escaped_flies : ℕ := 1

/-- The number of additional flies Betty needs -/
def additional_flies_needed : ℕ := 4

theorem betty_morning_flies :
  morning_flies = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_betty_morning_flies_l331_33118


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l331_33192

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l331_33192


namespace NUMINAMATH_CALUDE_cost_price_calculation_l331_33109

def selling_price : ℝ := 270
def profit_percentage : ℝ := 0.20

theorem cost_price_calculation :
  ∃ (cost_price : ℝ), 
    cost_price * (1 + profit_percentage) = selling_price ∧ 
    cost_price = 225 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l331_33109


namespace NUMINAMATH_CALUDE_laura_cycling_distance_l331_33180

def base_7_to_10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem laura_cycling_distance : base_7_to_10 3 5 1 6 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_laura_cycling_distance_l331_33180


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l331_33150

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (2 * x + 15) = 12) ∧ (x = 64.5) := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l331_33150


namespace NUMINAMATH_CALUDE_other_number_proof_l331_33154

theorem other_number_proof (a b : ℕ+) : 
  Nat.gcd a b = 12 → 
  Nat.lcm a b = 396 → 
  a = 36 → 
  b = 132 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l331_33154


namespace NUMINAMATH_CALUDE_employment_agency_payroll_l331_33165

/-- Calculates the total payroll for an employment agency --/
theorem employment_agency_payroll
  (total_hired : ℕ)
  (num_laborers : ℕ)
  (operator_pay : ℕ)
  (laborer_pay : ℕ)
  (h_total : total_hired = 35)
  (h_laborers : num_laborers = 19)
  (h_operator_pay : operator_pay = 140)
  (h_laborer_pay : laborer_pay = 90) :
  let num_operators := total_hired - num_laborers
  let operator_total := num_operators * operator_pay
  let laborer_total := num_laborers * laborer_pay
  operator_total + laborer_total = 3950 := by
  sorry

#check employment_agency_payroll

end NUMINAMATH_CALUDE_employment_agency_payroll_l331_33165


namespace NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l331_33196

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (jack_speed : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_distance : ℝ) : 
  jack_speed = 3 → 
  christina_speed = 3 → 
  lindy_speed = 10 → 
  lindy_distance = 400 → 
  ∃ (initial_distance : ℝ), 
    initial_distance = 240 ∧ 
    (initial_distance / 2) / jack_speed = lindy_distance / lindy_speed :=
sorry

end NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l331_33196


namespace NUMINAMATH_CALUDE_inequality_solution_l331_33103

theorem inequality_solution (x : ℝ) : (x + 3) * (x - 1) < 0 ↔ -3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l331_33103


namespace NUMINAMATH_CALUDE_intersection_point_condition_l331_33181

theorem intersection_point_condition (α β : ℝ) : 
  (∃ x y : ℝ, 
    (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧
    (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧
    y = -x) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_condition_l331_33181


namespace NUMINAMATH_CALUDE_pet_food_ratio_l331_33128

/-- Represents the amounts of pet food in kilograms -/
structure PetFood where
  dog : ℕ
  cat : ℕ
  bird : ℕ

/-- The total amount of pet food -/
def total_food (pf : PetFood) : ℕ := pf.dog + pf.cat + pf.bird

/-- The ratio of pet food types -/
def food_ratio (pf : PetFood) : (ℕ × ℕ × ℕ) :=
  let gcd := Nat.gcd pf.dog (Nat.gcd pf.cat pf.bird)
  (pf.dog / gcd, pf.cat / gcd, pf.bird / gcd)

theorem pet_food_ratio : 
  let bought := PetFood.mk 15 10 5
  let final := PetFood.mk 40 15 5
  let initial := PetFood.mk (final.dog - bought.dog) (final.cat - bought.cat) (final.bird - bought.bird)
  total_food final = 60 →
  food_ratio final = (8, 3, 1) := by
  sorry

end NUMINAMATH_CALUDE_pet_food_ratio_l331_33128


namespace NUMINAMATH_CALUDE_smallest_value_3a_plus_1_l331_33163

theorem smallest_value_3a_plus_1 (a : ℂ) (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ (z : ℂ), 3 * z + 1 = -1/8 ∧ ∀ (w : ℂ), 8 * w^2 + 6 * w + 2 = 0 → Complex.re (3 * w + 1) ≥ -1/8 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_3a_plus_1_l331_33163


namespace NUMINAMATH_CALUDE_min_ab_perpendicular_lines_l331_33116

/-- Given two perpendicular lines and b > 0, the minimum value of ab is 2 -/
theorem min_ab_perpendicular_lines (b : ℝ) (a : ℝ) (h : b > 0) :
  (∃ x y, (b^2 + 1) * x + a * y + 2 = 0) ∧ 
  (∃ x y, x - b^2 * y - 1 = 0) ∧
  ((b^2 + 1) * (1 / b^2) = -1) →
  (∀ c, ab ≥ c → c ≤ 2) ∧ (∃ d, ab = d ∧ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_ab_perpendicular_lines_l331_33116


namespace NUMINAMATH_CALUDE_product_of_eight_consecutive_odd_numbers_divisible_by_ten_l331_33104

theorem product_of_eight_consecutive_odd_numbers_divisible_by_ten (n : ℕ) (h : Odd n) :
  ∃ k : ℕ, (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) * (n + 14)) = 10 * k :=
by
  sorry

#check product_of_eight_consecutive_odd_numbers_divisible_by_ten

end NUMINAMATH_CALUDE_product_of_eight_consecutive_odd_numbers_divisible_by_ten_l331_33104


namespace NUMINAMATH_CALUDE_inequality_solution_l331_33171

theorem inequality_solution (x : ℝ) : (x - 1) / (x - 3) ≥ 3 ↔ x ∈ Set.Ioo 3 4 ∪ {4} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l331_33171


namespace NUMINAMATH_CALUDE_largest_integral_x_l331_33139

theorem largest_integral_x : ∃ x : ℤ, x = 4 ∧ 
  (∀ y : ℤ, (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9 → y ≤ x) ∧
  (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 7/9 := by
  sorry

end NUMINAMATH_CALUDE_largest_integral_x_l331_33139


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l331_33146

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -5) : x^3 + 1/x^3 = -110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l331_33146


namespace NUMINAMATH_CALUDE_nonnegative_integer_solution_l331_33174

theorem nonnegative_integer_solution (x y z : ℕ) :
  (16 / 3 : ℝ)^x * (27 / 25 : ℝ)^y * (5 / 4 : ℝ)^z = 256 →
  x + y + z = 6 := by
sorry

end NUMINAMATH_CALUDE_nonnegative_integer_solution_l331_33174


namespace NUMINAMATH_CALUDE_jake_has_seven_peaches_l331_33129

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 12

/-- The number of peaches Jake has more than Jill -/
def jake_more_than_jill : ℕ := 72

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - jake_fewer_than_steven

theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_seven_peaches_l331_33129


namespace NUMINAMATH_CALUDE_unique_seating_arrangement_l331_33142

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  rows_with_8 : ℕ
  rows_with_7 : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid (s : SeatingArrangement) : Prop :=
  s.rows_with_8 * 8 + s.rows_with_7 * 7 = 55

/-- Theorem stating the unique valid seating arrangement --/
theorem unique_seating_arrangement :
  ∃! s : SeatingArrangement, is_valid s ∧ s.rows_with_8 = 6 := by sorry

end NUMINAMATH_CALUDE_unique_seating_arrangement_l331_33142


namespace NUMINAMATH_CALUDE_three_numbers_problem_l331_33199

theorem three_numbers_problem (x y z : ℝ) : 
  (x / y = y / z) ∧ 
  (x - (y + z) = 2) ∧ 
  (x + (y - z) / 2 = 9) →
  ((x = 8 ∧ y = 4 ∧ z = 2) ∨ (x = -6.4 ∧ y = 11.2 ∧ z = -19.6)) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l331_33199


namespace NUMINAMATH_CALUDE_perimeter_of_figure_c_l331_33131

/-- Given a large rectangle made up of 20 identical small rectangles,
    this theorem proves that if the perimeter of figure A (6x2 small rectangles)
    and figure B (4x6 small rectangles) are both 56 cm,
    then the perimeter of figure C (2x6 small rectangles) is 40 cm. -/
theorem perimeter_of_figure_c (x y : ℝ) 
  (h1 : 6 * x + 2 * y = 56)  -- Perimeter of figure A
  (h2 : 4 * x + 6 * y = 56)  -- Perimeter of figure B
  : 2 * x + 6 * y = 40 :=    -- Perimeter of figure C
by sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_c_l331_33131


namespace NUMINAMATH_CALUDE_oplus_three_one_l331_33187

-- Define the operation ⊕ for real numbers
def oplus (a b : ℝ) : ℝ := 3 * a + 4 * b

-- State the theorem
theorem oplus_three_one : oplus 3 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_oplus_three_one_l331_33187


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l331_33179

theorem quadratic_equation_sum (x r s : ℝ) : 
  (15 * x^2 + 30 * x - 450 = 0) →
  ((x + r)^2 = s) →
  (r + s = 32) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l331_33179


namespace NUMINAMATH_CALUDE_quadratic_function_points_range_l331_33145

theorem quadratic_function_points_range (m n y₁ y₂ : ℝ) : 
  y₁ = (m - 2)^2 + n → 
  y₂ = (m - 1)^2 + n → 
  y₁ < y₂ → 
  m > 3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_points_range_l331_33145


namespace NUMINAMATH_CALUDE_molecular_weight_h2o_is_18_l331_33149

/-- The molecular weight of dihydrogen monoxide in grams per mole -/
def molecular_weight_h2o : ℝ := 18

/-- The number of moles of dihydrogen monoxide -/
def moles_h2o : ℝ := 7

/-- The total weight of dihydrogen monoxide in grams -/
def total_weight_h2o : ℝ := 126

/-- Theorem: The molecular weight of dihydrogen monoxide is 18 grams per mole -/
theorem molecular_weight_h2o_is_18 :
  molecular_weight_h2o = total_weight_h2o / moles_h2o :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_h2o_is_18_l331_33149


namespace NUMINAMATH_CALUDE_university_packaging_volume_l331_33105

/-- The minimum volume needed to package the university's collection given the box dimensions, cost per box, and minimum amount spent. -/
theorem university_packaging_volume
  (box_length : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (cost_per_box : ℝ)
  (min_amount_spent : ℝ)
  (h_box_length : box_length = 20)
  (h_box_width : box_width = 20)
  (h_box_height : box_height = 12)
  (h_cost_per_box : cost_per_box = 0.5)
  (h_min_amount_spent : min_amount_spent = 200) :
  (min_amount_spent / cost_per_box) * (box_length * box_width * box_height) = 1920000 :=
by sorry

end NUMINAMATH_CALUDE_university_packaging_volume_l331_33105


namespace NUMINAMATH_CALUDE_two_million_six_hundred_thousand_scientific_notation_l331_33121

/-- Scientific notation representation -/
def scientific_notation (n : ℝ) (x : ℝ) (p : ℤ) : Prop :=
  1 ≤ x ∧ x < 10 ∧ n = x * (10 : ℝ) ^ p

/-- Theorem: 2,600,000 in scientific notation -/
theorem two_million_six_hundred_thousand_scientific_notation :
  ∃ (x : ℝ) (p : ℤ), scientific_notation 2600000 x p ∧ x = 2.6 ∧ p = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_million_six_hundred_thousand_scientific_notation_l331_33121


namespace NUMINAMATH_CALUDE_function_fixed_point_l331_33101

theorem function_fixed_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_fixed_point_l331_33101


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l331_33178

theorem absolute_value_inequality (x : ℝ) : 
  |((7 - x) / 4)| < 3 ↔ 2 < x ∧ x < 19 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l331_33178


namespace NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l331_33173

/-- Predicate to check if the equation ax^2 + by^2 = 1 represents an ellipse -/
def is_ellipse (a b : ℝ) : Prop := sorry

/-- Theorem stating that ab > 0 is a necessary but not sufficient condition for ax^2 + by^2 = 1 to represent an ellipse -/
theorem ab_positive_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse a b → a * b > 0) ∧
  ¬(∀ a b : ℝ, a * b > 0 → is_ellipse a b) := by sorry

end NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l331_33173


namespace NUMINAMATH_CALUDE_community_avg_age_l331_33102

-- Define the ratio of women to men
def women_to_men_ratio : ℚ := 7 / 5

-- Define the average age of women
def avg_age_women : ℝ := 30

-- Define the average age of men
def avg_age_men : ℝ := 35

-- Theorem statement
theorem community_avg_age :
  let total_population := women_to_men_ratio + 1
  let weighted_age_sum := women_to_men_ratio * avg_age_women + avg_age_men
  weighted_age_sum / total_population = 385 / 12 :=
by sorry

end NUMINAMATH_CALUDE_community_avg_age_l331_33102


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_difference_l331_33194

theorem sin_cos_fourth_power_difference (θ : ℝ) (h : Real.cos (2 * θ) = Real.sqrt 2 / 3) :
  Real.sin θ ^ 4 - Real.cos θ ^ 4 = -(Real.sqrt 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_difference_l331_33194


namespace NUMINAMATH_CALUDE_batsman_overall_average_l331_33189

def total_matches : ℕ := 30
def first_set_matches : ℕ := 20
def second_set_matches : ℕ := 10
def first_set_average : ℕ := 30
def second_set_average : ℕ := 15

theorem batsman_overall_average :
  let first_set_total := first_set_matches * first_set_average
  let second_set_total := second_set_matches * second_set_average
  let total_runs := first_set_total + second_set_total
  total_runs / total_matches = 25 := by sorry

end NUMINAMATH_CALUDE_batsman_overall_average_l331_33189


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l331_33164

theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 26 → age_difference = 28 → 
  ∃ (years : ℕ), (son_age + years + age_difference) = 2 * (son_age + years) ∧ years = 2 :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l331_33164


namespace NUMINAMATH_CALUDE_negation_equivalence_l331_33170

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ 
  (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l331_33170


namespace NUMINAMATH_CALUDE_simplify_fraction_l331_33198

theorem simplify_fraction : (5^6 + 5^3) / (5^5 - 5^2) = 315 / 62 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l331_33198


namespace NUMINAMATH_CALUDE_largest_even_not_sum_of_composite_odds_l331_33141

/-- A function that checks if a natural number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a natural number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- A function that checks if a natural number can be expressed as the sum of two composite odd positive integers -/
def isSumOfTwoCompositeOdds (n : ℕ) : Prop :=
  ∃ a b, isComposite a ∧ isComposite b ∧ isOdd a ∧ isOdd b ∧ n = a + b

/-- Theorem stating that 38 is the largest even positive integer that cannot be expressed as the sum of two composite odd positive integers -/
theorem largest_even_not_sum_of_composite_odds :
  (∀ n : ℕ, n > 38 → isSumOfTwoCompositeOdds n) ∧
  ¬isSumOfTwoCompositeOdds 38 ∧
  (∀ n : ℕ, n < 38 → n % 2 = 0 → isSumOfTwoCompositeOdds n ∨ n < 38) :=
sorry

end NUMINAMATH_CALUDE_largest_even_not_sum_of_composite_odds_l331_33141


namespace NUMINAMATH_CALUDE_sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2_l331_33107

theorem sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2 :
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2_l331_33107


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l331_33124

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (a + Complex.I) / (1 + Complex.I) = Complex.I * b) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l331_33124


namespace NUMINAMATH_CALUDE_sally_age_proof_l331_33177

theorem sally_age_proof (sally_age_five_years_ago : ℕ) : 
  sally_age_five_years_ago = 7 → 
  (sally_age_five_years_ago + 5 + 2 : ℕ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sally_age_proof_l331_33177


namespace NUMINAMATH_CALUDE_fraction_problem_l331_33134

theorem fraction_problem : 
  let number : ℝ := 14.500000000000002
  let result : ℝ := 126.15
  let fraction : ℝ := result / (number ^ 2)
  fraction = 0.6 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l331_33134


namespace NUMINAMATH_CALUDE_arc_length_from_central_angle_l331_33136

theorem arc_length_from_central_angle (D : Real) (EF : Real) (DEF : Real) : 
  D = 80 → DEF = 45 → EF = 10 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_from_central_angle_l331_33136


namespace NUMINAMATH_CALUDE_min_value_theorem_l331_33126

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  x + 81 / x ≥ 18 ∧ (x + 81 / x = 18 ↔ x = 9) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l331_33126


namespace NUMINAMATH_CALUDE_min_value_of_c_l331_33182

theorem min_value_of_c (a b c d e : ℕ) : 
  a > 0 → 
  b = a + 1 → 
  c = b + 1 → 
  d = c + 1 → 
  e = d + 1 → 
  ∃ n : ℕ, a + b + c + d + e = n^3 → 
  ∃ m : ℕ, b + c + d = m^2 → 
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' > 0 ∧ 
    b' = a' + 1 ∧ 
    d' = c' + 1 ∧ 
    e' = d' + 1 ∧ 
    (∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) ∧ 
    (∃ m' : ℕ, b' + c' + d' = m'^2)) → 
  c' ≥ c → 
  c = 675 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_c_l331_33182


namespace NUMINAMATH_CALUDE_smallest_multiple_with_100_divisors_l331_33110

/-- The number of positive integral divisors of n -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_100_divisors :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n → ¬(is_multiple m 100 ∧ divisor_count m = 100)) ∧
    is_multiple n 100 ∧
    divisor_count n = 100 ∧
    n / 100 = 324 / 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_100_divisors_l331_33110


namespace NUMINAMATH_CALUDE_amusement_park_visitors_l331_33108

/-- Amusement park visitor count problem -/
theorem amusement_park_visitors :
  let morning_visitors : ℕ := 473
  let noon_departures : ℕ := 179
  let afternoon_visitors : ℕ := 268
  let total_visitors : ℕ := morning_visitors + afternoon_visitors
  let current_visitors : ℕ := morning_visitors - noon_departures + afternoon_visitors
  (total_visitors = 741) ∧ (current_visitors = 562) := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_visitors_l331_33108


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l331_33135

/-- The probability that a randomly selected point from a square with side length 6
    is within a circle of radius 2 centered at the origin is π/9 -/
theorem probability_point_in_circle (s : ℝ) (r : ℝ) : 
  s = 6 → r = 2 → (π * r^2) / (s^2) = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l331_33135


namespace NUMINAMATH_CALUDE_megan_folders_l331_33184

/-- Given the initial number of files, number of deleted files, and files per folder,
    calculate the number of folders needed to store all remaining files. -/
def folders_needed (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  let remaining_files := initial_files - deleted_files
  (remaining_files + files_per_folder - 1) / files_per_folder

/-- Prove that given 237 initial files, 53 deleted files, and 12 files per folder,
    the number of folders needed is 16. -/
theorem megan_folders : folders_needed 237 53 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_megan_folders_l331_33184


namespace NUMINAMATH_CALUDE_horner_rule_v2_l331_33114

def horner_polynomial (x : ℚ) : ℚ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

def horner_v2 (x : ℚ) : ℚ :=
  let v1 := 2*x^3 - 3*x^2 + x
  v1 * x + 2

theorem horner_rule_v2 :
  horner_v2 (-1) = -4 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v2_l331_33114


namespace NUMINAMATH_CALUDE_cannot_transform_to_target_l331_33188

/-- Represents a parabola equation in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines a simple transformation of a parabola --/
inductive SimpleTransformation
  | right : SimpleTransformation  -- Move 2 units right
  | up : SimpleTransformation     -- Move 1 unit up

/-- Applies a simple transformation to a parabola --/
def applyTransformation (p : Parabola) (t : SimpleTransformation) : Parabola :=
  match t with
  | SimpleTransformation.right => { a := p.a, b := p.b - 2 * p.a, c := p.c + p.a }
  | SimpleTransformation.up => { a := p.a, b := p.b, c := p.c + 1 }

/-- Applies a sequence of simple transformations to a parabola --/
def applyTransformations (p : Parabola) (ts : List SimpleTransformation) : Parabola :=
  ts.foldl applyTransformation p

theorem cannot_transform_to_target : 
  ∀ (ts : List SimpleTransformation),
    ts.length = 2 → 
    applyTransformations { a := 1, b := 6, c := 5 } ts ≠ { a := 1, b := 0, c := 1 } :=
sorry

end NUMINAMATH_CALUDE_cannot_transform_to_target_l331_33188


namespace NUMINAMATH_CALUDE_triangle_theorem_l331_33155

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem triangle_theorem (A B C D E : ℝ × ℝ) :
  Triangle A B C →
  D ∈ Circle A (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →
  E ∈ Circle A (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →
  D.1 = C.1 ∧ D.2 = C.2 →
  E.1 = C.1 ∧ E.2 = C.2 →
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 20 →
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 16 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 936 := by
sorry


end NUMINAMATH_CALUDE_triangle_theorem_l331_33155


namespace NUMINAMATH_CALUDE_equation_solutions_l331_33130

theorem equation_solutions (x : ℝ) :
  x ≠ 2 → x ≠ 4 →
  ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) = 
   (x - 2) * (x - 4) * (x - 2)) ↔ 
  (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l331_33130
