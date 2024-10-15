import Mathlib

namespace NUMINAMATH_CALUDE_remainder_divisibility_l1030_103049

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 20) → (∃ m : ℤ, N = 13 * m + 7) := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1030_103049


namespace NUMINAMATH_CALUDE_intersection_distance_product_l1030_103047

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x

def C₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 2 * Real.sqrt 3 = 0

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem intersection_distance_product :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    A ≠ B ∧
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) *
     Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 32/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l1030_103047


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1030_103051

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 2 * x + 8) + 
  (x^4 - 2 * x^3 + 3 * x^2 + 4 * x - 16) = 
  2 * x^5 - 2 * x^4 - x^3 + 8 * x^2 + 2 * x - 8 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1030_103051


namespace NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_l1030_103006

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 2000

/-- Represents the weight of one ton in pounds -/
def pounds_per_ton : ℕ := 2500

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (num_packets * packet_weight) / pounds_per_ton

theorem gunny_bag_capacity_is_13 : gunny_bag_capacity = 13 := by
  sorry

end NUMINAMATH_CALUDE_gunny_bag_capacity_is_13_l1030_103006


namespace NUMINAMATH_CALUDE_total_production_proof_l1030_103001

def day_shift_production : ℕ := 4400
def day_shift_multiplier : ℕ := 4

theorem total_production_proof :
  let second_shift_production := day_shift_production / day_shift_multiplier
  day_shift_production + second_shift_production = 5500 := by
  sorry

end NUMINAMATH_CALUDE_total_production_proof_l1030_103001


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_twelve_l1030_103030

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) : 
  12 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_twelve_l1030_103030


namespace NUMINAMATH_CALUDE_equation_one_solution_l1030_103062

theorem equation_one_solution (k : ℝ) : 
  (∃! x : ℝ, (3*x + 6)*(x - 4) = -40 + k*x) ↔ 
  (k = -6 + 8*Real.sqrt 3 ∨ k = -6 - 8*Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_equation_one_solution_l1030_103062


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_product_l1030_103027

theorem odd_sum_of_squares_implies_odd_product (n m : ℤ) 
  (h : Odd (n^2 + m^2)) : Odd (n * m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_product_l1030_103027


namespace NUMINAMATH_CALUDE_gum_ratio_proof_l1030_103003

def gum_ratio (total_gum : ℕ) (shane_chewed : ℕ) (shane_left : ℕ) : ℚ :=
  let shane_total := shane_chewed + shane_left
  let rick_total := shane_total * 2
  rick_total / total_gum

theorem gum_ratio_proof :
  gum_ratio 100 11 14 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_gum_ratio_proof_l1030_103003


namespace NUMINAMATH_CALUDE_women_percentage_of_men_l1030_103056

theorem women_percentage_of_men (W M : ℝ) (h : M = 2 * W) : W / M * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_women_percentage_of_men_l1030_103056


namespace NUMINAMATH_CALUDE_triangle_side_length_l1030_103016

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  B = 2 * A ∧
  a = 1 ∧ 
  b = Real.sqrt 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C ∧
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1030_103016


namespace NUMINAMATH_CALUDE_swap_numbers_l1030_103015

theorem swap_numbers (a b : ℕ) : 
  let c := b
  let b' := a
  let a' := c
  (a' = b ∧ b' = a) :=
by
  sorry

end NUMINAMATH_CALUDE_swap_numbers_l1030_103015


namespace NUMINAMATH_CALUDE_modulo_thirteen_seven_l1030_103046

theorem modulo_thirteen_seven (n : ℕ) : 
  13^7 ≡ n [ZMOD 7] → 0 ≤ n → n < 7 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_modulo_thirteen_seven_l1030_103046


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1030_103072

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 0 → a > 1) ↔ (∀ a : ℝ, a ≤ 1 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1030_103072


namespace NUMINAMATH_CALUDE_factorial_program_components_l1030_103038

/-- A structure representing a simple programming language --/
structure SimpleProgram where
  input : String
  loop_start : String
  loop_end : String

/-- Definition of a program that calculates factorial --/
def factorial_program (p : SimpleProgram) : Prop :=
  p.input = "INPUT" ∧ 
  p.loop_start = "WHILE" ∧ 
  p.loop_end = "WEND"

/-- Theorem stating that a program calculating factorial requires specific components --/
theorem factorial_program_components :
  ∃ (p : SimpleProgram), factorial_program p :=
sorry

end NUMINAMATH_CALUDE_factorial_program_components_l1030_103038


namespace NUMINAMATH_CALUDE_area_of_region_l1030_103008

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 32 ∧ 
   A = Real.pi * (Real.sqrt ((x + 4)^2 + (y + 5)^2))^2 ∧
   x^2 + y^2 + 8*x + 10*y = -9) := by
sorry

end NUMINAMATH_CALUDE_area_of_region_l1030_103008


namespace NUMINAMATH_CALUDE_basketball_tournament_wins_losses_l1030_103071

theorem basketball_tournament_wins_losses 
  (total_games : ℕ) 
  (points_per_win : ℕ) 
  (points_per_loss : ℕ) 
  (total_points : ℕ) 
  (h1 : total_games = 15) 
  (h2 : points_per_win = 3) 
  (h3 : points_per_loss = 1) 
  (h4 : total_points = 41) : 
  ∃ (wins losses : ℕ), 
    wins + losses = total_games ∧ 
    wins * points_per_win + losses * points_per_loss = total_points ∧ 
    wins = 13 ∧ 
    losses = 2 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tournament_wins_losses_l1030_103071


namespace NUMINAMATH_CALUDE_puppy_adoption_cost_is_96_l1030_103084

/-- Calculates the total cost of adopting a puppy and buying necessary supplies with a discount --/
def puppy_adoption_cost (adoption_fee : ℝ) (dog_food : ℝ) (treats_price : ℝ) (treats_quantity : ℕ)
  (toys : ℝ) (crate_bed_price : ℝ) (collar_leash : ℝ) (discount_rate : ℝ) : ℝ :=
  let supplies_cost := dog_food + treats_price * treats_quantity + toys + 2 * crate_bed_price + collar_leash
  let discounted_supplies := supplies_cost * (1 - discount_rate)
  adoption_fee + discounted_supplies

/-- Theorem stating that the total cost of adopting a puppy and buying supplies is $96.00 --/
theorem puppy_adoption_cost_is_96 :
  puppy_adoption_cost 20 20 2.5 2 15 20 15 0.2 = 96 := by
  sorry


end NUMINAMATH_CALUDE_puppy_adoption_cost_is_96_l1030_103084


namespace NUMINAMATH_CALUDE_intersection_point_D_l1030_103037

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 1

/-- The normal line equation at point (2, 4) -/
def normal_line (x y : ℝ) : Prop := y = -1/4 * x + 9/2

theorem intersection_point_D :
  let C : ℝ × ℝ := (2, 4)
  let D : ℝ × ℝ := (-2, 5)
  parabola C.1 C.2 →
  parabola D.1 D.2 ∧
  normal_line D.1 D.2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_D_l1030_103037


namespace NUMINAMATH_CALUDE_power_sum_equality_l1030_103061

theorem power_sum_equality : 2^567 + 8^5 / 8^3 = 2^567 + 64 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1030_103061


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1030_103020

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1030_103020


namespace NUMINAMATH_CALUDE_rectangle_area_implies_y_l1030_103063

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    if the area of the rectangle is 45 square units and y > 0, then y = 9. -/
theorem rectangle_area_implies_y (y : ℝ) : y > 0 → 5 * y = 45 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_implies_y_l1030_103063


namespace NUMINAMATH_CALUDE_exactly_three_valid_sets_l1030_103017

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)

/-- The sum of a ConsecutiveSet -/
def sum_consecutive_set (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Predicate for a valid set according to our conditions -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive_set s = 30

/-- The theorem to prove -/
theorem exactly_three_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), 
    sets.card = 3 ∧ 
    (∀ s ∈ sets, is_valid_set s) ∧
    (∀ s, is_valid_set s → s ∈ sets) :=
  sorry

end NUMINAMATH_CALUDE_exactly_three_valid_sets_l1030_103017


namespace NUMINAMATH_CALUDE_penny_nickel_dime_heads_probability_l1030_103041

def coin_flip_probability : ℚ :=
  let total_outcomes : ℕ := 2^5
  let successful_outcomes : ℕ := 2^2
  (successful_outcomes : ℚ) / total_outcomes

theorem penny_nickel_dime_heads_probability :
  coin_flip_probability = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_penny_nickel_dime_heads_probability_l1030_103041


namespace NUMINAMATH_CALUDE_secret_spread_days_l1030_103035

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The number of days required for 3280 students to know the secret -/
theorem secret_spread_days : ∃ n : ℕ, secret_spread n = 3280 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_secret_spread_days_l1030_103035


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l1030_103081

theorem abs_inequality_solution_set (x : ℝ) : 
  (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l1030_103081


namespace NUMINAMATH_CALUDE_first_day_over_500_l1030_103048

def paperclips (day : ℕ) : ℕ :=
  match day with
  | 0 => 5  -- Monday
  | 1 => 10 -- Tuesday
  | n + 2 => 3 * paperclips (n + 1)

theorem first_day_over_500 :
  (∀ d < 6, paperclips d ≤ 500) ∧ (paperclips 6 > 500) := by
  sorry

end NUMINAMATH_CALUDE_first_day_over_500_l1030_103048


namespace NUMINAMATH_CALUDE_not_in_set_A_l1030_103009

-- Define the set A
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 5}

-- Theorem statement
theorem not_in_set_A :
  (1, -5) ∉ A ∧ (2, 1) ∈ A ∧ (3, 4) ∈ A ∧ (4, 7) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_not_in_set_A_l1030_103009


namespace NUMINAMATH_CALUDE_kite_area_l1030_103012

/-- The area of a kite composed of two identical triangles -/
theorem kite_area (base height : ℝ) (h1 : base = 14) (h2 : height = 6) :
  2 * (1/2 * base * height) = 84 := by
  sorry

end NUMINAMATH_CALUDE_kite_area_l1030_103012


namespace NUMINAMATH_CALUDE_angle_sum_360_l1030_103058

theorem angle_sum_360 (k : ℝ) : k + 90 = 360 → k = 270 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_360_l1030_103058


namespace NUMINAMATH_CALUDE_rectangle_midpoint_distances_theorem_l1030_103053

def rectangle_midpoint_distances : ℝ := by
  -- Define the vertices of the rectangle
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (3, 4)
  let D : ℝ × ℝ := (0, 4)

  -- Define the midpoints of each side
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let O : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let P : ℝ × ℝ := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

  -- Calculate distances from A to each midpoint
  let d_AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let d_AN := Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2)
  let d_AO := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  let d_AP := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

  -- Sum of distances
  let total_distance := d_AM + d_AN + d_AO + d_AP

  -- Prove that the total distance equals the expected value
  sorry

theorem rectangle_midpoint_distances_theorem :
  rectangle_midpoint_distances = (3 * Real.sqrt 2) / 2 + Real.sqrt 13 + (Real.sqrt 73) / 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_midpoint_distances_theorem_l1030_103053


namespace NUMINAMATH_CALUDE_minimum_m_value_l1030_103039

theorem minimum_m_value (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : ∀ a b c, a > b ∧ b > c → (1 / (a - b) + m / (b - c) ≥ 9 / (a - c))) : 
  m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_minimum_m_value_l1030_103039


namespace NUMINAMATH_CALUDE_complex_division_simplification_l1030_103033

theorem complex_division_simplification (z : ℂ) (h : z = 1 - 2 * I) :
  5 * I / z = -2 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l1030_103033


namespace NUMINAMATH_CALUDE_linear_function_range_l1030_103032

/-- A linear function defined on a closed interval -/
def LinearFunction (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- The domain of the function -/
def Domain : Set ℝ := { x : ℝ | 1/4 ≤ x ∧ x ≤ 3/4 }

theorem linear_function_range (a b : ℝ) (h : a > 0) :
  Set.range (fun x => LinearFunction a b x) = Set.Icc (a/4 + b) (3*a/4 + b) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_range_l1030_103032


namespace NUMINAMATH_CALUDE_stamps_given_l1030_103023

theorem stamps_given (x : ℕ) (y : ℕ) : 
  (7 * x : ℕ) / (4 * x) = 7 / 4 →  -- Initial ratio
  ((7 * x - y) : ℕ) / (4 * x + y) = 6 / 5 →  -- Final ratio
  (7 * x - y) = (4 * x + y) + 8 →  -- Final difference
  y = 8 := by sorry

end NUMINAMATH_CALUDE_stamps_given_l1030_103023


namespace NUMINAMATH_CALUDE_snow_clearing_volume_l1030_103031

/-- The volume of snow on a rectangular pathway -/
def snow_volume (length width depth : ℚ) : ℚ :=
  length * width * depth

/-- Proof that the volume of snow on the given pathway is 67.5 cubic feet -/
theorem snow_clearing_volume :
  let length : ℚ := 30
  let width : ℚ := 3
  let depth : ℚ := 3/4
  snow_volume length width depth = 67.5 := by
sorry

end NUMINAMATH_CALUDE_snow_clearing_volume_l1030_103031


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1030_103004

/-- 
For an infinite geometric series with common ratio r and sum S, 
the first term a is given by the formula: a = S * (1 - r)
-/
def first_term_infinite_geometric_series (r : ℚ) (S : ℚ) : ℚ := S * (1 - r)

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 18) : 
  first_term_infinite_geometric_series r S = 24 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1030_103004


namespace NUMINAMATH_CALUDE_trays_for_school_staff_l1030_103057

def small_oatmeal_cookies : ℕ := 276
def large_oatmeal_cookies : ℕ := 92
def large_choc_chip_cookies : ℕ := 150
def small_cookies_per_tray : ℕ := 12
def large_cookies_per_tray : ℕ := 6

theorem trays_for_school_staff : 
  (large_choc_chip_cookies + large_cookies_per_tray - 1) / large_cookies_per_tray = 25 := by
  sorry

end NUMINAMATH_CALUDE_trays_for_school_staff_l1030_103057


namespace NUMINAMATH_CALUDE_common_solution_y_value_l1030_103013

theorem common_solution_y_value : ∃! y : ℝ, ∃ x : ℝ, 
  (x^2 + y^2 - 4 = 0) ∧ (x^2 - 4*y + 8 = 0) :=
by
  -- Proof goes here
  sorry

#check common_solution_y_value

end NUMINAMATH_CALUDE_common_solution_y_value_l1030_103013


namespace NUMINAMATH_CALUDE_teacher_student_meeting_l1030_103040

/-- Represents the teacher-student meeting scenario -/
structure MeetingScenario where
  total_participants : ℕ
  first_teacher_students : ℕ
  teachers : ℕ
  students : ℕ

/-- Checks if the given scenario satisfies the meeting conditions -/
def is_valid_scenario (m : MeetingScenario) : Prop :=
  m.total_participants = m.teachers + m.students ∧
  m.first_teacher_students = m.students - m.teachers + 1 ∧
  m.teachers > 0 ∧
  m.students > 0

/-- The theorem stating the correct number of teachers and students -/
theorem teacher_student_meeting :
  ∃ (m : MeetingScenario), is_valid_scenario m ∧ m.teachers = 8 ∧ m.students = 23 :=
sorry

end NUMINAMATH_CALUDE_teacher_student_meeting_l1030_103040


namespace NUMINAMATH_CALUDE_radius_is_seven_l1030_103070

/-- Represents a circle with a point P outside it and a secant PQR -/
structure CircleWithSecant where
  /-- Distance from P to the center of the circle -/
  s : ℝ
  /-- Length of external segment PQ -/
  pq : ℝ
  /-- Length of chord QR -/
  qr : ℝ

/-- The radius of the circle given the secant configuration -/
def radius (c : CircleWithSecant) : ℝ :=
  sorry

/-- Theorem stating that the radius is 7 given the specific measurements -/
theorem radius_is_seven (c : CircleWithSecant) 
  (h1 : c.s = 17) 
  (h2 : c.pq = 12) 
  (h3 : c.qr = 8) : 
  radius c = 7 := by
  sorry

end NUMINAMATH_CALUDE_radius_is_seven_l1030_103070


namespace NUMINAMATH_CALUDE_fifty_third_number_is_71_l1030_103099

def sequenceValue (n : ℕ) : ℕ := 
  let fullSets := (n - 1) / 3
  let remainder := (n - 1) % 3
  1 + 4 * fullSets + remainder + (if remainder = 2 then 1 else 0)

theorem fifty_third_number_is_71 : sequenceValue 53 = 71 := by
  sorry

end NUMINAMATH_CALUDE_fifty_third_number_is_71_l1030_103099


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l1030_103034

theorem min_value_of_quadratic (x : ℝ) :
  let z := 5 * x^2 - 20 * x + 45
  ∀ y : ℝ, z ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l1030_103034


namespace NUMINAMATH_CALUDE_spade_heart_diamond_probability_l1030_103029

/-- Represents a standard deck of cards -/
structure Deck :=
  (total : Nat)
  (spades : Nat)
  (hearts : Nat)
  (diamonds : Nat)

/-- Calculates the probability of drawing a specific card from the deck -/
def drawProbability (deck : Deck) (targetCards : Nat) : Rat :=
  targetCards / deck.total

/-- Represents the state of the deck after drawing a card -/
def drawCard (deck : Deck) : Deck :=
  { deck with total := deck.total - 1 }

/-- Standard 52-card deck -/
def standardDeck : Deck :=
  { total := 52, spades := 13, hearts := 13, diamonds := 13 }

theorem spade_heart_diamond_probability :
  let firstDraw := drawProbability standardDeck standardDeck.spades
  let secondDraw := drawProbability (drawCard standardDeck) standardDeck.hearts
  let thirdDraw := drawProbability (drawCard (drawCard standardDeck)) standardDeck.diamonds
  firstDraw * secondDraw * thirdDraw = 2197 / 132600 := by sorry

end NUMINAMATH_CALUDE_spade_heart_diamond_probability_l1030_103029


namespace NUMINAMATH_CALUDE_negation_of_implication_l1030_103098

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → 2*a > 2*b - 1) ↔ (a ≤ b → 2*a ≤ 2*b - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1030_103098


namespace NUMINAMATH_CALUDE_base_conversion_l1030_103011

/-- Given that the decimal number 26 converted to base r is 32, prove that r = 8 -/
theorem base_conversion (r : ℕ) (h : r > 1) : 
  (26 : ℕ).digits r = [3, 2] → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l1030_103011


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_cube_l1030_103080

theorem cube_sum_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^3 + 1/r^3 = 2 * Real.sqrt 5 ∨ r^3 + 1/r^3 = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_cube_l1030_103080


namespace NUMINAMATH_CALUDE_total_subjects_l1030_103002

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 77)
  (h2 : average_five = 74)
  (h3 : last_subject = 92) :
  ∃ n : ℕ, n = 6 ∧ 
    n * average_all = (n - 1) * average_five + last_subject :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l1030_103002


namespace NUMINAMATH_CALUDE_sports_league_games_l1030_103036

/-- Calculates the total number of games in a sports league season. -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  (total_teams * (intra_division_games * (teams_per_division - 1) + 
  inter_division_games * teams_per_division)) / 2

/-- Theorem stating the total number of games in the given sports league setup -/
theorem sports_league_games : 
  total_games 16 8 3 1 = 232 := by
  sorry

end NUMINAMATH_CALUDE_sports_league_games_l1030_103036


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1030_103091

/-- A line in 2D space represented by the equation f(x,y) = 0 -/
structure Line2D where
  f : ℝ → ℝ → ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The equation f(x,y) - f(x₁,y₁) - f(x₂,y₂) = 0 represents a line parallel to 
    the original line and passing through P₂ -/
theorem parallel_line_through_point (l : Line2D) (P₁ P₂ : Point2D) 
  (h₁ : l.f P₁.x P₁.y = 0)  -- P₁ is on the line l
  (h₂ : l.f P₂.x P₂.y ≠ 0)  -- P₂ is not on the line l
  : ∃ (m : Line2D), 
    (∀ x y, m.f x y = l.f x y - l.f P₁.x P₁.y - l.f P₂.x P₂.y) ∧ 
    (m.f P₂.x P₂.y = 0) ∧
    (∃ k : ℝ, ∀ x y, m.f x y = k * l.f x y) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1030_103091


namespace NUMINAMATH_CALUDE_parallel_lines_angle_condition_l1030_103088

-- Define the concept of lines and planes
variable (Line Plane : Type)

-- Define the concept of parallel lines
variable (parallel : Line → Line → Prop)

-- Define the concept of a line forming an angle with a plane
variable (angle_with_plane : Line → Plane → ℝ)

-- State the theorem
theorem parallel_lines_angle_condition 
  (a b : Line) (α : Plane) :
  (parallel a b → angle_with_plane a α = angle_with_plane b α) ∧
  ¬(angle_with_plane a α = angle_with_plane b α → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_condition_l1030_103088


namespace NUMINAMATH_CALUDE_shop_ratio_l1030_103074

/-- Given a shop with pencils, pens, and exercise books in the ratio 14 : 4 : 3,
    and 140 pencils, the ratio of exercise books to pens is 3 : 4. -/
theorem shop_ratio (pencils pens books : ℕ) : 
  pencils = 140 →
  pencils / 14 = pens / 4 →
  pencils / 14 = books / 3 →
  books / pens = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_shop_ratio_l1030_103074


namespace NUMINAMATH_CALUDE_number_equation_l1030_103025

theorem number_equation (x : ℝ) : 100 - x = x + 40 ↔ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1030_103025


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l1030_103019

theorem blue_lipstick_count (total_students : ℕ) 
  (h_total : total_students = 300)
  (h_lipstick : ∃ lipstick_wearers : ℕ, lipstick_wearers = total_students / 2)
  (h_red : ∃ red_wearers : ℕ, red_wearers = lipstick_wearers / 4)
  (h_pink : ∃ pink_wearers : ℕ, pink_wearers = lipstick_wearers / 3)
  (h_purple : ∃ purple_wearers : ℕ, purple_wearers = lipstick_wearers / 6)
  (h_blue : ∃ blue_wearers : ℕ, blue_wearers = lipstick_wearers - (red_wearers + pink_wearers + purple_wearers)) :
  blue_wearers = 37 := by
sorry

end NUMINAMATH_CALUDE_blue_lipstick_count_l1030_103019


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_is_six_l1030_103042

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  total_revenue : ℕ

/-- The specific rental business instance --/
def our_business : RentalBusiness := {
  canoe_price := 9
  kayak_price := 12
  canoe_kayak_ratio := 4/3
  total_revenue := 432
}

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (rb : RentalBusiness) : ℕ :=
  sorry

/-- Theorem stating that the difference between canoes and kayaks rented is 6 --/
theorem canoe_kayak_difference_is_six :
  canoe_kayak_difference our_business = 6 := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_is_six_l1030_103042


namespace NUMINAMATH_CALUDE_peter_walking_time_l1030_103043

/-- The time required to walk a given distance at a given pace -/
def timeToWalk (distance : ℝ) (pace : ℝ) : ℝ := distance * pace

theorem peter_walking_time :
  let totalDistance : ℝ := 2.5
  let walkingPace : ℝ := 20
  let distanceWalked : ℝ := 1
  let remainingDistance : ℝ := totalDistance - distanceWalked
  timeToWalk remainingDistance walkingPace = 30 := by
sorry

end NUMINAMATH_CALUDE_peter_walking_time_l1030_103043


namespace NUMINAMATH_CALUDE_prob_blue_or_green_l1030_103089

def cube_prob (blue_faces green_faces red_faces : ℕ) : ℚ :=
  (blue_faces + green_faces : ℚ) / (blue_faces + green_faces + red_faces)

theorem prob_blue_or_green : 
  cube_prob 3 1 2 = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_blue_or_green_l1030_103089


namespace NUMINAMATH_CALUDE_power_two_minus_one_div_seven_l1030_103044

theorem power_two_minus_one_div_seven (n : ℕ) :
  (7 ∣ (2^n - 1)) ↔ (3 ∣ n) := by
sorry

end NUMINAMATH_CALUDE_power_two_minus_one_div_seven_l1030_103044


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l1030_103018

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) :
  (Nat.choose n 2 : ℝ) * x^(n - 2) * a^2 = 84 ∧
  (Nat.choose n 3 : ℝ) * x^(n - 3) * a^3 = 280 ∧
  (Nat.choose n 4 : ℝ) * x^(n - 4) * a^4 = 560 →
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l1030_103018


namespace NUMINAMATH_CALUDE_patricks_class_size_l1030_103075

theorem patricks_class_size :
  ∃! b : ℕ,
    100 < b ∧ b < 200 ∧
    ∃ k : ℕ, b = 4 * k - 2 ∧
    ∃ l : ℕ, b = 6 * l - 3 ∧
    ∃ m : ℕ, b = 7 * m - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_patricks_class_size_l1030_103075


namespace NUMINAMATH_CALUDE_binomial_coefficient_prime_power_bound_l1030_103064

theorem binomial_coefficient_prime_power_bound 
  (p : Nat) (n k α : Nat) (h_prime : Prime p) 
  (h_divides : p ^ α ∣ Nat.choose n k) : 
  p ^ α ≤ n :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_prime_power_bound_l1030_103064


namespace NUMINAMATH_CALUDE_exact_one_solver_probability_l1030_103097

/-- The probability that exactly one person solves a problem, given the probabilities
    for two independent solvers. -/
theorem exact_one_solver_probability (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  (p₁ * (1 - p₂) + p₂ * (1 - p₁)) = 
  (p₁ + p₂ - 2 * p₁ * p₂) := by
  sorry

#check exact_one_solver_probability

end NUMINAMATH_CALUDE_exact_one_solver_probability_l1030_103097


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1030_103077

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x - 1 ≥ 0}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (univ \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1030_103077


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l1030_103085

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 : ℚ) / 3 = 27 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l1030_103085


namespace NUMINAMATH_CALUDE_max_sum_given_product_l1030_103028

theorem max_sum_given_product (a b : ℤ) (h : a * b = -72) : 
  (∀ (x y : ℤ), x * y = -72 → x + y ≤ a + b) → a + b = 71 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_product_l1030_103028


namespace NUMINAMATH_CALUDE_soup_offer_ratio_l1030_103073

/-- Represents the soup can offer --/
structure SoupOffer where
  total_cans : ℕ
  normal_price : ℚ
  total_paid : ℚ

/-- Calculates the buy to get ratio for a soup offer --/
def buyToGetRatio (offer : SoupOffer) : ℚ × ℚ :=
  let paid_cans := offer.total_paid / offer.normal_price
  let free_cans := offer.total_cans - paid_cans
  (paid_cans, free_cans)

/-- Theorem stating that the given offer results in a 1:1 ratio --/
theorem soup_offer_ratio (offer : SoupOffer) 
  (h1 : offer.total_cans = 30)
  (h2 : offer.normal_price = 0.6)
  (h3 : offer.total_paid = 9) :
  buyToGetRatio offer = (15, 15) := by
  sorry

#eval buyToGetRatio ⟨30, 0.6, 9⟩

end NUMINAMATH_CALUDE_soup_offer_ratio_l1030_103073


namespace NUMINAMATH_CALUDE_tower_comparison_l1030_103087

def tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (tower base n)

theorem tower_comparison (n : ℕ) : ∃ m : ℕ, ∀ k ≥ m,
  (tower 3 k > tower 2 (k + 1)) ∧ (tower 4 k > tower 3 k) := by
  sorry

#check tower_comparison

end NUMINAMATH_CALUDE_tower_comparison_l1030_103087


namespace NUMINAMATH_CALUDE_range_of_m_l1030_103067

theorem range_of_m : 
  (∀ x : ℝ, |2009 * x + 1| ≥ |m - 1| - 2) ↔ m ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1030_103067


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l1030_103014

theorem part_to_whole_ratio (N : ℚ) (P : ℚ) : 
  (1 / 4 : ℚ) * P = 25 →
  (40 / 100 : ℚ) * N = 300 →
  P / ((2 / 5 : ℚ) * N) = (1 / 3 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l1030_103014


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1030_103079

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

def monotonic_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem monotonic_decreasing_interval :
  ∀ x, x > 0 → (monotonic_decreasing f 0 1) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l1030_103079


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1030_103026

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem stating that if M(3, a-2) and N(b, a) are symmetric with respect to the origin, then a + b = -2 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin 3 (a - 2) b a → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1030_103026


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1030_103083

def M : Set Int := {-1, 3, 5}
def N : Set Int := {-1, 0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1030_103083


namespace NUMINAMATH_CALUDE_inequality_proof_l1030_103005

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + (1 / (x * y)) ≤ (1 / x) + (1 / y) + x * y := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1030_103005


namespace NUMINAMATH_CALUDE_common_solution_y_value_l1030_103045

theorem common_solution_y_value (x y : ℝ) : 
  x^2 + y^2 - 4 = 0 ∧ x^2 - y + 2 = 0 → y = 2 :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_value_l1030_103045


namespace NUMINAMATH_CALUDE_terminal_zeros_25_times_240_l1030_103069

/-- The number of terminal zeros in a positive integer -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The prime factorization of 25 -/
def primeFactor25 : ℕ → ℕ
| 5 => 2
| _ => 0

/-- The prime factorization of 240 -/
def primeFactor240 : ℕ → ℕ
| 2 => 4
| 3 => 1
| 5 => 1
| _ => 0

theorem terminal_zeros_25_times_240 : 
  terminalZeros (25 * 240) = 3 := by sorry

end NUMINAMATH_CALUDE_terminal_zeros_25_times_240_l1030_103069


namespace NUMINAMATH_CALUDE_intersection_M_N_l1030_103010

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2, 3, 4}

-- Define the complement of M with respect to U
def M_complement : Set Int := {-1, 1}

-- Define set N
def N : Set Int := {0, 1, 2, 3}

-- Define set M based on its complement
def M : Set Int := U \ M_complement

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1030_103010


namespace NUMINAMATH_CALUDE_prop_variations_l1030_103093

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 2 → x^2 - 5*x + 6 = 0

-- Define the converse
def converse (x : ℝ) : Prop := x^2 - 5*x + 6 = 0 → x = 2

-- Define the inverse
def inverse (x : ℝ) : Prop := x ≠ 2 → x^2 - 5*x + 6 ≠ 0

-- Define the contrapositive
def contrapositive (x : ℝ) : Prop := x^2 - 5*x + 6 ≠ 0 → x ≠ 2

-- Theorem stating the truth values of converse, inverse, and contrapositive
theorem prop_variations :
  (∃ x : ℝ, ¬(converse x)) ∧
  (∃ x : ℝ, ¬(inverse x)) ∧
  (∀ x : ℝ, contrapositive x) :=
sorry

end NUMINAMATH_CALUDE_prop_variations_l1030_103093


namespace NUMINAMATH_CALUDE_multiply_three_a_two_ab_l1030_103065

theorem multiply_three_a_two_ab (a b : ℝ) : 3 * a * (2 * a * b) = 6 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_a_two_ab_l1030_103065


namespace NUMINAMATH_CALUDE_wall_building_time_l1030_103094

theorem wall_building_time 
  (men_days_constant : ℕ → ℕ → ℕ) 
  (h1 : men_days_constant 10 6 = men_days_constant 15 4) 
  (h2 : ∀ m d, men_days_constant m d = m * d) :
  (10 : ℚ) * 6 / 15 = 4 :=
sorry

end NUMINAMATH_CALUDE_wall_building_time_l1030_103094


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1030_103076

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1030_103076


namespace NUMINAMATH_CALUDE_sin_value_given_sum_and_tan_condition_l1030_103066

theorem sin_value_given_sum_and_tan_condition (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) : 
  Real.sin θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_given_sum_and_tan_condition_l1030_103066


namespace NUMINAMATH_CALUDE_ice_cream_unsold_l1030_103007

theorem ice_cream_unsold (chocolate mango vanilla strawberry : ℕ)
  (h_chocolate : chocolate = 50)
  (h_mango : mango = 54)
  (h_vanilla : vanilla = 80)
  (h_strawberry : strawberry = 40)
  (sold_chocolate : ℚ)
  (sold_mango : ℚ)
  (sold_vanilla : ℚ)
  (sold_strawberry : ℚ)
  (h_sold_chocolate : sold_chocolate = 3 / 5)
  (h_sold_mango : sold_mango = 2 / 3)
  (h_sold_vanilla : sold_vanilla = 3 / 4)
  (h_sold_strawberry : sold_strawberry = 5 / 8) :
  chocolate - Int.floor (sold_chocolate * chocolate) +
  mango - Int.floor (sold_mango * mango) +
  vanilla - Int.floor (sold_vanilla * vanilla) +
  strawberry - Int.floor (sold_strawberry * strawberry) = 73 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_unsold_l1030_103007


namespace NUMINAMATH_CALUDE_flooring_problem_l1030_103000

theorem flooring_problem (room_length room_width box_area boxes_needed : ℕ) 
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : box_area = 10)
  (h4 : boxes_needed = 7) :
  room_length * room_width - boxes_needed * box_area = 250 :=
by sorry

end NUMINAMATH_CALUDE_flooring_problem_l1030_103000


namespace NUMINAMATH_CALUDE_amount_owed_l1030_103082

theorem amount_owed (rate_per_car : ℚ) (cars_washed : ℚ) (h1 : rate_per_car = 9/4) (h2 : cars_washed = 10/3) : 
  rate_per_car * cars_washed = 15/2 := by
sorry

end NUMINAMATH_CALUDE_amount_owed_l1030_103082


namespace NUMINAMATH_CALUDE_problem_statement_l1030_103024

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem problem_statement (x : ℝ) 
  (h : deriv f x = 2 * f x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin (2 * x)) = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1030_103024


namespace NUMINAMATH_CALUDE_set_of_values_for_a_l1030_103022

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + a = 0}

theorem set_of_values_for_a (a : ℝ) : 
  (∀ B : Set ℝ, B ⊆ A a → B = ∅ ∨ B = A a) → 
  (a > 1 ∨ a < -1) :=
sorry

end NUMINAMATH_CALUDE_set_of_values_for_a_l1030_103022


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l1030_103050

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l1030_103050


namespace NUMINAMATH_CALUDE_work_completion_time_l1030_103055

theorem work_completion_time (a b c : ℝ) : 
  (b = 12) →  -- B can do the work in 12 days
  (1/a + 1/b = 1/4) →  -- A and B working together finish the work in 4 days
  (a = 6) -- A can do the work alone in 6 days
:= by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1030_103055


namespace NUMINAMATH_CALUDE_inequality_proof_l1030_103092

theorem inequality_proof (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1030_103092


namespace NUMINAMATH_CALUDE_parabola_intersection_l1030_103054

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem parabola_intersection :
  (f (-1) = 0) →  -- The parabola intersects the x-axis at (-1, 0)
  (∃ x : ℝ, x ≠ -1 ∧ f x = 0 ∧ x = 3) :=  -- There exists another intersection point at (3, 0)
by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1030_103054


namespace NUMINAMATH_CALUDE_eight_digit_number_theorem_l1030_103086

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def last_digit (n : ℕ) : ℕ := n % 10

def move_last_to_first (n : ℕ) : ℕ :=
  (last_digit n) * 10^7 + n / 10

theorem eight_digit_number_theorem (B : ℕ) 
  (h1 : B > 7777777)
  (h2 : is_coprime B 36)
  (h3 : ∃ A : ℕ, A = move_last_to_first B) :
  ∃ A_min A_max : ℕ, 
    (A_min = move_last_to_first B ∧ A_min ≥ 17777779) ∧
    (A_max = move_last_to_first B ∧ A_max ≤ 99999998) ∧
    (∀ A : ℕ, A = move_last_to_first B → A_min ≤ A ∧ A ≤ A_max) :=
by sorry

end NUMINAMATH_CALUDE_eight_digit_number_theorem_l1030_103086


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1030_103090

-- 1. 0.175÷0.25÷4 = 0.175
theorem problem_1 : (0.175 / 0.25) / 4 = 0.175 := by sorry

-- 2. 1.4×99+1.4 = 140
theorem problem_2 : 1.4 * 99 + 1.4 = 140 := by sorry

-- 3. 3.6÷4-1.2×6 = -6.3
theorem problem_3 : 3.6 / 4 - 1.2 * 6 = -6.3 := by sorry

-- 4. (3.2+0.16)÷0.8 = 4.2
theorem problem_4 : (3.2 + 0.16) / 0.8 = 4.2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1030_103090


namespace NUMINAMATH_CALUDE_zeros_in_square_of_near_power_of_ten_l1030_103068

theorem zeros_in_square_of_near_power_of_ten : 
  (∃ n : ℕ, n = (10^12 - 5)^2 ∧ 
   ∃ m : ℕ, m > 0 ∧ n = m * 10^12 ∧ m % 10 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_near_power_of_ten_l1030_103068


namespace NUMINAMATH_CALUDE_equation_two_distinct_roots_l1030_103060

theorem equation_two_distinct_roots (a : ℝ) : 
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    ((Real.sqrt (6 * x₁ - x₁^2 - 4) + a - 2) * ((a - 2) * x₁ - 3 * a + 4) = 0) ∧
    ((Real.sqrt (6 * x₂ - x₂^2 - 4) + a - 2) * ((a - 2) * x₂ - 3 * a + 4) = 0)) ↔ 
  (a = 2 - Real.sqrt 5 ∨ a = 0 ∨ a = 1 ∨ (2 - 2 / Real.sqrt 5 < a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_equation_two_distinct_roots_l1030_103060


namespace NUMINAMATH_CALUDE_crow_nest_ditch_distance_crow_problem_solution_l1030_103059

/-- The distance between a crow's nest and a ditch, given the crow's flying pattern and speed. -/
theorem crow_nest_ditch_distance (trips : ℕ) (time : ℝ) (speed : ℝ) : ℝ :=
  let distance_km := speed * time / (2 * trips)
  let distance_m := distance_km * 1000
  200

/-- Proof that the distance between the nest and the ditch is 200 meters. -/
theorem crow_problem_solution :
  crow_nest_ditch_distance 15 1.5 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_crow_nest_ditch_distance_crow_problem_solution_l1030_103059


namespace NUMINAMATH_CALUDE_linear_equation_properties_l1030_103078

/-- Given a linear equation x + 2y = -6, this theorem proves:
    1. y can be expressed as y = -3 - x/2
    2. y is a negative number greater than -2 if and only if -6 < x < -2
-/
theorem linear_equation_properties (x y : ℝ) (h : x + 2 * y = -6) :
  (y = -3 - x / 2) ∧
  (y < 0 ∧ y > -2 ↔ -6 < x ∧ x < -2) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_properties_l1030_103078


namespace NUMINAMATH_CALUDE_gemma_pizza_change_l1030_103021

def pizza_order (num_pizzas : ℕ) (price_per_pizza : ℕ) (tip : ℕ) (payment : ℕ) : ℕ :=
  payment - (num_pizzas * price_per_pizza + tip)

theorem gemma_pizza_change : pizza_order 4 10 5 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gemma_pizza_change_l1030_103021


namespace NUMINAMATH_CALUDE_memory_card_picture_size_l1030_103052

theorem memory_card_picture_size 
  (total_pictures_a : ℕ) 
  (size_a : ℕ) 
  (total_pictures_b : ℕ) 
  (h1 : total_pictures_a = 3000)
  (h2 : size_a = 8)
  (h3 : total_pictures_b = 4000) :
  (total_pictures_a * size_a) / total_pictures_b = 6 :=
by sorry

end NUMINAMATH_CALUDE_memory_card_picture_size_l1030_103052


namespace NUMINAMATH_CALUDE_expression_nonnegative_l1030_103096

theorem expression_nonnegative (x : ℝ) : 
  (2*x - 6*x^2 + 9*x^3) / (9 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 :=
sorry

end NUMINAMATH_CALUDE_expression_nonnegative_l1030_103096


namespace NUMINAMATH_CALUDE_polynomial_identity_l1030_103095

theorem polynomial_identity (p : ℝ → ℝ) : 
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) →
  p 3 = 10 →
  p = fun x => x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1030_103095
