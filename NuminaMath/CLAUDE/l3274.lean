import Mathlib

namespace NUMINAMATH_CALUDE_smaller_bill_denomination_l3274_327445

/-- Given a cashier with bills of two denominations, prove the value of the smaller denomination. -/
theorem smaller_bill_denomination
  (total_bills : ℕ)
  (total_value : ℕ)
  (smaller_bills : ℕ)
  (twenty_bills : ℕ)
  (h_total_bills : total_bills = smaller_bills + twenty_bills)
  (h_total_bills_value : total_bills = 30)
  (h_total_value : total_value = 330)
  (h_smaller_bills : smaller_bills = 27)
  (h_twenty_bills : twenty_bills = 3) :
  ∃ (x : ℕ), x * smaller_bills + 20 * twenty_bills = total_value ∧ x = 10 := by
sorry


end NUMINAMATH_CALUDE_smaller_bill_denomination_l3274_327445


namespace NUMINAMATH_CALUDE_multiple_without_zero_l3274_327443

/-- A function that checks if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Prop :=
  ∃ (k : ℕ), n % (10^(k+1)) / (10^k) = 0

theorem multiple_without_zero (n : ℕ) (h : n % 10 ≠ 0) :
  ∃ (k : ℕ), k % n = 0 ∧ ¬containsZero k := by
  sorry

end NUMINAMATH_CALUDE_multiple_without_zero_l3274_327443


namespace NUMINAMATH_CALUDE_only_negative_two_less_than_negative_one_l3274_327412

theorem only_negative_two_less_than_negative_one : ∀ x : ℚ, 
  (x = 0 ∨ x = -1/2 ∨ x = 1 ∨ x = -2) → (x < -1 ↔ x = -2) :=
by
  sorry

end NUMINAMATH_CALUDE_only_negative_two_less_than_negative_one_l3274_327412


namespace NUMINAMATH_CALUDE_smallest_k_bound_l3274_327414

def S : Set (ℝ → ℝ) :=
  {f | (∀ x ∈ Set.Icc 0 1, 0 ≤ f x) ∧
       f 1 = 1 ∧
       ∀ x y, x + y ≤ 1 → f x + f y ≤ f (x + y)}

theorem smallest_k_bound (f : ℝ → ℝ) (h : f ∈ S) :
  (∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x) ∧
  ∀ k < 2, ∃ g ∈ S, ∃ x ∈ Set.Icc 0 1, g x > k * x :=
sorry

end NUMINAMATH_CALUDE_smallest_k_bound_l3274_327414


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l3274_327410

/-- A quadratic function with vertex (4, -1) passing through (0, 7) has coefficients a = 1/2, b = -4, and c = 7. -/
theorem quadratic_coefficients :
  ∀ (f : ℝ → ℝ) (a b c : ℝ),
    (∀ x, f x = a * x^2 + b * x + c) →
    (∀ x, f x = f (8 - x)) →
    f 4 = -1 →
    f 0 = 7 →
    a = (1/2 : ℝ) ∧ b = -4 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l3274_327410


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l3274_327401

theorem sum_of_quadratic_roots (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 - 6*y₁ + 8 = 0 ∧ y₂^2 - 6*y₂ + 8 = 0 ∧ y₁ + y₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l3274_327401


namespace NUMINAMATH_CALUDE_rectangular_piece_too_large_l3274_327476

theorem rectangular_piece_too_large (square_area : ℝ) (rect_area : ℝ) (ratio_length : ℝ) (ratio_width : ℝ) :
  square_area = 400 →
  rect_area = 300 →
  ratio_length = 3 →
  ratio_width = 2 →
  ∃ (rect_length : ℝ), 
    rect_length * (rect_length * ratio_width / ratio_length) = rect_area ∧
    rect_length > Real.sqrt square_area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_piece_too_large_l3274_327476


namespace NUMINAMATH_CALUDE_S_is_circle_l3274_327449

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 - 4 * Complex.I)}

-- Theorem stating that S is a circle
theorem S_is_circle : ∃ (c : ℂ) (r : ℝ), S = {z : ℂ | Complex.abs (z - c) = r} := by
  sorry

end NUMINAMATH_CALUDE_S_is_circle_l3274_327449


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l3274_327459

/-- Given three rectangles with specific side length ratios, prove the ratio of areas -/
theorem rectangle_area_ratio (a b c d e f : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) 
  (h3 : a / e = 7 / 4) 
  (h4 : b / f = 7 / 4) 
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : c ≠ 0) (h8 : d ≠ 0) (h9 : e ≠ 0) (h10 : f ≠ 0) :
  (a * b) / ((c * d) + (e * f)) = 441 / 1369 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l3274_327459


namespace NUMINAMATH_CALUDE_hemisphere_volume_l3274_327407

/-- Given a hemisphere with surface area (excluding the base) of 256π cm²,
    prove that its volume is (2048√2)/3 π cm³. -/
theorem hemisphere_volume (r : ℝ) (h : 2 * Real.pi * r^2 = 256 * Real.pi) :
  (2/3) * Real.pi * r^3 = (2048 * Real.sqrt 2 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_volume_l3274_327407


namespace NUMINAMATH_CALUDE_verandah_area_l3274_327448

/-- The area of a verandah surrounding a rectangular room -/
theorem verandah_area (room_length room_width verandah_width : ℝ) :
  room_length = 15 ∧ room_width = 12 ∧ verandah_width = 2 →
  (room_length + 2 * verandah_width) * (room_width + 2 * verandah_width) -
  room_length * room_width = 124 := by sorry

end NUMINAMATH_CALUDE_verandah_area_l3274_327448


namespace NUMINAMATH_CALUDE_julia_stairs_difference_l3274_327491

theorem julia_stairs_difference (jonny_stairs julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs < jonny_stairs / 3 →
  jonny_stairs + julia_stairs = 1685 →
  (jonny_stairs / 3) - julia_stairs = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_stairs_difference_l3274_327491


namespace NUMINAMATH_CALUDE_fifteenth_number_base5_l3274_327487

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The 15th number in base 5 counting system -/
def fifteenthNumberBase5 : List ℕ := toBase5 15

theorem fifteenth_number_base5 :
  fifteenthNumberBase5 = [3, 0] :=
sorry

end NUMINAMATH_CALUDE_fifteenth_number_base5_l3274_327487


namespace NUMINAMATH_CALUDE_power_of_four_in_expression_l3274_327434

theorem power_of_four_in_expression (x : ℕ) : 
  (2 * x + 5 + 2 = 29) → x = 11 := by sorry

end NUMINAMATH_CALUDE_power_of_four_in_expression_l3274_327434


namespace NUMINAMATH_CALUDE_divisor_condition_solutions_l3274_327478

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The condition that the number of divisors equals the cube root of 4n -/
def divisor_condition (n : ℕ) : Prop :=
  num_divisors n = (4 * n : ℝ) ^ (1/3 : ℝ)

/-- The main theorem stating that the divisor condition is satisfied only for 2, 128, and 2000 -/
theorem divisor_condition_solutions :
  ∀ n : ℕ, n > 0 → (divisor_condition n ↔ n = 2 ∨ n = 128 ∨ n = 2000) := by
  sorry


end NUMINAMATH_CALUDE_divisor_condition_solutions_l3274_327478


namespace NUMINAMATH_CALUDE_game_lives_distribution_l3274_327462

/-- Given a game with initial players, new players joining, and a total number of lives,
    calculate the number of lives per player. -/
def lives_per_player (initial_players : ℕ) (new_players : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players + new_players)

/-- Theorem: In a game with 4 initial players, 5 new players joining, and a total of 27 lives,
    each player has 3 lives. -/
theorem game_lives_distribution :
  lives_per_player 4 5 27 = 3 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_distribution_l3274_327462


namespace NUMINAMATH_CALUDE_right_triangle_integer_area_l3274_327465

theorem right_triangle_integer_area (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ S : ℕ, S * 2 = a * b := by
sorry

end NUMINAMATH_CALUDE_right_triangle_integer_area_l3274_327465


namespace NUMINAMATH_CALUDE_f_is_even_l3274_327403

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^2)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : ∀ x, f g (-x) = f g x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l3274_327403


namespace NUMINAMATH_CALUDE_broken_line_endpoint_characterization_l3274_327430

/-- A broken line from O to M -/
structure BrokenLine where
  segments : List (ℝ × ℝ)
  start_at_origin : segments.foldl (λ acc (x, y) => (acc.1 + x, acc.2 + y)) (0, 0) = (0, 0)
  unit_length : segments.foldl (λ acc (x, y) => acc + x^2 + y^2) 0 = 1

/-- Predicate to check if a broken line satisfies the intersection condition -/
def satisfies_intersection_condition (l : BrokenLine) : Prop :=
  ∀ (a b : ℝ), (∀ (x y : ℝ), (x, y) ∈ l.segments → (a * x + b * y ≠ 0 ∨ a * x + b * y ≠ 1))

theorem broken_line_endpoint_characterization (x y : ℝ) :
  (∃ (l : BrokenLine), satisfies_intersection_condition l ∧ 
   l.segments.foldl (λ acc (dx, dy) => (acc.1 + dx, acc.2 + dy)) (0, 0) = (x, y)) →
  x^2 + y^2 ≤ 1 ∧ |x| + |y| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_endpoint_characterization_l3274_327430


namespace NUMINAMATH_CALUDE_sqrt2_power0_plus_neg2_power3_l3274_327477

theorem sqrt2_power0_plus_neg2_power3 : (Real.sqrt 2) ^ 0 + (-2) ^ 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_power0_plus_neg2_power3_l3274_327477


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3274_327428

/-- Given a square with perimeter 80 units divided into two congruent rectangles
    by a horizontal line, prove that the perimeter of one rectangle is 60 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 80) :
  let square_side := square_perimeter / 4
  let rectangle_width := square_side
  let rectangle_height := square_side / 2
  2 * (rectangle_width + rectangle_height) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3274_327428


namespace NUMINAMATH_CALUDE_complement_union_equality_l3274_327469

-- Define the sets M, N, and U
variable (M N U : Set α)

-- Define the conditions
variable (hM : M.Nonempty)
variable (hN : N.Nonempty)
variable (hU : U.Nonempty)
variable (hMN : M ⊆ N)
variable (hNU : N ⊆ U)

-- State the theorem
theorem complement_union_equality :
  (U \ M) ∪ (U \ N) = U \ M :=
sorry

end NUMINAMATH_CALUDE_complement_union_equality_l3274_327469


namespace NUMINAMATH_CALUDE_probability_exactly_three_less_than_seven_l3274_327489

def probability_less_than_7 : ℚ := 1 / 2

def number_of_dice : ℕ := 6

def target_count : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_exactly_three_less_than_seven :
  (choose number_of_dice target_count : ℚ) * probability_less_than_7^target_count * (1 - probability_less_than_7)^(number_of_dice - target_count) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_exactly_three_less_than_seven_l3274_327489


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l3274_327468

/-- The curve defined by xy = 2 -/
def curve (x y : ℝ) : Prop := x * y = 2

/-- An arbitrary ellipse in the coordinate plane -/
def ellipse (x y : ℝ) : Prop := sorry

/-- The four points of intersection satisfy both the curve and ellipse equations -/
axiom intersection_points (x y : ℝ) : curve x y ∧ ellipse x y ↔ 
  (x = 3 ∧ y = 2/3) ∨ (x = -4 ∧ y = -1/2) ∨ (x = 1/4 ∧ y = 8) ∨ (x = -2/3 ∧ y = -3)

theorem fourth_intersection_point : 
  ∃ (x y : ℝ), curve x y ∧ ellipse x y ∧ x = -2/3 ∧ y = -3 :=
sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l3274_327468


namespace NUMINAMATH_CALUDE_dorothy_sea_glass_count_l3274_327490

-- Define the sea glass counts for Blanche and Rose
def blanche_green : ℕ := 12
def blanche_red : ℕ := 3
def rose_red : ℕ := 9
def rose_blue : ℕ := 11

-- Define Dorothy's sea glass counts based on the conditions
def dorothy_red : ℕ := 2 * (blanche_red + rose_red)
def dorothy_blue : ℕ := 3 * rose_blue

-- Define Dorothy's total sea glass count
def dorothy_total : ℕ := dorothy_red + dorothy_blue

-- Theorem to prove
theorem dorothy_sea_glass_count : dorothy_total = 57 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_sea_glass_count_l3274_327490


namespace NUMINAMATH_CALUDE_truck_rental_percentage_l3274_327463

/-- The percentage of trucks returned given the total number of trucks,
    the number of trucks rented out, and the number of trucks returned -/
def percentage_returned (total : ℕ) (rented : ℕ) (returned : ℕ) : ℚ :=
  (returned : ℚ) / (rented : ℚ) * 100

theorem truck_rental_percentage (total : ℕ) (rented : ℕ) (returned : ℕ)
  (h_total : total = 24)
  (h_rented : rented = total)
  (h_returned : returned ≥ 12) :
  percentage_returned total rented returned = 50 := by
sorry

end NUMINAMATH_CALUDE_truck_rental_percentage_l3274_327463


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3274_327464

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3274_327464


namespace NUMINAMATH_CALUDE_lauren_mail_total_l3274_327433

/-- The total number of pieces of mail sent by Lauren over four days -/
def total_mail (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem stating the total number of pieces of mail sent by Lauren -/
theorem lauren_mail_total : ∃ (monday tuesday wednesday thursday : ℕ),
  monday = 65 ∧
  tuesday = monday + 10 ∧
  wednesday = tuesday - 5 ∧
  thursday = wednesday + 15 ∧
  total_mail monday tuesday wednesday thursday = 295 :=
by sorry

end NUMINAMATH_CALUDE_lauren_mail_total_l3274_327433


namespace NUMINAMATH_CALUDE_inequality_proof_l3274_327497

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * b * c) + 1 ≥ 3 * (1 / (a^2 + b^2 + c^2) + 1 / (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3274_327497


namespace NUMINAMATH_CALUDE_only_pairD_not_opposite_l3274_327415

-- Define a structure for a pair of quantities
structure QuantityPair where
  first : String
  second : String

-- Define the function to check if a pair has opposite meanings
def hasOppositeMeanings (pair : QuantityPair) : Bool :=
  match pair with
  | ⟨"Income of 200 yuan", "expenditure of 20 yuan"⟩ => true
  | ⟨"Rise of 10 meters", "fall of 7 meters"⟩ => true
  | ⟨"Exceeding 0.05 mm", "falling short of 0.03 m"⟩ => true
  | ⟨"Increase of 2 years", "decrease of 2 liters"⟩ => false
  | _ => false

-- Define the pairs
def pairA : QuantityPair := ⟨"Income of 200 yuan", "expenditure of 20 yuan"⟩
def pairB : QuantityPair := ⟨"Rise of 10 meters", "fall of 7 meters"⟩
def pairC : QuantityPair := ⟨"Exceeding 0.05 mm", "falling short of 0.03 m"⟩
def pairD : QuantityPair := ⟨"Increase of 2 years", "decrease of 2 liters"⟩

-- Theorem statement
theorem only_pairD_not_opposite : 
  (hasOppositeMeanings pairA = true) ∧ 
  (hasOppositeMeanings pairB = true) ∧ 
  (hasOppositeMeanings pairC = true) ∧ 
  (hasOppositeMeanings pairD = false) :=
sorry

end NUMINAMATH_CALUDE_only_pairD_not_opposite_l3274_327415


namespace NUMINAMATH_CALUDE_quadratic_shift_l3274_327451

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := 2 * x^2

-- Define the vertical shift
def vertical_shift : ℝ := 1

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function x + vertical_shift

-- Theorem statement
theorem quadratic_shift :
  ∀ x : ℝ, shifted_function x = 2 * x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_shift_l3274_327451


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3274_327453

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) :
  x^3 + 1/x^3 = -18 := by sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3274_327453


namespace NUMINAMATH_CALUDE_new_cube_edge_l3274_327474

/-- Given three cubes with edges 6 cm, 8 cm, and 10 cm, prove that when melted and formed into a new cube, the edge of the new cube is 12 cm. -/
theorem new_cube_edge (cube1 cube2 cube3 new_cube : ℝ) : 
  cube1 = 6 → cube2 = 8 → cube3 = 10 → 
  (cube1^3 + cube2^3 + cube3^3)^(1/3) = new_cube → 
  new_cube = 12 := by
sorry

#eval (6^3 + 8^3 + 10^3)^(1/3) -- This should evaluate to 12

end NUMINAMATH_CALUDE_new_cube_edge_l3274_327474


namespace NUMINAMATH_CALUDE_area_of_arcsin_cos_l3274_327402

open Set
open MeasureTheory
open Interval

noncomputable def f (x : ℝ) := Real.arcsin (Real.cos x)

theorem area_of_arcsin_cos (a b : ℝ) (h : 0 ≤ a ∧ b = 2 * Real.pi) :
  (∫ x in a..b, |f x| ) = Real.pi^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_arcsin_cos_l3274_327402


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_equality_l3274_327439

theorem floor_sqrt_sum_equality (n : ℕ) : 
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (n + 1) + Real.sqrt (n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_equality_l3274_327439


namespace NUMINAMATH_CALUDE_repair_cost_equals_profit_l3274_327475

/-- Proves that the repair cost equals the profit under given conditions --/
theorem repair_cost_equals_profit (original_cost : ℝ) : 
  let repair_cost := 0.1 * original_cost
  let selling_price := 1.2 * original_cost
  let profit := selling_price - (original_cost + repair_cost)
  profit = 1100 ∧ profit / original_cost = 0.2 → repair_cost = 1100 := by
sorry

end NUMINAMATH_CALUDE_repair_cost_equals_profit_l3274_327475


namespace NUMINAMATH_CALUDE_sports_club_non_athletic_parents_l3274_327413

/-- Represents a sports club with members and their parents' athletic status -/
structure SportsClub where
  total_members : ℕ
  athletic_dads : ℕ
  athletic_moms : ℕ
  both_athletic : ℕ
  no_dads : ℕ

/-- Calculates the number of members with non-athletic parents in a sports club -/
def members_with_non_athletic_parents (club : SportsClub) : ℕ :=
  club.total_members - (club.athletic_dads + club.athletic_moms - club.both_athletic - club.no_dads)

/-- Theorem stating the number of members with non-athletic parents in the given sports club -/
theorem sports_club_non_athletic_parents :
  let club : SportsClub := {
    total_members := 50,
    athletic_dads := 25,
    athletic_moms := 30,
    both_athletic := 10,
    no_dads := 5
  }
  members_with_non_athletic_parents club = 10 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_non_athletic_parents_l3274_327413


namespace NUMINAMATH_CALUDE_circle_condition_chord_length_l3274_327422

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem for the range of m
theorem circle_condition (m : ℝ) :
  (∃ x y, circle_equation x y m) ↔ (m < 1 ∨ m > 4) :=
sorry

-- Theorem for the chord length
theorem chord_length :
  let m : ℝ := -2
  let center : ℝ × ℝ := (-2, 2)
  let radius : ℝ := 3 * Real.sqrt 2
  let d : ℝ := Real.sqrt 5
  2 * Real.sqrt (radius^2 - d^2) = 2 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_chord_length_l3274_327422


namespace NUMINAMATH_CALUDE_smallest_number_is_21_l3274_327473

/-- A sequence of 25 consecutive natural numbers satisfying certain conditions -/
def ConsecutiveSequence (start : ℕ) : Prop :=
  ∃ (seq : Fin 25 → ℕ),
    (∀ i, seq i = start + i) ∧
    (((Finset.filter (λ i => seq i % 2 = 0) Finset.univ).card : ℚ) / 25 = 12 / 25) ∧
    (((Finset.filter (λ i => seq i < 30) Finset.univ).card : ℚ) / 25 = 9 / 25)

/-- The smallest number in the sequence is 21 -/
theorem smallest_number_is_21 :
  ∃ (start : ℕ), ConsecutiveSequence start ∧ ∀ s, ConsecutiveSequence s → start ≤ s :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_is_21_l3274_327473


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_43_l3274_327437

theorem least_positive_integer_multiple_of_43 :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), y < x → ¬(43 ∣ (2*y)^2 + 2*33*(2*y) + 33^2)) ∧ 
    (43 ∣ (2*x)^2 + 2*33*(2*x) + 33^2) ∧
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_43_l3274_327437


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l3274_327456

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l3274_327456


namespace NUMINAMATH_CALUDE_tangency_lines_through_diagonal_intersection_l3274_327400

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Vector Point 4

-- Function to check if a quadrilateral is circumscribed around a circle
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Function to get tangency points
def tangency_points (q : Quadrilateral) (c : Circle) : Vector Point 4 := sorry

-- Function to get lines connecting opposite tangency points
def opposite_tangency_lines (q : Quadrilateral) (c : Circle) : Vector Line 2 := sorry

-- Function to get diagonals of a quadrilateral
def diagonals (q : Quadrilateral) : Vector Line 2 := sorry

-- Function to check if two lines intersect
def lines_intersect (l1 : Line) (l2 : Line) : Prop := sorry

-- Function to get intersection point of two lines
def intersection_point (l1 : Line) (l2 : Line) : Point := sorry

-- Theorem statement
theorem tangency_lines_through_diagonal_intersection 
  (q : Quadrilateral) (c : Circle) : 
  is_circumscribed q c → 
  let tl := opposite_tangency_lines q c
  let d := diagonals q
  lines_intersect tl[0] tl[1] ∧ 
  lines_intersect d[0] d[1] ∧
  intersection_point tl[0] tl[1] = intersection_point d[0] d[1] := by
  sorry

end NUMINAMATH_CALUDE_tangency_lines_through_diagonal_intersection_l3274_327400


namespace NUMINAMATH_CALUDE_sin_690_degrees_l3274_327454

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l3274_327454


namespace NUMINAMATH_CALUDE_symmetric_points_product_l3274_327483

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are opposites of each other -/
def symmetric_wrt_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_product (a b : ℝ) :
  symmetric_wrt_x_axis (2, a) (b + 1, 3) → a * b = -3 := by
  sorry

#check symmetric_points_product

end NUMINAMATH_CALUDE_symmetric_points_product_l3274_327483


namespace NUMINAMATH_CALUDE_train_length_train_length_approx_145m_l3274_327467

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms : ℝ := speed_kmh * (1000 / 3600)
  speed_ms * time_s

/-- Proof that a train's length is approximately 145 meters -/
theorem train_length_approx_145m (speed_kmh : ℝ) (time_s : ℝ)
  (h1 : speed_kmh = 58)
  (h2 : time_s = 9) :
  ∃ ε > 0, |train_length speed_kmh time_s - 145| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_approx_145m_l3274_327467


namespace NUMINAMATH_CALUDE_triangle_side_length_l3274_327480

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A + C = 2 * B → 
  A + B + C = π → 
  a = 1 → 
  b = Real.sqrt 3 → 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos B) →
  c = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3274_327480


namespace NUMINAMATH_CALUDE_solution_of_square_eq_zero_l3274_327450

theorem solution_of_square_eq_zero :
  ∀ x : ℝ, x^2 = 0 ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_solution_of_square_eq_zero_l3274_327450


namespace NUMINAMATH_CALUDE_initial_volumes_l3274_327471

/-- Represents a cubic container with water --/
structure Container where
  capacity : ℝ
  initialVolume : ℝ
  currentVolume : ℝ

/-- The problem setup --/
def problemSetup : (Container × Container × Container) → Prop := fun (a, b, c) =>
  -- Capacities in ratio 1:8:27
  b.capacity = 8 * a.capacity ∧ c.capacity = 27 * a.capacity ∧
  -- Initial volumes in ratio 1:2:3
  b.initialVolume = 2 * a.initialVolume ∧ c.initialVolume = 3 * a.initialVolume ∧
  -- Same depth after first transfer
  a.currentVolume / a.capacity = b.currentVolume / b.capacity ∧
  b.currentVolume / b.capacity = c.currentVolume / c.capacity ∧
  -- Transfer from C to B
  ∃ (transferCB : ℝ), transferCB = 128 * (4/7) ∧
    c.currentVolume = c.initialVolume - transferCB ∧
    b.currentVolume = b.initialVolume + transferCB ∧
  -- Transfer from B to A, A's depth becomes twice B's
  ∃ (transferBA : ℝ), 
    a.currentVolume / a.capacity = 2 * (b.currentVolume - transferBA) / b.capacity ∧
  -- A has 10θ liters less than initially
  ∃ (θ : ℝ), a.currentVolume = a.initialVolume - 10 * θ

/-- The theorem to prove --/
theorem initial_volumes (a b c : Container) :
  problemSetup (a, b, c) →
  a.initialVolume = 500 ∧ b.initialVolume = 1000 ∧ c.initialVolume = 1500 := by
  sorry

end NUMINAMATH_CALUDE_initial_volumes_l3274_327471


namespace NUMINAMATH_CALUDE_not_even_if_symmetric_to_x_squared_l3274_327424

-- Define the function g(x) = x^2 for x ≥ 0
def g (x : ℝ) : ℝ := x^2

-- Define symmetry with respect to y = x
def symmetricToYEqualsX (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Define an even function
def isEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem not_even_if_symmetric_to_x_squared (f : ℝ → ℝ) 
  (h_sym : symmetricToYEqualsX f g) : ¬ (isEven f) := by
  sorry

end NUMINAMATH_CALUDE_not_even_if_symmetric_to_x_squared_l3274_327424


namespace NUMINAMATH_CALUDE_saloon_prices_l3274_327447

/-- The cost of items in a saloon -/
structure SaloonPrices where
  sandwich : ℚ
  coffee : ℚ
  donut : ℚ

/-- The total cost of a purchase -/
def total_cost (p : SaloonPrices) (s c d : ℕ) : ℚ :=
  s * p.sandwich + c * p.coffee + d * p.donut

/-- The prices in the saloon satisfy the given conditions -/
def satisfies_conditions (p : SaloonPrices) : Prop :=
  total_cost p 4 1 10 = 169/100 ∧ total_cost p 3 1 7 = 126/100

theorem saloon_prices (p : SaloonPrices) (h : satisfies_conditions p) :
  total_cost p 1 1 1 = 40/100 := by
  sorry

end NUMINAMATH_CALUDE_saloon_prices_l3274_327447


namespace NUMINAMATH_CALUDE_max_profit_selling_price_l3274_327440

/-- Represents the profit function for a product sale --/
def profit_function (initial_cost initial_price initial_sales price_sensitivity : ℝ) (x : ℝ) : ℝ :=
  (x - initial_cost) * (initial_sales - (x - initial_price) * price_sensitivity)

/-- Theorem stating the maximum profit and optimal selling price --/
theorem max_profit_selling_price 
  (initial_cost : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (price_sensitivity : ℝ)
  (h_initial_cost : initial_cost = 8)
  (h_initial_price : initial_price = 10)
  (h_initial_sales : initial_sales = 60)
  (h_price_sensitivity : price_sensitivity = 10) :
  ∃ (max_profit optimal_price : ℝ),
    max_profit = 160 ∧ 
    optimal_price = 12 ∧
    ∀ x, profit_function initial_cost initial_price initial_sales price_sensitivity x ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_max_profit_selling_price_l3274_327440


namespace NUMINAMATH_CALUDE_scientific_notation_of_20_8_billion_l3274_327484

/-- Expresses 20.8 billion in scientific notation -/
theorem scientific_notation_of_20_8_billion :
  20.8 * (10 : ℝ)^9 = 2.08 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_20_8_billion_l3274_327484


namespace NUMINAMATH_CALUDE_trailer_homes_proof_l3274_327498

/-- Represents the number of new trailer homes added -/
def new_homes : ℕ := 17

/-- Represents the initial number of trailer homes -/
def initial_homes : ℕ := 25

/-- Represents the initial average age of trailer homes in years -/
def initial_avg_age : ℚ := 15

/-- Represents the current average age of all trailer homes in years -/
def current_avg_age : ℚ := 12

/-- Represents the time elapsed since new homes were added, in years -/
def years_passed : ℕ := 3

theorem trailer_homes_proof :
  (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
  (initial_homes + new_homes) = current_avg_age :=
sorry

end NUMINAMATH_CALUDE_trailer_homes_proof_l3274_327498


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3274_327438

/-- The parabola function -/
def f (x : ℝ) : ℝ := -(x + 2)^2 + 6

/-- The y-axis -/
def y_axis : Set ℝ := {x | x = 0}

/-- Theorem: The intersection point of the parabola and the y-axis is (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃! p : ℝ × ℝ, p.1 ∈ y_axis ∧ p.2 = f p.1 ∧ p = (0, 2) := by
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3274_327438


namespace NUMINAMATH_CALUDE_parabola_focus_l3274_327460

/-- The focus of a parabola y^2 = 8x with directrix x + 2 = 0 is at (2,0) -/
theorem parabola_focus (x y : ℝ) : 
  (y^2 = 8*x) →  -- point (x,y) is on the parabola
  (∀ (a b : ℝ), (a + 2 = 0) → ((x - a)^2 + (y - b)^2 = 4)) → -- distance to directrix equals distance to (2,0)
  (x = 2 ∧ y = 0) -- focus is at (2,0)
  := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l3274_327460


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l3274_327427

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ n - 17 ≠ 0 ∧ 7*n + 8 ≠ 0 ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (7*n + 8)) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    m - 17 = 0 ∨ 7*m + 8 = 0 ∨
    (∀ (j : ℕ), j > 1 → ¬(j ∣ (m - 17) ∧ j ∣ (7*m + 8)))) ∧
  n = 144 :=
by sorry


end NUMINAMATH_CALUDE_least_reducible_fraction_l3274_327427


namespace NUMINAMATH_CALUDE_num_sandwich_combinations_l3274_327492

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 3

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 5

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 4

/-- Represents the number of sandwiches excluded due to the turkey/swiss cheese combination. -/
def turkey_swiss_exclusions : ℕ := num_bread

/-- Represents the number of sandwiches excluded due to the roast beef/rye bread combination. -/
def roast_beef_rye_exclusions : ℕ := num_cheese

/-- Calculates the total number of possible sandwich combinations without restrictions. -/
def total_combinations : ℕ := num_bread * num_meat * num_cheese

/-- Theorem stating that the number of different sandwiches that can be ordered is 53. -/
theorem num_sandwich_combinations : 
  total_combinations - turkey_swiss_exclusions - roast_beef_rye_exclusions = 53 := by
  sorry

end NUMINAMATH_CALUDE_num_sandwich_combinations_l3274_327492


namespace NUMINAMATH_CALUDE_max_table_sum_l3274_327461

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 4 ∧ 
  (∀ x ∈ top, x ∈ primes) ∧ 
  (∀ x ∈ left, x ∈ primes) ∧
  17 ∈ top ∧
  (∀ x ∈ primes, x ∈ top ∨ x ∈ left) ∧
  (∀ x ∈ top, ∀ y ∈ left, x ≠ y)

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum) * (left.sum)

theorem max_table_sum :
  ∀ top left, is_valid_arrangement top left →
  table_sum top left ≤ 825 :=
sorry

end NUMINAMATH_CALUDE_max_table_sum_l3274_327461


namespace NUMINAMATH_CALUDE_pyramid_properties_l3274_327446

/-- Represents a pyramid ABCD with given edge lengths -/
structure Pyramid where
  DA : ℝ
  DB : ℝ
  DC : ℝ
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- The specific pyramid from the problem -/
def specific_pyramid : Pyramid :=
  { DA := 15
    DB := 12
    DC := 12
    AB := 9
    AC := 9
    BC := 3 }

/-- Calculates the radius of the circumscribed sphere around the pyramid -/
def circumscribed_sphere_radius (p : Pyramid) : ℝ := sorry

/-- Calculates the volume of the pyramid -/
def pyramid_volume (p : Pyramid) : ℝ := sorry

theorem pyramid_properties :
  circumscribed_sphere_radius specific_pyramid = 7.5 ∧
  pyramid_volume specific_pyramid = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_properties_l3274_327446


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l3274_327466

theorem min_sum_of_reciprocal_sum_eq_one (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 1 / a + 1 / b = 1) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1 → a + b ≤ x + y ∧ a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l3274_327466


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3274_327404

/-- Calculates the amount of money John spent out of pocket to buy a computer and accessories after selling his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value discount_rate : ℝ) 
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400)
  (h4 : discount_rate = 0.2) : 
  computer_cost + accessories_cost - playstation_value * (1 - discount_rate) = 580 := by
  sorry

end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3274_327404


namespace NUMINAMATH_CALUDE_stamp_collection_theorem_l3274_327423

/-- Calculates the total value of a stamp collection given the following conditions:
    - The total number of stamps in the collection
    - The number of stamps in a subset of the collection
    - The total value of the subset of stamps
    Assumes that all stamps have the same value. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * (subset_value / subset_stamps)

/-- Proves that a collection of 18 stamps, where 6 stamps are worth 18 dollars,
    has a total value of 54 dollars. -/
theorem stamp_collection_theorem :
  stamp_collection_value 18 6 18 = 54 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_theorem_l3274_327423


namespace NUMINAMATH_CALUDE_malcolm_followers_difference_l3274_327495

def malcolm_social_media (instagram_followers facebook_followers : ℕ) : Prop :=
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  ∃ (youtube_followers : ℕ),
    youtube_followers > tiktok_followers ∧
    instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 ∧
    youtube_followers - tiktok_followers = 510

theorem malcolm_followers_difference :
  malcolm_social_media 240 500 :=
by sorry

end NUMINAMATH_CALUDE_malcolm_followers_difference_l3274_327495


namespace NUMINAMATH_CALUDE_angle_sum_less_than_three_halves_pi_l3274_327457

theorem angle_sum_less_than_three_halves_pi
  (α β : Real)
  (h1 : π / 2 < α ∧ α < π)
  (h2 : π / 2 < β ∧ β < π)
  (h3 : Real.tan α < Real.tan (π / 2 - β)) :
  α + β < 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_less_than_three_halves_pi_l3274_327457


namespace NUMINAMATH_CALUDE_expression_bounds_l3274_327485

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1/2) (hd : 0 ≤ d ∧ d ≤ 1/2) :
  let expr := Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
               Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2)
  2 * Real.sqrt 2 ≤ expr ∧ expr ≤ 4 ∧ 
  ∀ x, 2 * Real.sqrt 2 ≤ x ∧ x ≤ 4 → ∃ a b c d, 
    0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1/2 ∧ 0 ≤ d ∧ d ≤ 1/2 ∧
    x = Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
        Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l3274_327485


namespace NUMINAMATH_CALUDE_cylinder_volume_approximation_l3274_327488

/-- The volume of a cylinder with diameter 14 cm and height 2 cm is approximately 307.88 cubic centimeters. -/
theorem cylinder_volume_approximation :
  let d : ℝ := 14  -- diameter in cm
  let h : ℝ := 2   -- height in cm
  let r : ℝ := d / 2  -- radius in cm
  let π : ℝ := Real.pi
  let V : ℝ := π * r^2 * h  -- volume formula
  ∃ ε > 0, abs (V - 307.88) < ε ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_approximation_l3274_327488


namespace NUMINAMATH_CALUDE_bryan_work_hours_l3274_327486

/-- Represents Bryan's daily work schedule --/
structure WorkSchedule where
  customer_outreach : ℝ
  advertisement : ℝ
  marketing : ℝ

/-- Calculates the total working hours given a work schedule --/
def total_hours (schedule : WorkSchedule) : ℝ :=
  schedule.customer_outreach + schedule.advertisement + schedule.marketing

/-- Theorem stating Bryan's total working hours --/
theorem bryan_work_hours :
  ∀ (schedule : WorkSchedule),
    schedule.customer_outreach = 4 →
    schedule.advertisement = schedule.customer_outreach / 2 →
    schedule.marketing = 2 →
    total_hours schedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_bryan_work_hours_l3274_327486


namespace NUMINAMATH_CALUDE_solve_candy_problem_l3274_327405

def candy_problem (packs : ℕ) (paid : ℕ) (change : ℕ) : Prop :=
  packs = 3 ∧ paid = 20 ∧ change = 11 →
  (paid - change) / packs = 3

theorem solve_candy_problem : candy_problem 3 20 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l3274_327405


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_115_2_l3274_327418

theorem percentage_of_360_equals_115_2 : 
  let whole : ℝ := 360
  let part : ℝ := 115.2
  let percentage : ℝ := (part / whole) * 100
  percentage = 32 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_115_2_l3274_327418


namespace NUMINAMATH_CALUDE_square_area_l3274_327408

/-- Given a square ABCD composed of two identical rectangles and two squares with side lengths 2 cm and 4 cm respectively, prove that the area of ABCD is 36 cm². -/
theorem square_area (s : ℝ) (h1 : s > 0) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  a + 2 = 4 ∧
  b + 4 = s ∧
  s = 6 ∧
  s^2 = 36 := by
sorry

end NUMINAMATH_CALUDE_square_area_l3274_327408


namespace NUMINAMATH_CALUDE_points_needed_for_average_l3274_327479

/-- 
Given a basketball player who has scored 333 points in 10 games, 
this theorem proves that the player needs to score 41 points in the 11th game 
to achieve an average of 34 points over 11 games.
-/
theorem points_needed_for_average (total_points : ℕ) (num_games : ℕ) (target_average : ℕ) :
  total_points = 333 →
  num_games = 10 →
  target_average = 34 →
  (total_points + 41) / (num_games + 1) = target_average := by
  sorry

end NUMINAMATH_CALUDE_points_needed_for_average_l3274_327479


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3274_327470

theorem quadratic_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) →
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4*a*c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3274_327470


namespace NUMINAMATH_CALUDE_equation_solution_l3274_327429

theorem equation_solution : 
  {x : ℝ | x * (x - 14) = 0} = {0, 14} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3274_327429


namespace NUMINAMATH_CALUDE_left_handed_rock_lovers_l3274_327426

theorem left_handed_rock_lovers (total : Nat) (left_handed : Nat) (rock_lovers : Nat) (right_handed_non_rock : Nat)
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : rock_lovers = 18)
  (h4 : right_handed_non_rock = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ x : Nat, x = left_handed + rock_lovers - total + right_handed_non_rock ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_rock_lovers_l3274_327426


namespace NUMINAMATH_CALUDE_saline_solution_concentration_l3274_327425

/-- Proves that given a tank with 100 gallons of pure water and 66.67 gallons of saline solution
    added to create a 10% salt solution, the original saline solution must have contained 25% salt. -/
theorem saline_solution_concentration
  (pure_water : ℝ)
  (saline_added : ℝ)
  (final_concentration : ℝ)
  (h1 : pure_water = 100)
  (h2 : saline_added = 66.67)
  (h3 : final_concentration = 0.1)
  : (final_concentration * (pure_water + saline_added)) / saline_added = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_saline_solution_concentration_l3274_327425


namespace NUMINAMATH_CALUDE_expected_no_allergies_is_75_l3274_327421

/-- The probability that an American does not suffer from allergies -/
def prob_no_allergies : ℚ := 1/4

/-- The size of the random sample of Americans -/
def sample_size : ℕ := 300

/-- The expected number of people in the sample who do not suffer from allergies -/
def expected_no_allergies : ℚ := prob_no_allergies * sample_size

theorem expected_no_allergies_is_75 : expected_no_allergies = 75 := by
  sorry

end NUMINAMATH_CALUDE_expected_no_allergies_is_75_l3274_327421


namespace NUMINAMATH_CALUDE_sum_of_xy_l3274_327452

theorem sum_of_xy (x y : ℝ) 
  (eq1 : x^2 + 3*x*y + y^2 = 909)
  (eq2 : 3*x^2 + x*y + 3*y^2 = 1287) :
  x + y = 27 ∨ x + y = -27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l3274_327452


namespace NUMINAMATH_CALUDE_last_three_digits_of_3_800_l3274_327472

theorem last_three_digits_of_3_800 (h : 3^400 ≡ 1 [ZMOD 500]) :
  3^800 ≡ 1 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_3_800_l3274_327472


namespace NUMINAMATH_CALUDE_skew_iff_a_neq_zero_l3274_327444

def line1 (a t : ℝ) : Fin 4 → ℝ := fun i =>
  match i with
  | 0 => 1 + 2*t
  | 1 => 3 + 4*t
  | 2 => 0 + 1*t
  | 3 => a + 3*t

def line2 (u : ℝ) : Fin 4 → ℝ := fun i =>
  match i with
  | 0 => 3 + 4*u
  | 1 => 4 + 5*u
  | 2 => 1 + 2*u
  | 3 => 0 + 1*u

def are_skew (a : ℝ) : Prop :=
  ∀ t u : ℝ, line1 a t ≠ line2 u

theorem skew_iff_a_neq_zero (a : ℝ) :
  are_skew a ↔ a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_skew_iff_a_neq_zero_l3274_327444


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l3274_327416

/-- Given two sets A and B, prove that if they are equal and have the specified form,
    then x = 2 and y = 2 -/
theorem set_equality_implies_values (x y : ℝ) : 
  ({x, y^2, 1} : Set ℝ) = ({1, 2*x, y} : Set ℝ) → x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l3274_327416


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3274_327409

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  -3*x*y - 3*x^2 + 4*x*y + 2*x^2 = x*y - x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  3*(a^2 - 2*a*b) - 5*(a^2 + 4*a*b) = -2*a^2 - 26*a*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3274_327409


namespace NUMINAMATH_CALUDE_monotonic_function_constraint_l3274_327411

theorem monotonic_function_constraint (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x ∈ Set.Icc (-1) 2, Monotone (fun x => -1/3 * x^3 + a * x^2 + b * x)) →
  a + b ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_constraint_l3274_327411


namespace NUMINAMATH_CALUDE_plane_perpendicular_theorem_l3274_327441

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_theorem 
  (α β : Plane) (n : Line) 
  (h1 : contains β n) 
  (h2 : perpendicular n α) :
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_theorem_l3274_327441


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3274_327481

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 = 1 - a 1 →
  a 4 = 9 - a 3 →
  a 4 + a 5 = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3274_327481


namespace NUMINAMATH_CALUDE_coinciding_directrices_l3274_327435

/-- Given a hyperbola and a parabola with coinciding directrices, prove that p = 3 -/
theorem coinciding_directrices (p : ℝ) : p > 0 → ∃ (x y : ℝ),
  (x^2 / 3 - y^2 = 1 ∧ y^2 = 2*p*x ∧ 
   (x = -3/2 ∨ x = 3/2) ∧ x = -p/2) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_directrices_l3274_327435


namespace NUMINAMATH_CALUDE_sum_of_distances_constant_l3274_327406

/-- A regular tetrahedron in three-dimensional space -/
structure RegularTetrahedron where
  -- Define the properties of a regular tetrahedron here
  -- (We don't need to fully define it for this statement)

/-- A point inside a regular tetrahedron -/
structure InnerPoint (t : RegularTetrahedron) where
  -- Define the properties of an inner point here
  -- (We don't need to fully define it for this statement)

/-- The sum of distances from a point to all faces of a regular tetrahedron -/
def sum_of_distances_to_faces (t : RegularTetrahedron) (p : InnerPoint t) : ℝ :=
  sorry -- Definition would go here

/-- Theorem stating that the sum of distances from any point inside a regular tetrahedron to its faces is constant -/
theorem sum_of_distances_constant (t : RegularTetrahedron) :
  ∃ c : ℝ, ∀ p : InnerPoint t, sum_of_distances_to_faces t p = c :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_constant_l3274_327406


namespace NUMINAMATH_CALUDE_scaling_transformation_correct_l3274_327458

/-- Scaling transformation function -/
def scale (sx sy : ℚ) (p : ℚ × ℚ) : ℚ × ℚ :=
  (sx * p.1, sy * p.2)

/-- The initial point -/
def initial_point : ℚ × ℚ := (1, 2)

/-- The scaling factors -/
def sx : ℚ := 1/2
def sy : ℚ := 1/3

/-- The expected result after transformation -/
def expected_result : ℚ × ℚ := (1/2, 2/3)

theorem scaling_transformation_correct :
  scale sx sy initial_point = expected_result := by
  sorry

end NUMINAMATH_CALUDE_scaling_transformation_correct_l3274_327458


namespace NUMINAMATH_CALUDE_percentage_good_fruits_is_87_point_6_percent_l3274_327436

/-- Calculates the percentage of fruits in good condition given the quantities and spoilage rates --/
def percentageGoodFruits (oranges bananas apples pears : ℕ) 
  (orangesSpoilage bananaSpoilage appleSpoilage pearSpoilage : ℚ) : ℚ :=
  let totalFruits := oranges + bananas + apples + pears
  let goodOranges := oranges - (oranges * orangesSpoilage).floor
  let goodBananas := bananas - (bananas * bananaSpoilage).floor
  let goodApples := apples - (apples * appleSpoilage).floor
  let goodPears := pears - (pears * pearSpoilage).floor
  let totalGoodFruits := goodOranges + goodBananas + goodApples + goodPears
  (totalGoodFruits : ℚ) / (totalFruits : ℚ) * 100

/-- Theorem stating that the percentage of good fruits is 87.6% given the problem conditions --/
theorem percentage_good_fruits_is_87_point_6_percent :
  percentageGoodFruits 600 400 800 200 (15/100) (3/100) (12/100) (25/100) = 876/10 := by
  sorry


end NUMINAMATH_CALUDE_percentage_good_fruits_is_87_point_6_percent_l3274_327436


namespace NUMINAMATH_CALUDE_max_percentage_offering_either_or_both_l3274_327499

-- Define the percentage of companies offering wireless internet
def wireless_internet_percentage : ℚ := 20 / 100

-- Define the percentage of companies offering free snacks
def free_snacks_percentage : ℚ := 70 / 100

-- Theorem statement
theorem max_percentage_offering_either_or_both :
  ∃ (max_percentage : ℚ),
    max_percentage = wireless_internet_percentage + free_snacks_percentage ∧
    max_percentage ≤ 1 ∧
    ∀ (actual_percentage : ℚ),
      actual_percentage ≤ max_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_percentage_offering_either_or_both_l3274_327499


namespace NUMINAMATH_CALUDE_sss_sufficient_for_angle_construction_l3274_327419

/-- A triangle in a plane -/
structure Triangle :=
  (A B C : Point)

/-- Congruence relation between triangles -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

/-- Length of a side in a triangle -/
def SideLength (t : Triangle) (side : Fin 3) : ℝ := sorry

/-- Angle measure in a triangle -/
def AngleMeasure (t : Triangle) (angle : Fin 3) : ℝ := sorry

/-- SSS congruence criterion -/
axiom sss_congruence (t1 t2 : Triangle) :
  (∀ i : Fin 3, SideLength t1 i = SideLength t2 i) → Congruent t1 t2

/-- Compass and straightedge construction -/
def ConstructibleAngle (θ : ℝ) : Prop := sorry

/-- Theorem: SSS is sufficient for angle construction -/
theorem sss_sufficient_for_angle_construction (θ : ℝ) (t : Triangle) :
  (∃ i : Fin 3, AngleMeasure t i = θ) →
  ConstructibleAngle θ :=
sorry

end NUMINAMATH_CALUDE_sss_sufficient_for_angle_construction_l3274_327419


namespace NUMINAMATH_CALUDE_negative_eight_meters_westward_l3274_327493

-- Define the direction type
inductive Direction
| East
| West

-- Define a function to convert meters to a direction and magnitude
def metersToDirection (x : ℤ) : Direction × ℕ :=
  if x ≥ 0 then
    (Direction.East, x.natAbs)
  else
    (Direction.West, (-x).natAbs)

-- State the theorem
theorem negative_eight_meters_westward :
  metersToDirection (-8) = (Direction.West, 8) :=
sorry

end NUMINAMATH_CALUDE_negative_eight_meters_westward_l3274_327493


namespace NUMINAMATH_CALUDE_subtract_three_from_M_l3274_327420

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def M : List Bool := [false, false, false, false, true, true, false, true]

theorem subtract_three_from_M :
  decimal_to_binary (binary_to_decimal M - 3) = 
    [true, false, true, true, false, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_subtract_three_from_M_l3274_327420


namespace NUMINAMATH_CALUDE_convex_polygon_in_rectangle_l3274_327455

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and properties to define a convex polygon
  is_convex : Bool  -- Placeholder for convexity property

/-- A rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a shape -/
class HasArea (α : Type) where
  area : α → ℝ

/-- Instance for ConvexPolygon -/
instance : HasArea ConvexPolygon where
  area := sorry

/-- Instance for Rectangle -/
instance : HasArea Rectangle where
  area r := r.width * r.height

/-- A polygon is contained in a rectangle -/
def ContainedIn (p : ConvexPolygon) (r : Rectangle) : Prop :=
  sorry  -- Definition of containment

theorem convex_polygon_in_rectangle :
  ∀ (p : ConvexPolygon), HasArea.area p = 1 →
  ∃ (r : Rectangle), ContainedIn p r ∧ HasArea.area r ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_in_rectangle_l3274_327455


namespace NUMINAMATH_CALUDE_square_side_differences_l3274_327494

theorem square_side_differences (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ > a₂) (h₂ : a₂ > a₃) (h₃ : a₃ > a₄)
  (diff₁ : a₁ - a₂ = 11) (diff₂ : a₂ - a₃ = 5) (diff₃ : a₃ - a₄ = 13) :
  a₁ - a₄ = 29 := by
sorry

end NUMINAMATH_CALUDE_square_side_differences_l3274_327494


namespace NUMINAMATH_CALUDE_root_product_l3274_327496

theorem root_product (a b : ℝ) : 
  (a^2 + 2*a - 2023 = 0) → 
  (b^2 + 2*b - 2023 = 0) → 
  (a + 1) * (b + 1) = -2024 := by
sorry

end NUMINAMATH_CALUDE_root_product_l3274_327496


namespace NUMINAMATH_CALUDE_unique_postal_codes_exist_l3274_327417

def PostalCode := Fin 6 → Fin 7

def validDigits (code : PostalCode) : Prop :=
  ∀ i : Fin 6, code i < 7 ∧ code i ≠ 4

def distinctDigits (code : PostalCode) : Prop :=
  ∀ i j : Fin 6, i ≠ j → code i ≠ code j

def matchingPositions (code1 code2 : PostalCode) : Nat :=
  (List.range 6).filter (λ i => code1 i = code2 i) |>.length

def A : PostalCode := λ i => [3, 2, 0, 6, 5, 1][i]
def B : PostalCode := λ i => [1, 0, 5, 2, 6, 3][i]
def C : PostalCode := λ i => [6, 1, 2, 3, 0, 5][i]
def D : PostalCode := λ i => [3, 1, 6, 2, 5, 0][i]

theorem unique_postal_codes_exist : 
  ∃! (M N : PostalCode), 
    validDigits M ∧ validDigits N ∧
    distinctDigits M ∧ distinctDigits N ∧
    M ≠ N ∧
    (matchingPositions A M = 2 ∧ matchingPositions A N = 2) ∧
    (matchingPositions B M = 2 ∧ matchingPositions B N = 2) ∧
    (matchingPositions C M = 2 ∧ matchingPositions C N = 2) ∧
    (matchingPositions D M = 3 ∧ matchingPositions D N = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_postal_codes_exist_l3274_327417


namespace NUMINAMATH_CALUDE_prime_sequence_ones_digit_l3274_327482

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to get the ones digit of a number
def onesDigit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) : 
  isPrime p → isPrime q → isPrime r → isPrime s →
  p > 3 →
  q = p + 4 →
  r = q + 4 →
  s = r + 4 →
  onesDigit p = 9 := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_ones_digit_l3274_327482


namespace NUMINAMATH_CALUDE_remainder_102938475610_div_12_l3274_327432

theorem remainder_102938475610_div_12 : 102938475610 % 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_102938475610_div_12_l3274_327432


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3274_327431

/-- A differentiable function satisfying certain conditions -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  differentiable : Differentiable ℝ f
  domain : ∀ x, x < 0 → f x ≠ 0
  condition : ∀ x, x < 0 → 3 * f x + x * deriv f x < 0

/-- The solution set of the inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | (x + 2016)^3 * f (x + 2016) + 8 * f (-2) < 0}

theorem solution_set_characterization (f : ℝ → ℝ) [SpecialFunction f] :
  SolutionSet f = Set.Ioo (-2018) (-2016) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3274_327431


namespace NUMINAMATH_CALUDE_crayons_left_in_drawer_l3274_327442

theorem crayons_left_in_drawer (initial_crayons : ℕ) (crayons_taken : ℕ) : 
  initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 :=
by sorry

end NUMINAMATH_CALUDE_crayons_left_in_drawer_l3274_327442
