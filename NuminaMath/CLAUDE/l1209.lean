import Mathlib

namespace NUMINAMATH_CALUDE_egg_collection_difference_l1209_120921

/-- Egg collection problem -/
theorem egg_collection_difference :
  ∀ (benjamin carla trisha : ℕ),
  benjamin = 6 →
  carla = 3 * benjamin →
  benjamin + carla + trisha = 26 →
  benjamin - trisha = 4 :=
by sorry

end NUMINAMATH_CALUDE_egg_collection_difference_l1209_120921


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1209_120937

theorem arithmetic_sequence_difference : 
  let seq1_start : ℕ := 2001
  let seq1_end : ℕ := 2093
  let seq2_start : ℕ := 301
  let seq2_end : ℕ := 393
  let seq1_sum := (seq1_start + seq1_end) * (seq1_end - seq1_start + 1) / 2
  let seq2_sum := (seq2_start + seq2_end) * (seq2_end - seq2_start + 1) / 2
  seq1_sum - seq2_sum = 158100 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1209_120937


namespace NUMINAMATH_CALUDE_three_digit_number_puzzle_l1209_120948

theorem three_digit_number_puzzle (A B : ℕ) : 
  (100 ≤ A * 100 + 30 + B) ∧ 
  (A * 100 + 30 + B < 1000) ∧ 
  (A * 100 + 30 + B - 41 = 591) → 
  B = 2 := by sorry

end NUMINAMATH_CALUDE_three_digit_number_puzzle_l1209_120948


namespace NUMINAMATH_CALUDE_square_ratio_theorem_l1209_120942

theorem square_ratio_theorem (area_ratio : ℚ) (side_ratio : ℚ) 
  (a b c : ℕ) (h1 : area_ratio = 50 / 98) :
  side_ratio = Real.sqrt (area_ratio) ∧
  side_ratio = 5 / 7 ∧
  (a : ℚ) * Real.sqrt b / (c : ℚ) = side_ratio ∧
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_theorem_l1209_120942


namespace NUMINAMATH_CALUDE_existence_of_invariant_sequences_l1209_120909

/-- A binary sequence is a function from ℕ to {0, 1} -/
def BinarySeq := ℕ → Fin 2

/-- Remove odd-indexed elements from a sequence -/
def removeOdd (s : BinarySeq) : BinarySeq :=
  fun n => s (2 * n + 1)

/-- Remove even-indexed elements from a sequence -/
def removeEven (s : BinarySeq) : BinarySeq :=
  fun n => s (2 * n)

/-- A sequence is invariant under odd removal if removing odd-indexed elements results in the same sequence -/
def invariantUnderOddRemoval (s : BinarySeq) : Prop :=
  ∀ n, s n = removeOdd s n

/-- A sequence is invariant under even removal if removing even-indexed elements results in the same sequence -/
def invariantUnderEvenRemoval (s : BinarySeq) : Prop :=
  ∀ n, s n = removeEven s n

theorem existence_of_invariant_sequences :
  (∃ s : BinarySeq, invariantUnderOddRemoval s) ∧
  (∃ s : BinarySeq, invariantUnderEvenRemoval s) :=
sorry

end NUMINAMATH_CALUDE_existence_of_invariant_sequences_l1209_120909


namespace NUMINAMATH_CALUDE_fraction_equality_l1209_120961

theorem fraction_equality : (1721^2 - 1714^2) / (1728^2 - 1707^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1209_120961


namespace NUMINAMATH_CALUDE_equilateral_is_cute_specific_triangle_is_cute_right_angled_cute_triangle_side_length_l1209_120920

/-- A triangle is cute if the sum of the squares of two sides is equal to twice the square of the third side -/
def IsCuteTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2 ∨ a^2 + c^2 = 2 * b^2 ∨ b^2 + c^2 = 2 * a^2

theorem equilateral_is_cute (a : ℝ) (ha : a > 0) : IsCuteTriangle a a a :=
  sorry

theorem specific_triangle_is_cute : IsCuteTriangle 4 (2 * Real.sqrt 6) (2 * Real.sqrt 5) :=
  sorry

theorem right_angled_cute_triangle_side_length 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_cute : IsCuteTriangle a b c) 
  (h_ac : b = 2 * Real.sqrt 2) : 
  c = 2 * Real.sqrt 6 ∨ c = 2 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_equilateral_is_cute_specific_triangle_is_cute_right_angled_cute_triangle_side_length_l1209_120920


namespace NUMINAMATH_CALUDE_friends_team_assignment_l1209_120903

theorem friends_team_assignment (n : ℕ) (k : ℕ) :
  n = 8 → k = 4 → (k : ℕ) ^ n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l1209_120903


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_2_100_without_zero_l1209_120945

/-- A function that checks if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Prop :=
  ∃ k : ℕ, (n / (10^k)) % 10 = 0

/-- Theorem stating that there exists an integer divisible by 2^100 that does not contain the digit 0 -/
theorem exists_number_divisible_by_2_100_without_zero :
  ∃ n : ℕ, (n % (2^100) = 0) ∧ ¬(containsZero n) := by
  sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_2_100_without_zero_l1209_120945


namespace NUMINAMATH_CALUDE_jorge_corn_yield_l1209_120923

/-- Calculates the total corn yield from Jorge's land -/
theorem jorge_corn_yield (total_land : ℝ) (good_soil_yield : ℝ) 
  (clay_soil_fraction : ℝ) (h1 : total_land = 60) 
  (h2 : good_soil_yield = 400) (h3 : clay_soil_fraction = 1/3) : 
  total_land * (clay_soil_fraction * (good_soil_yield / 2) + 
  (1 - clay_soil_fraction) * good_soil_yield) = 20000 := by
  sorry

#check jorge_corn_yield

end NUMINAMATH_CALUDE_jorge_corn_yield_l1209_120923


namespace NUMINAMATH_CALUDE_ladder_slide_speed_l1209_120993

theorem ladder_slide_speed (x y : ℝ) (dx_dt : ℝ) (h1 : x^2 + y^2 = 5^2) 
  (h2 : x = 1.4) (h3 : dx_dt = 3) : 
  ∃ dy_dt : ℝ, 2*x*dx_dt + 2*y*dy_dt = 0 ∧ |dy_dt| = 0.875 := by
sorry

end NUMINAMATH_CALUDE_ladder_slide_speed_l1209_120993


namespace NUMINAMATH_CALUDE_club_membership_l1209_120981

theorem club_membership (total : ℕ) (difference : ℕ) (first_year : ℕ) : 
  total = 128 →
  difference = 12 →
  first_year = total / 2 + difference / 2 →
  first_year = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_club_membership_l1209_120981


namespace NUMINAMATH_CALUDE_solve_equation_l1209_120940

theorem solve_equation (x : ℚ) : (3 * x + 5) / 5 = 17 → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1209_120940


namespace NUMINAMATH_CALUDE_upstream_speed_is_eight_l1209_120991

/-- Represents the speed of a man in a stream -/
structure StreamSpeed where
  downstream : ℝ
  stream : ℝ

/-- Calculates the upstream speed given downstream and stream speeds -/
def upstreamSpeed (s : StreamSpeed) : ℝ :=
  s.downstream - 2 * s.stream

/-- Theorem stating that for given downstream and stream speeds, the upstream speed is 8 -/
theorem upstream_speed_is_eight (s : StreamSpeed) 
  (h1 : s.downstream = 15) 
  (h2 : s.stream = 3.5) : 
  upstreamSpeed s = 8 := by
  sorry

end NUMINAMATH_CALUDE_upstream_speed_is_eight_l1209_120991


namespace NUMINAMATH_CALUDE_watch_cost_price_l1209_120934

/-- The cost price of a watch, given certain selling conditions. -/
def cost_price : ℝ := 1166.67

/-- The selling price at a loss. -/
def selling_price_loss : ℝ := 0.90 * cost_price

/-- The selling price at a gain. -/
def selling_price_gain : ℝ := 1.02 * cost_price

/-- Theorem stating the cost price of the watch given the selling conditions. -/
theorem watch_cost_price :
  (selling_price_loss = 0.90 * cost_price) ∧
  (selling_price_gain = 1.02 * cost_price) ∧
  (selling_price_gain - selling_price_loss = 140) →
  cost_price = 1166.67 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1209_120934


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l1209_120965

theorem max_students_equal_distribution (pens pencils erasers notebooks : ℕ) 
  (h1 : pens = 1802)
  (h2 : pencils = 1203)
  (h3 : erasers = 1508)
  (h4 : notebooks = 2400) :
  Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l1209_120965


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l1209_120914

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  /-- The diagonals of the quadrilateral bisect each other -/
  diagonals_bisect : Bool
  /-- The diagonals of the quadrilateral are perpendicular -/
  diagonals_perpendicular : Bool
  /-- Two adjacent sides of the quadrilateral are equal -/
  two_adjacent_sides_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.diagonals_bisect ∧ q.diagonals_perpendicular ∧ q.two_adjacent_sides_equal

/-- The main theorem stating that a quadrilateral with the given properties is most likely a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.diagonals_bisect = true) 
  (h2 : q.diagonals_perpendicular = true) 
  (h3 : q.two_adjacent_sides_equal = true) : 
  is_kite q :=
sorry

end NUMINAMATH_CALUDE_special_quadrilateral_is_kite_l1209_120914


namespace NUMINAMATH_CALUDE_safari_leopards_l1209_120949

theorem safari_leopards (total_animals : ℕ) 
  (saturday_lions sunday_buffaloes monday_rhinos : ℕ)
  (saturday_elephants monday_warthogs : ℕ) :
  total_animals = 20 →
  saturday_lions = 3 →
  saturday_elephants = 2 →
  sunday_buffaloes = 2 →
  monday_rhinos = 5 →
  monday_warthogs = 3 →
  ∃ (sunday_leopards : ℕ),
    total_animals = 
      saturday_lions + saturday_elephants + 
      sunday_buffaloes + sunday_leopards +
      monday_rhinos + monday_warthogs ∧
    sunday_leopards = 5 := by
  sorry

end NUMINAMATH_CALUDE_safari_leopards_l1209_120949


namespace NUMINAMATH_CALUDE_ratio_evaluation_l1209_120918

theorem ratio_evaluation : (2^3002 * 3^3005) / 6^3003 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l1209_120918


namespace NUMINAMATH_CALUDE_circle_equation_l1209_120900

/-- A circle with center on the line y = x passing through (-1, 1) and (1, 3) has the equation (x-1)^2 + (y-1)^2 = 4 -/
theorem circle_equation : ∀ (a : ℝ),
  (∀ (x y : ℝ), (x - a)^2 + (y - a)^2 = (a + 1)^2 + (a - 1)^2) →
  (∀ (x y : ℝ), (x - a)^2 + (y - a)^2 = (a - 1)^2 + (a - 3)^2) →
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1209_120900


namespace NUMINAMATH_CALUDE_compare_sqrt_l1209_120980

theorem compare_sqrt : -2 * Real.sqrt 11 > -3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_l1209_120980


namespace NUMINAMATH_CALUDE_largest_digit_change_l1209_120997

/-- The original incorrect sum -/
def incorrect_sum : ℕ := 1742

/-- The first addend in the original problem -/
def addend1 : ℕ := 789

/-- The second addend in the original problem -/
def addend2 : ℕ := 436

/-- The third addend in the original problem -/
def addend3 : ℕ := 527

/-- The corrected first addend after changing a digit -/
def corrected_addend1 : ℕ := 779

theorem largest_digit_change :
  (∃ (d : ℕ), d ≤ 9 ∧
    corrected_addend1 + addend2 + addend3 = incorrect_sum ∧
    d = (addend1 / 10) % 10 - (corrected_addend1 / 10) % 10 ∧
    ∀ (x y z : ℕ), x ≤ addend1 ∧ y ≤ addend2 ∧ z ≤ addend3 →
      x + y + z = incorrect_sum →
      (∃ (d' : ℕ), d' ≤ 9 ∧ d' = (addend1 / 10) % 10 - (x / 10) % 10) →
      d' ≤ d) :=
sorry

end NUMINAMATH_CALUDE_largest_digit_change_l1209_120997


namespace NUMINAMATH_CALUDE_total_athletes_l1209_120974

/-- Represents the number of athletes in each sport -/
structure Athletes :=
  (football : ℕ)
  (baseball : ℕ)
  (soccer : ℕ)
  (basketball : ℕ)

/-- The ratio of athletes in different sports -/
def ratio : Athletes := ⟨10, 7, 5, 4⟩

/-- The number of basketball players -/
def basketball_players : ℕ := 16

/-- Theorem stating the total number of athletes in the school -/
theorem total_athletes : 
  ∃ (k : ℕ), 
    k * ratio.football + 
    k * ratio.baseball + 
    k * ratio.soccer + 
    k * ratio.basketball = 104 ∧
    k * ratio.basketball = basketball_players :=
by sorry

end NUMINAMATH_CALUDE_total_athletes_l1209_120974


namespace NUMINAMATH_CALUDE_binomial_320_320_l1209_120968

theorem binomial_320_320 : Nat.choose 320 320 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_320_320_l1209_120968


namespace NUMINAMATH_CALUDE_garden_theorem_l1209_120995

def garden_problem (initial_plants : ℕ) (day1_eaten : ℕ) (day3_eaten : ℕ) : ℕ :=
  let remaining_day1 := initial_plants - day1_eaten
  let remaining_day2 := remaining_day1 / 2
  remaining_day2 - day3_eaten

theorem garden_theorem :
  garden_problem 30 20 1 = 4 := by
  sorry

#eval garden_problem 30 20 1

end NUMINAMATH_CALUDE_garden_theorem_l1209_120995


namespace NUMINAMATH_CALUDE_mixture_weight_l1209_120972

/-- Given a mixture of substances a and b in the ratio 9:11, where 26.1 kg of a is used,
    prove that the total weight of the mixture is 58 kg. -/
theorem mixture_weight (a b : ℝ) (h1 : a / b = 9 / 11) (h2 : a = 26.1) :
  a + b = 58 := by sorry

end NUMINAMATH_CALUDE_mixture_weight_l1209_120972


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l1209_120953

theorem power_of_two_divisibility (n : ℕ) : 
  (n ≥ 1 ∧ ∃ m : ℕ, m ≥ 1 ∧ (2^n - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l1209_120953


namespace NUMINAMATH_CALUDE_amount_distribution_l1209_120922

theorem amount_distribution (amount : ℕ) : 
  (∀ (x y : ℕ), x = amount / 14 ∧ y = amount / 18 → x = y + 80) →
  amount = 5040 := by
sorry

end NUMINAMATH_CALUDE_amount_distribution_l1209_120922


namespace NUMINAMATH_CALUDE_t_shirt_cost_l1209_120966

/-- Calculates the cost of each t-shirt given the total cost, number of t-shirts and pants, and cost of each pair of pants. -/
theorem t_shirt_cost (total_cost : ℕ) (num_tshirts num_pants pants_cost : ℕ) :
  total_cost = 1500 ∧ 
  num_tshirts = 5 ∧ 
  num_pants = 4 ∧ 
  pants_cost = 250 →
  (total_cost - num_pants * pants_cost) / num_tshirts = 100 := by
  sorry

end NUMINAMATH_CALUDE_t_shirt_cost_l1209_120966


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1209_120944

theorem partial_fraction_decomposition :
  let P : ℚ := 17/7
  let Q : ℚ := 4/7
  ∀ x : ℚ, x ≠ 10 → x ≠ -4 →
    (3*x + 4) / (x^2 - 6*x - 40) = P / (x - 10) + Q / (x + 4) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1209_120944


namespace NUMINAMATH_CALUDE_triangle_max_area_l1209_120910

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  b = 2 →
  (1 - Real.sqrt 3 * Real.cos B) / (Real.sqrt 3 * Real.sin B) = 1 / Real.tan C →
  ∃ (S : ℝ), S = Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) ∧
  ∀ (S' : ℝ), S' = Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2)) → S' ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1209_120910


namespace NUMINAMATH_CALUDE_triangle_external_angle_l1209_120979

theorem triangle_external_angle (a b c x : ℝ) : 
  a = 50 → b = 40 → c = 90 → a + b + c = 180 → 
  x + 45 = 180 → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_triangle_external_angle_l1209_120979


namespace NUMINAMATH_CALUDE_base_power_zero_l1209_120990

theorem base_power_zero (b : ℝ) (x y : ℝ) (h1 : 3^x * b^y = 59049) (h2 : x - y = 10) (h3 : x = 10) : y = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_power_zero_l1209_120990


namespace NUMINAMATH_CALUDE_trig_problem_l1209_120919

theorem trig_problem (x : ℝ) (h : Real.sin (x + π/6) = 1/3) :
  Real.sin (5*π/6 - x) - (Real.sin (π/3 - x))^2 = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l1209_120919


namespace NUMINAMATH_CALUDE_f_of_one_equals_twentyone_l1209_120952

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 - 3 * (x + 3) + 1

-- State the theorem
theorem f_of_one_equals_twentyone : f 1 = 21 := by sorry

end NUMINAMATH_CALUDE_f_of_one_equals_twentyone_l1209_120952


namespace NUMINAMATH_CALUDE_june_design_purple_tiles_l1209_120954

/-- Represents the number of tiles of each color in June's design -/
structure TileDesign where
  total : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  purple : Nat

/-- Theorem stating the number of purple tiles in June's design -/
theorem june_design_purple_tiles (d : TileDesign) : 
  d.total = 20 ∧ 
  d.yellow = 3 ∧ 
  d.blue = d.yellow + 1 ∧ 
  d.white = 7 → 
  d.purple = 6 := by
  sorry

#check june_design_purple_tiles

end NUMINAMATH_CALUDE_june_design_purple_tiles_l1209_120954


namespace NUMINAMATH_CALUDE_molecular_weight_AlBr3_10_moles_value_l1209_120992

/-- The molecular weight of 10 moles of AlBr3 -/
def molecular_weight_AlBr3_10_moles : ℝ :=
  let atomic_weight_Al : ℝ := 26.98
  let atomic_weight_Br : ℝ := 79.90
  let molecular_weight_AlBr3 : ℝ := atomic_weight_Al + 3 * atomic_weight_Br
  10 * molecular_weight_AlBr3

/-- Theorem stating that the molecular weight of 10 moles of AlBr3 is 2666.8 grams -/
theorem molecular_weight_AlBr3_10_moles_value :
  molecular_weight_AlBr3_10_moles = 2666.8 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_AlBr3_10_moles_value_l1209_120992


namespace NUMINAMATH_CALUDE_solution_set_l1209_120976

theorem solution_set (x : ℝ) : 
  33 * 32 ≤ x ∧ 
  Int.floor x + Int.ceil x = 5 → 
  2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l1209_120976


namespace NUMINAMATH_CALUDE_garage_wheels_count_l1209_120951

/-- The number of bikes that can be assembled -/
def num_bikes : ℕ := 9

/-- The number of wheels required for each bike -/
def wheels_per_bike : ℕ := 2

/-- The total number of wheels in the garage -/
def total_wheels : ℕ := num_bikes * wheels_per_bike

theorem garage_wheels_count : total_wheels = 18 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_count_l1209_120951


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1209_120960

/-- The polynomial to be divided -/
def P (x : ℝ) : ℝ := 9*x^3 - 5*x^2 + 8*x + 15

/-- The divisor polynomial -/
def D (x : ℝ) : ℝ := x - 3

/-- The quotient polynomial -/
def Q (x : ℝ) : ℝ := 9*x^2 + 22*x + 74

/-- The remainder -/
def R : ℝ := 237

theorem polynomial_division_theorem :
  ∀ x : ℝ, P x = D x * Q x + R :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1209_120960


namespace NUMINAMATH_CALUDE_equation_solution_l1209_120989

theorem equation_solution : ∃ x : ℝ, (2*x - 1)^2 - (1 - 3*x)^2 = 5*(1 - x)*(x + 1) ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1209_120989


namespace NUMINAMATH_CALUDE_max_value_condition_l1209_120935

-- Define the function f(x)
def f (a x : ℝ) : ℝ := (x - a)^2 * (x - 1)

-- State the theorem
theorem max_value_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a a) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_max_value_condition_l1209_120935


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l1209_120957

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height : ℝ) (circle_circumference : ℝ),
    rectangle_width = 9 →
    rectangle_height = 12 →
    circle_circumference = π * (rectangle_width^2 + rectangle_height^2).sqrt →
    circle_circumference = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l1209_120957


namespace NUMINAMATH_CALUDE_average_of_numbers_l1209_120994

theorem average_of_numbers : 
  let numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755, 755]
  (numbers.sum / numbers.length : ℚ) = 700 := by
sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1209_120994


namespace NUMINAMATH_CALUDE_probability_4H_before_3T_value_l1209_120988

/-- The probability of getting 4 heads before 3 tails in repeated fair coin flips -/
def probability_4H_before_3T : ℚ :=
  13 / 17

/-- Theorem stating that the probability of getting 4 heads before 3 tails
    in repeated fair coin flips is 13/17 -/
theorem probability_4H_before_3T_value :
  probability_4H_before_3T = 13 / 17 := by
  sorry

#eval Nat.gcd 13 17  -- To verify that 13 and 17 are coprime

end NUMINAMATH_CALUDE_probability_4H_before_3T_value_l1209_120988


namespace NUMINAMATH_CALUDE_hyperbola_slope_product_l1209_120911

/-- The product of slopes for a hyperbola -/
theorem hyperbola_slope_product (a b c : ℝ) (P Q : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  b^2 = a * c →
  ((P.1^2 / a^2) - (P.2^2 / b^2) = 1) →
  ((Q.1^2 / a^2) - (Q.2^2 / b^2) = 1) →
  let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let k_PQ := (Q.2 - P.2) / (Q.1 - P.1)
  let k_OM := M.2 / M.1
  k_PQ ≠ 0 →
  M.1 ≠ 0 →
  k_PQ * k_OM = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_slope_product_l1209_120911


namespace NUMINAMATH_CALUDE_min_value_theorem_max_value_theorem_l1209_120999

-- Part 1
theorem min_value_theorem (x : ℝ) (hx : x > 0) : 12 / x + 3 * x ≥ 12 := by
  sorry

-- Part 2
theorem max_value_theorem (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) : x * (1 - 3 * x) ≤ 1/12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_max_value_theorem_l1209_120999


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1209_120939

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → a ∈ Set.Ioi 3 ∪ Set.Iio (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1209_120939


namespace NUMINAMATH_CALUDE_fraction_addition_l1209_120906

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1209_120906


namespace NUMINAMATH_CALUDE_circle_line_distance_l1209_120963

theorem circle_line_distance (a : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 4*y = 0}
  let line := {(x, y) : ℝ × ℝ | x - y + a = 0}
  let center := (1, 2)
  let distance := |1 - 2 + a| / Real.sqrt 2
  (distance = Real.sqrt 2 / 2) → (a = 2 ∨ a = 0) := by
sorry

end NUMINAMATH_CALUDE_circle_line_distance_l1209_120963


namespace NUMINAMATH_CALUDE_book_pages_l1209_120983

/-- The number of pages Ceasar has already read -/
def pages_read : ℕ := 147

/-- The number of pages Ceasar has left to read -/
def pages_left : ℕ := 416

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_read + pages_left

/-- Theorem stating that the total number of pages in the book is 563 -/
theorem book_pages : total_pages = 563 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l1209_120983


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1209_120985

theorem polynomial_coefficient_sum (A B C D : ℚ) : 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 6) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 32 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1209_120985


namespace NUMINAMATH_CALUDE_fraction_equality_l1209_120912

theorem fraction_equality : 
  (2 + 4 - 8 + 16 + 32 - 64 + 128 - 256) / (4 + 8 - 16 + 32 + 64 - 128 + 256 - 512) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1209_120912


namespace NUMINAMATH_CALUDE_m_range_l1209_120904

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 - m) * x + 1 < (2 - m) * y + 1

-- Define the theorem
theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 
  1 < m ∧ m < 2 := by
  sorry


end NUMINAMATH_CALUDE_m_range_l1209_120904


namespace NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l1209_120936

theorem largest_of_eight_consecutive_integers (n : ℕ) : 
  (n > 0) →
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) = 3224) →
  (n + 7 = 406) := by
sorry

end NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l1209_120936


namespace NUMINAMATH_CALUDE_problem_solution_l1209_120941

theorem problem_solution : (2013^2 - 2013 - 1) / 2013 = 2012 - 1/2013 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1209_120941


namespace NUMINAMATH_CALUDE_quadratic_properties_l1209_120956

/-- Represents a quadratic function f(x) = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The quadratic function satisfying the given conditions -/
def f : QuadraticFunction := {
  a := -1,
  b := 2,
  c := 3,
  a_nonzero := by norm_num
}

/-- Evaluation of the quadratic function -/
def eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_properties (t : ℝ) :
  let f := f
  (∃ x y, x > 0 ∧ y > 0 ∧ eval f x = y ∧ ∀ x', eval f x' ≤ eval f x) ∧ 
  (∀ m n, 0 < m → m < 4 → eval f m = n → -5 < n ∧ n ≤ 4) ∧
  (eval f (-2) = t ∧ eval f 4 = t) ∧
  (∀ p, (∀ x, eval f x < 2*x + p) → p > 3) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_properties_l1209_120956


namespace NUMINAMATH_CALUDE_bank_exceeds_500_on_day_9_l1209_120915

def deposit_amount (day : ℕ) : ℕ :=
  if day ≤ 1 then 3
  else if day % 2 = 0 then 3 * deposit_amount (day - 2)
  else deposit_amount (day - 1)

def total_amount (day : ℕ) : ℕ :=
  List.sum (List.map deposit_amount (List.range (day + 1)))

theorem bank_exceeds_500_on_day_9 :
  total_amount 8 ≤ 500 ∧ total_amount 9 > 500 :=
sorry

end NUMINAMATH_CALUDE_bank_exceeds_500_on_day_9_l1209_120915


namespace NUMINAMATH_CALUDE_fifteen_points_max_planes_l1209_120996

def max_planes (n : ℕ) : ℕ := n.choose 3

theorem fifteen_points_max_planes :
  max_planes 15 = 455 :=
by sorry

end NUMINAMATH_CALUDE_fifteen_points_max_planes_l1209_120996


namespace NUMINAMATH_CALUDE_interval_for_quadratic_function_l1209_120925

/-- The function f(x) = -x^2 -/
def f (x : ℝ) : ℝ := -x^2

theorem interval_for_quadratic_function (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ≥ 2*a) ∧  -- minimum value condition
  (∀ x ∈ Set.Icc a b, f x ≤ 2*b) ∧  -- maximum value condition
  (∃ x ∈ Set.Icc a b, f x = 2*a) ∧  -- minimum value is achieved
  (∃ x ∈ Set.Icc a b, f x = 2*b) →  -- maximum value is achieved
  a = 1 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_interval_for_quadratic_function_l1209_120925


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l1209_120964

/-- An arithmetic sequence with first term a and common difference d -/
structure ArithmeticSequence where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a + (n - 1) * seq.d)

/-- Theorem: If the sum of the first 100 terms is 800 and the sum of the next 100 terms is 7500,
    then the first term of the arithmetic sequence is -24.835 -/
theorem first_term_of_arithmetic_sequence
  (seq : ArithmeticSequence)
  (h1 : sum_n_terms seq 100 = 800)
  (h2 : sum_n_terms seq 200 - sum_n_terms seq 100 = 7500) :
  seq.a = -4967 / 200 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l1209_120964


namespace NUMINAMATH_CALUDE_fifteen_percent_of_x_is_ninety_l1209_120917

theorem fifteen_percent_of_x_is_ninety (x : ℝ) : (15 / 100) * x = 90 → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_x_is_ninety_l1209_120917


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_inequality_l1209_120931

theorem sum_reciprocal_squares_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 3) :
  1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 ≤ 1/(a^2*b^2*c^2*d^2) := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_inequality_l1209_120931


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1209_120938

theorem sin_cos_identity (α c d : ℝ) (h : c > 0) (k : d > 0) 
  (eq : (Real.sin α)^6 / c + (Real.cos α)^6 / d = 1 / (c + d)) :
  (Real.sin α)^12 / c^5 + (Real.cos α)^12 / d^5 = 1 / (c + d)^5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1209_120938


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1209_120913

/-- For all a > 0 and a ≠ 1, the function f(x) = a^(x-3) - 3 passes through the point (3, -2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) - 3
  f 3 = -2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1209_120913


namespace NUMINAMATH_CALUDE_perimeter_ratio_not_integer_l1209_120977

theorem perimeter_ratio_not_integer (a k l : ℕ+) (h : a^2 = k * l) :
  ¬ ∃ (n : ℕ), (k + l : ℚ) / (2 * a) = n := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_not_integer_l1209_120977


namespace NUMINAMATH_CALUDE_condition_equivalence_l1209_120967

theorem condition_equivalence :
  (∀ x y : ℝ, x > y ↔ x^3 > y^3) ∧
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧
  (∃ x y : ℝ, x^2 > y^2 ∧ x ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_condition_equivalence_l1209_120967


namespace NUMINAMATH_CALUDE_exists_nonprime_between_primes_l1209_120947

/-- A number is prime if it's greater than 1 and its only positive divisors are 1 and itself. -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ n → d = 1 ∨ d = n

/-- There exists a natural number n such that n is not prime, but both n-2 and n+2 are prime. -/
theorem exists_nonprime_between_primes : ∃ n : ℕ, 
  ¬ isPrime n ∧ isPrime (n - 2) ∧ isPrime (n + 2) :=
sorry

end NUMINAMATH_CALUDE_exists_nonprime_between_primes_l1209_120947


namespace NUMINAMATH_CALUDE_circle_equation_from_center_and_point_specific_circle_equation_l1209_120908

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def lies_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of a circle given its center and a point on the circle -/
theorem circle_equation_from_center_and_point 
  (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  ∃ (c : Circle), 
    c.center = center ∧ 
    lies_on_circle c point ∧
    ∀ (x y : ℝ), lies_on_circle c (x, y) ↔ (x - center.1)^2 + (y - center.2)^2 = c.radius^2 := by
  sorry

/-- The specific circle equation for the given problem -/
theorem specific_circle_equation : 
  ∃ (c : Circle), 
    c.center = (0, 4) ∧ 
    lies_on_circle c (3, 0) ∧
    ∀ (x y : ℝ), lies_on_circle c (x, y) ↔ x^2 + (y - 4)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_from_center_and_point_specific_circle_equation_l1209_120908


namespace NUMINAMATH_CALUDE_women_work_nine_hours_l1209_120969

/-- Represents the work scenario with men and women -/
structure WorkScenario where
  men_count : ℕ
  men_days : ℕ
  men_hours_per_day : ℕ
  women_count : ℕ
  women_days : ℕ
  women_efficiency : Rat

/-- Calculates the number of hours women work per day -/
def women_hours_per_day (ws : WorkScenario) : Rat :=
  (ws.men_count * ws.men_days * ws.men_hours_per_day : Rat) /
  (ws.women_count * ws.women_days * ws.women_efficiency)

/-- The given work scenario -/
def given_scenario : WorkScenario :=
  { men_count := 15
  , men_days := 21
  , men_hours_per_day := 8
  , women_count := 21
  , women_days := 20
  , women_efficiency := 2/3 }

theorem women_work_nine_hours : women_hours_per_day given_scenario = 9 := by
  sorry

end NUMINAMATH_CALUDE_women_work_nine_hours_l1209_120969


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1209_120943

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem stating that "q > 1" is neither necessary nor sufficient for a geometric sequence to be monotonically increasing -/
theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(GeometricSequence a q ∧ (q > 1 ↔ MonotonicallyIncreasing a)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1209_120943


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1209_120932

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1209_120932


namespace NUMINAMATH_CALUDE_smallest_pencil_collection_l1209_120962

theorem smallest_pencil_collection (P : ℕ) : 
  P > 2 ∧ 
  P % 5 = 2 ∧ 
  P % 9 = 2 ∧ 
  P % 11 = 2 ∧ 
  (∀ Q : ℕ, Q > 2 ∧ Q % 5 = 2 ∧ Q % 9 = 2 ∧ Q % 11 = 2 → P ≤ Q) →
  P = 497 := by
sorry

end NUMINAMATH_CALUDE_smallest_pencil_collection_l1209_120962


namespace NUMINAMATH_CALUDE_sqrt_xy_eq_three_halves_l1209_120933

theorem sqrt_xy_eq_three_halves (x y : ℝ) (h : |2*x + 1| + Real.sqrt (9 + 2*y) = 0) :
  Real.sqrt (x * y) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_xy_eq_three_halves_l1209_120933


namespace NUMINAMATH_CALUDE_solve_for_P_l1209_120946

theorem solve_for_P : ∃ P : ℝ, (P^4)^(1/3) = 9 * 81^(1/9) → P = 3^(11/6) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_P_l1209_120946


namespace NUMINAMATH_CALUDE_equation_solution_l1209_120902

theorem equation_solution (x : ℝ) : 
  (1 / x + 1 / (x + 2) - 1 / (x + 4) - 1 / (x + 6) - 1 / (x + 8) - 1 / (x + 10) + 1 / (x + 12) + 1 / (x + 14) = 0) ↔ 
  (x = -7 ∨ x = -7 + Real.sqrt (19 + 6 * Real.sqrt 5) ∨ 
   x = -7 - Real.sqrt (19 + 6 * Real.sqrt 5) ∨ 
   x = -7 + Real.sqrt (19 - 6 * Real.sqrt 5) ∨ 
   x = -7 - Real.sqrt (19 - 6 * Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1209_120902


namespace NUMINAMATH_CALUDE_corn_yield_ratio_l1209_120970

/-- Represents the corn yield ratio problem --/
theorem corn_yield_ratio :
  let johnson_hectares : ℕ := 1
  let johnson_yield_per_2months : ℕ := 80
  let neighbor_hectares : ℕ := 2
  let total_months : ℕ := 6
  let total_yield : ℕ := 1200
  let neighbor_yield_ratio : ℚ := 
    (total_yield - johnson_yield_per_2months * (total_months / 2) * johnson_hectares) /
    (johnson_yield_per_2months * (total_months / 2) * neighbor_hectares)
  neighbor_yield_ratio = 2
  := by sorry

end NUMINAMATH_CALUDE_corn_yield_ratio_l1209_120970


namespace NUMINAMATH_CALUDE_evans_county_population_l1209_120984

theorem evans_county_population (num_cities : ℕ) (lower_bound upper_bound : ℝ) :
  num_cities = 25 →
  lower_bound = 3200 →
  upper_bound = 3600 →
  (num_cities : ℝ) * ((lower_bound + upper_bound) / 2) = 85000 := by
  sorry

end NUMINAMATH_CALUDE_evans_county_population_l1209_120984


namespace NUMINAMATH_CALUDE_samanthas_last_name_has_seven_letters_l1209_120973

/-- The length of Jamie's last name -/
def jamies_last_name_length : ℕ := 4

/-- The length of Bobbie's last name -/
def bobbies_last_name_length : ℕ := 
  2 * jamies_last_name_length + 2

/-- The length of Samantha's last name -/
def samanthas_last_name_length : ℕ := 
  bobbies_last_name_length - 3

/-- Theorem stating that Samantha's last name has 7 letters -/
theorem samanthas_last_name_has_seven_letters : 
  samanthas_last_name_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_samanthas_last_name_has_seven_letters_l1209_120973


namespace NUMINAMATH_CALUDE_train_passing_time_l1209_120958

/-- The time taken for a slower train to pass the driver of a faster train -/
theorem train_passing_time (length : ℝ) (speed_fast speed_slow : ℝ) :
  length = 500 →
  speed_fast = 45 →
  speed_slow = 30 →
  let relative_speed := speed_fast + speed_slow
  let relative_speed_ms := relative_speed * 1000 / 3600
  let time := length / relative_speed_ms
  ∃ ε > 0, |time - 24| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l1209_120958


namespace NUMINAMATH_CALUDE_counterfeit_banknote_theorem_l1209_120982

/-- Represents a banknote with a natural number denomination -/
structure Banknote where
  denomination : ℕ

/-- Represents a collection of banknotes -/
def BanknoteCollection := List Banknote

/-- The detector's reading of the total sum -/
def detectorSum (collection : BanknoteCollection) : ℕ := sorry

/-- The actual sum of genuine banknotes -/
def genuineSum (collection : BanknoteCollection) : ℕ := sorry

/-- Predicate to check if a collection has pairwise different denominations -/
def hasPairwiseDifferentDenominations (collection : BanknoteCollection) : Prop := sorry

/-- Predicate to check if a collection has exactly one counterfeit banknote -/
def hasExactlyOneCounterfeit (collection : BanknoteCollection) : Prop := sorry

/-- The denomination of the counterfeit banknote -/
def counterfeitDenomination (collection : BanknoteCollection) : ℕ := sorry

theorem counterfeit_banknote_theorem (collection : BanknoteCollection) 
  (h1 : hasPairwiseDifferentDenominations collection)
  (h2 : hasExactlyOneCounterfeit collection) :
  detectorSum collection - genuineSum collection = counterfeitDenomination collection := by
  sorry

end NUMINAMATH_CALUDE_counterfeit_banknote_theorem_l1209_120982


namespace NUMINAMATH_CALUDE_hyperbola_x_axis_l1209_120950

/-- Given k > 1, the equation (1-k)x^2 + y^2 = k^2 - 1 represents a hyperbola with its real axis along the x-axis -/
theorem hyperbola_x_axis (k : ℝ) (h : k > 1) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), ((1-k)*x^2 + y^2 = k^2 - 1) ↔ (x^2/a^2 - y^2/b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_x_axis_l1209_120950


namespace NUMINAMATH_CALUDE_maddy_graduation_time_l1209_120959

/-- The number of semesters Maddy needs to be in college -/
def semesters_needed (total_credits : ℕ) (credits_per_class : ℕ) (classes_per_semester : ℕ) : ℕ :=
  total_credits / (credits_per_class * classes_per_semester)

/-- Proof that Maddy needs 8 semesters to graduate -/
theorem maddy_graduation_time :
  semesters_needed 120 3 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_maddy_graduation_time_l1209_120959


namespace NUMINAMATH_CALUDE_subtraction_problem_l1209_120907

theorem subtraction_problem (x : ℤ) : x - 46 = 15 → x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1209_120907


namespace NUMINAMATH_CALUDE_special_representation_theorem_l1209_120930

theorem special_representation_theorem (n : ℕ) (h1 : n ≥ 2) :
  (∃ k d : ℕ, k > 1 ∧ k ∣ n ∧ d ∣ n ∧
   (∀ m : ℕ, m > 1 → m ∣ n → m ≥ k) ∧
   n = k^2 + d^2) ↔ n = 8 ∨ n = 20 := by
sorry

end NUMINAMATH_CALUDE_special_representation_theorem_l1209_120930


namespace NUMINAMATH_CALUDE_women_at_dance_event_l1209_120978

/-- Represents a dance event with men and women -/
structure DanceEvent where
  men : ℕ
  women : ℕ
  men_dances : ℕ
  women_dances : ℕ

/-- Calculates the total number of dance pairs in the event -/
def total_dance_pairs (event : DanceEvent) : ℕ :=
  event.men * event.men_dances

/-- Theorem: Given the conditions of the dance event, prove that 24 women attended -/
theorem women_at_dance_event (event : DanceEvent) 
  (h1 : event.men_dances = 4)
  (h2 : event.women_dances = 3)
  (h3 : event.men = 18) :
  event.women = 24 := by
  sorry

#check women_at_dance_event

end NUMINAMATH_CALUDE_women_at_dance_event_l1209_120978


namespace NUMINAMATH_CALUDE_max_difference_of_constrained_integers_l1209_120975

theorem max_difference_of_constrained_integers : 
  ∃ (P Q : ℤ), 
    (∃ (x : ℤ), x^2 ≤ 729 ∧ 729 ≤ -x^3 ∧ (x = P ∨ x = Q)) ∧
    (∀ (R S : ℤ), (∃ (y : ℤ), y^2 ≤ 729 ∧ 729 ≤ -y^3 ∧ (y = R ∨ y = S)) → 
      10 * (P - Q) ≥ 10 * (R - S)) ∧
    10 * (P - Q) = 180 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_of_constrained_integers_l1209_120975


namespace NUMINAMATH_CALUDE_t_shaped_figure_perimeter_l1209_120987

/-- A geometric figure composed of four identical squares in a T shape -/
structure TShapedFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 144 cm² -/
  area_eq : 4 * side_length ^ 2 = 144

/-- The perimeter of a T-shaped figure -/
def perimeter (f : TShapedFigure) : ℝ :=
  5 * f.side_length

theorem t_shaped_figure_perimeter (f : TShapedFigure) : perimeter f = 30 :=
sorry

end NUMINAMATH_CALUDE_t_shaped_figure_perimeter_l1209_120987


namespace NUMINAMATH_CALUDE_investment_percentage_l1209_120986

/-- Given two investments with a total of $2000, where $600 is invested at 8%,
    and the annual income from the first investment exceeds the second by $92,
    prove that the percentage of the first investment is 10%. -/
theorem investment_percentage : 
  ∀ (total_investment first_investment_amount first_investment_rate : ℝ),
  total_investment = 2000 →
  first_investment_amount = 1400 →
  first_investment_rate * first_investment_amount - 0.08 * 600 = 92 →
  first_investment_rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_l1209_120986


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1209_120905

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 9 / b) ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1209_120905


namespace NUMINAMATH_CALUDE_apple_delivery_problem_l1209_120928

theorem apple_delivery_problem (first_grade_value second_grade_value : ℝ)
  (price_difference : ℝ) (quantity_difference : ℝ) :
  first_grade_value = 228 →
  second_grade_value = 180 →
  price_difference = 0.9 →
  quantity_difference = 5 →
  ∃ x : ℝ,
    x > 0 ∧
    x + quantity_difference > 0 ∧
    (first_grade_value / x - price_difference) * (2 * x + quantity_difference) =
      first_grade_value + second_grade_value ∧
    2 * x + quantity_difference = 85 :=
by sorry

end NUMINAMATH_CALUDE_apple_delivery_problem_l1209_120928


namespace NUMINAMATH_CALUDE_unique_solution_divisor_system_l1209_120955

theorem unique_solution_divisor_system :
  ∀ a b : ℕ+,
  (∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℕ+),
    a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧ a₇ < a₈ ∧ a₈ < a₉ ∧ a₉ < a₁₀ ∧ a₁₀ < a₁₁ ∧
    a₁ ∣ a ∧ a₂ ∣ a ∧ a₃ ∣ a ∧ a₄ ∣ a ∧ a₅ ∣ a ∧ a₆ ∣ a ∧ a₇ ∣ a ∧ a₈ ∣ a ∧ a₉ ∣ a ∧ a₁₀ ∣ a ∧ a₁₁ ∣ a) →
  (∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ : ℕ+),
    b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧ b₅ < b₆ ∧ b₆ < b₇ ∧ b₇ < b₈ ∧ b₈ < b₉ ∧ b₉ < b₁₀ ∧ b₁₀ < b₁₁ ∧
    b₁ ∣ b ∧ b₂ ∣ b ∧ b₃ ∣ b ∧ b₄ ∣ b ∧ b₅ ∣ b ∧ b₆ ∣ b ∧ b₇ ∣ b ∧ b₈ ∣ b ∧ b₉ ∣ b ∧ b₁₀ ∣ b ∧ b₁₁ ∣ b) →
  a₁₀ + b₁₀ = a →
  a₁₁ + b₁₁ = b →
  a = 1024 ∧ b = 2048 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_divisor_system_l1209_120955


namespace NUMINAMATH_CALUDE_number_problem_l1209_120926

theorem number_problem : 
  ∃ (number : ℝ), number * 11 = 165 ∧ number = 15 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l1209_120926


namespace NUMINAMATH_CALUDE_rain_duration_problem_l1209_120901

theorem rain_duration_problem (x : ℝ) : 
  let first_day := 10
  let second_day := first_day + x
  let third_day := 2 * second_day
  first_day + second_day + third_day = 46 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_rain_duration_problem_l1209_120901


namespace NUMINAMATH_CALUDE_y_to_x_value_l1209_120929

theorem y_to_x_value (x y : ℝ) (h : (x - 2)^2 + |y + 1/3| = 0) : y^x = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_y_to_x_value_l1209_120929


namespace NUMINAMATH_CALUDE_simplify_sqrt_450_l1209_120927

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_450_l1209_120927


namespace NUMINAMATH_CALUDE_prime_ratio_natural_numbers_l1209_120998

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_ratio_natural_numbers :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
    (is_prime ((x * y^3) / (x + y)) ↔ x = 14 ∧ y = 2) :=
by sorry


end NUMINAMATH_CALUDE_prime_ratio_natural_numbers_l1209_120998


namespace NUMINAMATH_CALUDE_fraction_sum_division_specific_fraction_sum_division_l1209_120924

theorem fraction_sum_division (a b c d : ℚ) :
  (a / b + c / d) / 4 = (a * d + b * c) / (4 * b * d) :=
by sorry

theorem specific_fraction_sum_division :
  (2 / 5 + 1 / 3) / 4 = 11 / 60 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_division_specific_fraction_sum_division_l1209_120924


namespace NUMINAMATH_CALUDE_base_height_proof_l1209_120916

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_valid_inches : inches < 12

/-- Converts a Height to feet -/
def heightToFeet (h : Height) : ℚ :=
  h.feet + h.inches / 12

theorem base_height_proof (sculpture_height : Height) 
    (h_sculpture_height : sculpture_height = ⟨2, 10, by norm_num⟩) 
    (combined_height : ℚ) 
    (h_combined_height : combined_height = 3) :
  let base_height := combined_height - heightToFeet sculpture_height
  base_height * 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_height_proof_l1209_120916


namespace NUMINAMATH_CALUDE_rectangle_area_l1209_120971

/-- A rectangle with a diagonal of 17 cm and a perimeter of 46 cm has an area of 120 cm². -/
theorem rectangle_area (l w : ℝ) : 
  l > 0 → w > 0 → l^2 + w^2 = 17^2 → 2*l + 2*w = 46 → l * w = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1209_120971
