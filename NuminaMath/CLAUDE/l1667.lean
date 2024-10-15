import Mathlib

namespace NUMINAMATH_CALUDE_compute_custom_op_l1667_166796

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem compute_custom_op : custom_op (custom_op 8 6) 2 = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_compute_custom_op_l1667_166796


namespace NUMINAMATH_CALUDE_farey_consecutive_fraction_l1667_166735

/-- Represents a fraction as a pair of integers -/
structure Fraction where
  numerator : ℤ
  denominator : ℤ
  den_nonzero : denominator ≠ 0

/-- Checks if three fractions are consecutive in a Farey sequence -/
def consecutive_in_farey (f1 f2 f3 : Fraction) : Prop :=
  f1.numerator * f2.denominator - f1.denominator * f2.numerator = 1 ∧
  f3.numerator * f2.denominator - f3.denominator * f2.numerator = 1

/-- The main theorem about three consecutive fractions in a Farey sequence -/
theorem farey_consecutive_fraction (a b c d x y : ℤ) 
  (hb : b ≠ 0) (hd : d ≠ 0) (hy : y ≠ 0)
  (h_order : (a : ℚ) / b < x / y ∧ x / y < c / d)
  (h_consecutive : consecutive_in_farey 
    ⟨a, b, hb⟩ 
    ⟨x, y, hy⟩ 
    ⟨c, d, hd⟩) :
  (x : ℚ) / y = (a + c) / (b + d) := by
  sorry

end NUMINAMATH_CALUDE_farey_consecutive_fraction_l1667_166735


namespace NUMINAMATH_CALUDE_horse_value_is_240_l1667_166736

/-- Represents the payment terms and actual service of a soldier --/
structure SoldierPayment where
  total_payment : ℕ  -- Total payment promised for full service in florins
  service_period : ℕ  -- Full service period in months
  actual_service : ℕ  -- Actual service period in months
  cash_payment : ℕ   -- Cash payment given at the end of actual service

/-- Calculates the value of a horse given to a soldier as part of payment --/
def horse_value (p : SoldierPayment) : ℕ :=
  p.total_payment - (p.total_payment / p.service_period * p.actual_service + p.cash_payment)

/-- Theorem stating the value of the horse in the given problem --/
theorem horse_value_is_240 (p : SoldierPayment) 
  (h1 : p.total_payment = 300)
  (h2 : p.service_period = 36)
  (h3 : p.actual_service = 17)
  (h4 : p.cash_payment = 15) :
  horse_value p = 240 := by
  sorry

end NUMINAMATH_CALUDE_horse_value_is_240_l1667_166736


namespace NUMINAMATH_CALUDE_first_part_interest_rate_l1667_166725

/-- Proves that given the specified conditions, the interest rate of the first part is 3% -/
theorem first_part_interest_rate 
  (total_investment : ℝ) 
  (first_part : ℝ) 
  (second_part_rate : ℝ) 
  (total_interest : ℝ) : 
  total_investment = 4000 →
  first_part = 2800 →
  second_part_rate = 0.05 →
  total_interest = 144 →
  (first_part * (3 / 100) + (total_investment - first_part) * second_part_rate = total_interest) :=
by
  sorry

#check first_part_interest_rate

end NUMINAMATH_CALUDE_first_part_interest_rate_l1667_166725


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l1667_166744

/-- The distance from the origin to a line passing through a given point with a given direction vector -/
theorem distance_origin_to_line (P : ℝ × ℝ) (n : ℝ × ℝ) : 
  P.1 = 2 ∧ P.2 = 0 ∧ n.1 = 1 ∧ n.2 = -1 →
  Real.sqrt ((P.1^2 + P.2^2) * (n.1^2 + n.2^2) - (P.1*n.1 + P.2*n.2)^2) / Real.sqrt (n.1^2 + n.2^2) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l1667_166744


namespace NUMINAMATH_CALUDE_gilbert_herb_count_l1667_166787

/-- Represents the number of herb plants Gilbert has at different stages of spring -/
structure HerbGarden where
  initial_basil : ℕ
  initial_parsley : ℕ
  initial_mint : ℕ
  extra_basil : ℕ
  eaten_mint : ℕ

/-- Calculates the final number of herb plants in Gilbert's garden -/
def final_herb_count (garden : HerbGarden) : ℕ :=
  garden.initial_basil + garden.initial_parsley + garden.initial_mint + garden.extra_basil - garden.eaten_mint

/-- Theorem stating that Gilbert had 5 herb plants at the end of spring -/
theorem gilbert_herb_count :
  ∀ (garden : HerbGarden),
    garden.initial_basil = 3 →
    garden.initial_parsley = 1 →
    garden.initial_mint = 2 →
    garden.extra_basil = 1 →
    garden.eaten_mint = 2 →
    final_herb_count garden = 5 := by
  sorry


end NUMINAMATH_CALUDE_gilbert_herb_count_l1667_166787


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1667_166728

theorem trigonometric_equation_solution (x : ℝ) : 
  (∃ (n : ℤ), x = Real.pi / 2 * (2 * ↑n + 1)) ∨ 
  (∃ (k : ℤ), x = Real.pi / 18 * (4 * ↑k + 1)) ↔ 
  Real.sin (3 * x) + Real.sin (5 * x) = 2 * (Real.cos (2 * x))^2 - 2 * (Real.sin (3 * x))^2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1667_166728


namespace NUMINAMATH_CALUDE_odd_sequence_sum_l1667_166747

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n / 2 * (a₁ + aₙ)

theorem odd_sequence_sum :
  ∃ (n : ℕ), 
    let a₁ := 1
    let aₙ := 79
    let sum := arithmetic_sum a₁ aₙ n
    n > 0 ∧ aₙ = a₁ + 2 * (n - 1) ∧ 3 * sum = 4800 := by
  sorry

end NUMINAMATH_CALUDE_odd_sequence_sum_l1667_166747


namespace NUMINAMATH_CALUDE_intersection_M_N_l1667_166795

def M : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - x^2)}
def N : Set ℝ := {x | ∃ y, y = x^2 - 1}

theorem intersection_M_N : M ∩ N = Set.Icc (-1) (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1667_166795


namespace NUMINAMATH_CALUDE_newberg_total_landed_l1667_166776

/-- Represents the passenger data for an airport -/
structure AirportData where
  onTime : ℕ
  late : ℕ
  cancelled : ℕ

/-- Calculates the total number of landed passengers, excluding cancelled flights -/
def totalLanded (data : AirportData) : ℕ :=
  data.onTime + data.late

/-- Theorem: The total number of passengers who landed in Newberg last year is 28,690 -/
theorem newberg_total_landed :
  let airportA : AirportData := ⟨16507, 256, 198⟩
  let airportB : AirportData := ⟨11792, 135, 151⟩
  totalLanded airportA + totalLanded airportB = 28690 := by
  sorry


end NUMINAMATH_CALUDE_newberg_total_landed_l1667_166776


namespace NUMINAMATH_CALUDE_even_odd_function_sum_l1667_166755

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem even_odd_function_sum (f g : ℝ → ℝ) 
  (hf : is_even_function f) (hg : is_odd_function g) 
  (h : ∀ x, f x + g x = Real.exp x) : 
  ∀ x, g x = Real.exp x - Real.exp (-x) := by
  sorry

end NUMINAMATH_CALUDE_even_odd_function_sum_l1667_166755


namespace NUMINAMATH_CALUDE_toothpicks_10th_stage_l1667_166770

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 5
  else toothpicks (n - 1) + 3 * n

/-- The theorem stating that the 10th stage has 167 toothpicks -/
theorem toothpicks_10th_stage : toothpicks 10 = 167 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_10th_stage_l1667_166770


namespace NUMINAMATH_CALUDE_two_number_problem_l1667_166782

def is_solution (x y : ℕ) : Prop :=
  (x + y = 667) ∧ 
  (Nat.lcm x y / Nat.gcd x y = 120)

theorem two_number_problem :
  ∀ x y : ℕ, is_solution x y → 
    ((x = 115 ∧ y = 552) ∨ (x = 552 ∧ y = 115) ∨ 
     (x = 232 ∧ y = 435) ∨ (x = 435 ∧ y = 232)) :=
by sorry

end NUMINAMATH_CALUDE_two_number_problem_l1667_166782


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l1667_166761

/-- Represents a hyperbola with vertices on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a line with slope k passing through (3,0) -/
structure Line where
  k : ℝ

/-- Defines when a line intersects a hyperbola at exactly one point -/
def intersects_at_one_point (h : Hyperbola) (l : Line) : Prop :=
  ∃! x y : ℝ, x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ y = l.k * (x - 3)

/-- The main theorem to be proved -/
theorem hyperbola_intersection_theorem (h : Hyperbola) (l : Line) :
  h.a = 4 ∧ h.b = 3 →
  intersects_at_one_point h l ↔ 
    l.k = 3/4 ∨ l.k = -3/4 ∨ l.k = 3*Real.sqrt 7/7 ∨ l.k = -3*Real.sqrt 7/7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l1667_166761


namespace NUMINAMATH_CALUDE_room_length_proof_l1667_166794

/-- Given the cost of carpeting, carpet width, cost per meter, and room breadth, 
    prove the length of the room. -/
theorem room_length_proof 
  (total_cost : ℝ) 
  (carpet_width : ℝ) 
  (cost_per_meter : ℝ) 
  (room_breadth : ℝ) 
  (h1 : total_cost = 36)
  (h2 : carpet_width = 0.75)
  (h3 : cost_per_meter = 0.30)
  (h4 : room_breadth = 6) :
  ∃ (room_length : ℝ), room_length = 15 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l1667_166794


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l1667_166764

/-- Represents the ratio of ingredients in a recipe -/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio -/
def original_ratio : Ratio := ⟨7, 2, 1⟩

/-- The new recipe ratio -/
def new_ratio : Ratio :=
  let flour_water_doubled := original_ratio.flour / original_ratio.water * 2
  let flour_sugar_halved := original_ratio.flour / original_ratio.sugar / 2
  ⟨flour_water_doubled * original_ratio.water, original_ratio.water, flour_sugar_halved⟩

/-- The amount of water in the new recipe (in cups) -/
def new_water_amount : ℚ := 2

theorem sugar_amount_in_new_recipe :
  (new_water_amount * new_ratio.sugar / new_ratio.water) = 1 :=
sorry

end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l1667_166764


namespace NUMINAMATH_CALUDE_natalia_albums_count_l1667_166754

/-- Represents the number of items in Natalia's library --/
structure LibraryItems where
  novels : Nat
  comics : Nat
  documentaries : Nat
  albums : Nat

/-- Represents the crate information --/
structure CrateInfo where
  capacity : Nat
  count : Nat

/-- Theorem: Given the library items and crate information, prove that Natalia has 209 albums --/
theorem natalia_albums_count
  (items : LibraryItems)
  (crates : CrateInfo)
  (h1 : items.novels = 145)
  (h2 : items.comics = 271)
  (h3 : items.documentaries = 419)
  (h4 : crates.capacity = 9)
  (h5 : crates.count = 116)
  (h6 : items.novels + items.comics + items.documentaries + items.albums = crates.capacity * crates.count) :
  items.albums = 209 := by
  sorry


end NUMINAMATH_CALUDE_natalia_albums_count_l1667_166754


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1667_166757

theorem min_value_of_expression (x : ℝ) (h : x > 0) :
  3 * x + 1 / (x^3) ≥ 4 ∧
  (3 * x + 1 / (x^3) = 4 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1667_166757


namespace NUMINAMATH_CALUDE_fraction_equality_l1667_166784

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + 2 * y) / (2 * x - 5 * y) = 3) : 
  (x + 3 * y) / (3 * x - y) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1667_166784


namespace NUMINAMATH_CALUDE_marks_vaccine_wait_l1667_166771

/-- Theorem: Mark's wait for first vaccine appointment
Given:
- The total waiting time is 38 days
- There's a 20-day wait between appointments
- There's a 14-day wait for full effectiveness after the second appointment
Prove: The wait for the first appointment is 4 days
-/
theorem marks_vaccine_wait (total_wait : ℕ) (between_appointments : ℕ) (full_effectiveness : ℕ) :
  total_wait = 38 →
  between_appointments = 20 →
  full_effectiveness = 14 →
  total_wait = between_appointments + full_effectiveness + 4 :=
by sorry

end NUMINAMATH_CALUDE_marks_vaccine_wait_l1667_166771


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1667_166767

theorem hyperbola_asymptote (m : ℝ) :
  (∀ x y : ℝ, x^2 / |m| - y^2 / (|m| + 3) = 1) →
  (2 * Real.sqrt 5 = Real.sqrt (2 * |m| + 3)) →
  (∃ k : ℝ, k = 2 ∧ ∀ x : ℝ, k * x = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1667_166767


namespace NUMINAMATH_CALUDE_darnel_distance_difference_l1667_166711

theorem darnel_distance_difference :
  let sprint_distance : ℚ := 875 / 1000
  let jog_distance : ℚ := 75 / 100
  sprint_distance - jog_distance = 125 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_darnel_distance_difference_l1667_166711


namespace NUMINAMATH_CALUDE_song_duration_l1667_166783

theorem song_duration (initial_songs : ℕ) (added_songs : ℕ) (total_time : ℕ) :
  initial_songs = 25 →
  added_songs = 10 →
  total_time = 105 →
  (initial_songs + added_songs) * (total_time / (initial_songs + added_songs)) = total_time →
  total_time / (initial_songs + added_songs) = 3 :=
by sorry

end NUMINAMATH_CALUDE_song_duration_l1667_166783


namespace NUMINAMATH_CALUDE_rectangle_formations_6_7_l1667_166733

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of ways to form a rectangle given h horizontal lines and v vertical lines -/
def rectangle_formations (h v : ℕ) : ℕ := choose_2 h * choose_2 v

/-- Theorem stating that with 6 horizontal and 7 vertical lines, there are 315 ways to form a rectangle -/
theorem rectangle_formations_6_7 : rectangle_formations 6 7 = 315 := by sorry

end NUMINAMATH_CALUDE_rectangle_formations_6_7_l1667_166733


namespace NUMINAMATH_CALUDE_roots_of_equation_l1667_166723

theorem roots_of_equation (x : ℝ) : 
  (x - 1) * (x - 2) = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1667_166723


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l1667_166718

theorem multiplication_addition_equality : 3.5 * 0.3 + 1.2 * 0.4 = 1.53 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l1667_166718


namespace NUMINAMATH_CALUDE_larger_circle_radius_l1667_166749

theorem larger_circle_radius (r : ℝ) (h1 : r = 2) : ∃ R : ℝ,
  (∀ i j : Fin 4, i ≠ j → (∃ c₁ c₂ : ℝ × ℝ, 
    dist c₁ c₂ = 2 * r ∧ 
    (∀ x : ℝ × ℝ, dist x c₁ ≤ r ∨ dist x c₂ ≤ r))) →
  (∃ C : ℝ × ℝ, ∀ i : Fin 4, ∃ c : ℝ × ℝ, 
    dist C c = R - r ∧ 
    (∀ x : ℝ × ℝ, dist x c ≤ r → dist x C ≤ R)) →
  R = 4 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l1667_166749


namespace NUMINAMATH_CALUDE_two_point_eight_million_scientific_notation_l1667_166763

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem two_point_eight_million_scientific_notation :
  toScientificNotation 2800000 = ScientificNotation.mk 2.8 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_two_point_eight_million_scientific_notation_l1667_166763


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1667_166791

/-- Triangle type with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  valid : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b

/-- Theorem statement for triangle inequality -/
theorem triangle_area_inequality (t : Triangle) :
  t.S ≤ (t.a^2 + t.b^2 + t.c^2) / (4 * Real.sqrt 3) ∧
  (t.S = (t.a^2 + t.b^2 + t.c^2) / (4 * Real.sqrt 3) ↔ t.a = t.b ∧ t.b = t.c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l1667_166791


namespace NUMINAMATH_CALUDE_cos_A_eq_11_15_l1667_166773

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_A_eq_C (q : Quadrilateral) : Prop :=
  sorry

def side_AB_eq_150 (q : Quadrilateral) : Prop :=
  sorry

def side_CD_eq_150 (q : Quadrilateral) : Prop :=
  sorry

def side_AD_ne_BC (q : Quadrilateral) : Prop :=
  sorry

def perimeter_eq_520 (q : Quadrilateral) : Prop :=
  sorry

-- Define cos A
def cos_A (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem cos_A_eq_11_15 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_angle : angle_A_eq_C q)
  (h_AB : side_AB_eq_150 q)
  (h_CD : side_CD_eq_150 q)
  (h_AD_ne_BC : side_AD_ne_BC q)
  (h_perimeter : perimeter_eq_520 q) :
  cos_A q = 11/15 := by sorry

end NUMINAMATH_CALUDE_cos_A_eq_11_15_l1667_166773


namespace NUMINAMATH_CALUDE_mod_equivalence_solution_l1667_166727

theorem mod_equivalence_solution : ∃ (n : ℕ), n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_solution_l1667_166727


namespace NUMINAMATH_CALUDE_archies_backyard_sod_l1667_166720

/-- The area of sod needed for Archie's backyard -/
def sod_area (backyard_length backyard_width shed_length shed_width : ℕ) : ℕ :=
  backyard_length * backyard_width - shed_length * shed_width

/-- Theorem stating the correct amount of sod needed for Archie's backyard -/
theorem archies_backyard_sod : sod_area 20 13 3 5 = 245 := by
  sorry

end NUMINAMATH_CALUDE_archies_backyard_sod_l1667_166720


namespace NUMINAMATH_CALUDE_order_of_trig_functions_l1667_166719

theorem order_of_trig_functions : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_trig_functions_l1667_166719


namespace NUMINAMATH_CALUDE_no_solutions_sqrt_1452_l1667_166786

theorem no_solutions_sqrt_1452 : 
  ¬ ∃ (x y : ℕ), 0 < x ∧ x < y ∧ Real.sqrt 1452 = Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_sqrt_1452_l1667_166786


namespace NUMINAMATH_CALUDE_complex_cube_sum_l1667_166730

theorem complex_cube_sum (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 8) :
  Complex.abs (w^3 + z^3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_sum_l1667_166730


namespace NUMINAMATH_CALUDE_simplify_fraction_l1667_166739

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1667_166739


namespace NUMINAMATH_CALUDE_dragon_tower_theorem_l1667_166799

/-- Represents the configuration of a dragon tethered to a cylindrical tower. -/
structure DragonTower where
  towerRadius : ℝ
  ropeLength : ℝ
  dragonHeight : ℝ
  ropeTowerDistance : ℝ

/-- Represents the parameters of the rope touching the tower. -/
structure RopeParameters where
  p : ℕ
  q : ℕ
  r : ℕ

/-- Theorem stating the relationship between the dragon-tower configuration
    and the rope parameters. -/
theorem dragon_tower_theorem (dt : DragonTower) (rp : RopeParameters) :
  dt.towerRadius = 10 ∧
  dt.ropeLength = 30 ∧
  dt.dragonHeight = 6 ∧
  dt.ropeTowerDistance = 6 ∧
  Nat.Prime rp.r ∧
  (rp.p - Real.sqrt rp.q) / rp.r = Real.sqrt ((dt.ropeLength - dt.ropeTowerDistance)^2 - dt.towerRadius^2) -
    (dt.ropeLength * Real.sqrt (dt.towerRadius^2 + dt.dragonHeight^2)) / dt.towerRadius +
    dt.dragonHeight * Real.sqrt (dt.towerRadius^2 + dt.dragonHeight^2) / dt.towerRadius →
  rp.p + rp.q + rp.r = 993 :=
by sorry

end NUMINAMATH_CALUDE_dragon_tower_theorem_l1667_166799


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1667_166726

theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 20 = 80) (h2 : Q 100 = 20) :
  ∃ R : ℝ → ℝ, ∀ x, Q x = (x - 20) * (x - 100) * R x + (-3/4 * x + 95) := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1667_166726


namespace NUMINAMATH_CALUDE_even_function_implies_f_3_equals_5_l1667_166778

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) * (x - a)

-- State the theorem
theorem even_function_implies_f_3_equals_5 :
  (∀ x : ℝ, f a x = f a (-x)) → f a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_f_3_equals_5_l1667_166778


namespace NUMINAMATH_CALUDE_system_of_inequalities_l1667_166721

theorem system_of_inequalities (x : ℝ) : 
  (x - 1 < 3 ∧ x + 1 ≥ (1 + 2*x) / 3) ↔ -2 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l1667_166721


namespace NUMINAMATH_CALUDE_total_length_of_items_l1667_166769

theorem total_length_of_items (rubber pen pencil : ℝ) 
  (h1 : pen = rubber + 3)
  (h2 : pen = pencil - 2)
  (h3 : pencil = 12) :
  rubber + pen + pencil = 29 := by
sorry

end NUMINAMATH_CALUDE_total_length_of_items_l1667_166769


namespace NUMINAMATH_CALUDE_balance_spheres_l1667_166741

/-- Represents the density of a material -/
structure Density where
  value : ℝ
  positive : value > 0

/-- Represents the volume of a sphere -/
structure Volume where
  value : ℝ
  positive : value > 0

/-- Represents the mass of a sphere -/
structure Mass where
  value : ℝ
  positive : value > 0

/-- Represents a sphere with its properties -/
structure Sphere where
  density : Density
  volume : Volume
  mass : Mass

/-- Theorem: Balance of two spheres in air -/
theorem balance_spheres (cast_iron wood : Sphere) (air_density : Density) : 
  cast_iron.density.value > wood.density.value →
  cast_iron.volume.value < wood.volume.value →
  cast_iron.mass.value < wood.mass.value →
  (cast_iron.density.value - air_density.value) * cast_iron.volume.value = 
  (wood.density.value - air_density.value) * wood.volume.value →
  ∃ (fulcrum_position : ℝ), 
    fulcrum_position > 0 ∧ 
    fulcrum_position < 1 ∧ 
    fulcrum_position * cast_iron.mass.value = (1 - fulcrum_position) * wood.mass.value :=
by
  sorry

end NUMINAMATH_CALUDE_balance_spheres_l1667_166741


namespace NUMINAMATH_CALUDE_tree_distance_l1667_166781

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, and a sign 30 feet beyond the last tree,
    the total distance between the first tree and the sign is 205 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (s : ℝ) : 
  n = 8 → d = 100 → s = 30 → 
  (n - 1) * (d / 4) + s = 205 :=
by sorry

end NUMINAMATH_CALUDE_tree_distance_l1667_166781


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1667_166717

theorem regular_polygon_sides : ∃ (n : ℕ), n > 2 ∧ (2 * n - n * (n - 3) / 2 = 0) ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1667_166717


namespace NUMINAMATH_CALUDE_monic_quartic_value_l1667_166777

def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value (p : ℝ → ℝ) :
  is_monic_quartic p →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 5 = 40 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_value_l1667_166777


namespace NUMINAMATH_CALUDE_equation_solution_l1667_166793

theorem equation_solution :
  let f (x : ℝ) := Real.sqrt (7*x - 3) + Real.sqrt (2*x - 2)
  ∃ (x : ℝ), (f x = 3 ↔ (x = 2 ∨ x = 172/25)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1667_166793


namespace NUMINAMATH_CALUDE_remaining_black_portion_l1667_166756

/-- The fraction of black area remaining after one transformation -/
def black_fraction : ℚ := 3 / 4

/-- The number of transformations applied -/
def num_transformations : ℕ := 5

/-- The theorem stating the remaining black portion after transformations -/
theorem remaining_black_portion :
  black_fraction ^ num_transformations = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_remaining_black_portion_l1667_166756


namespace NUMINAMATH_CALUDE_women_to_men_ratio_l1667_166705

/-- Given an event with guests, prove the ratio of women to men --/
theorem women_to_men_ratio 
  (total_guests : ℕ) 
  (num_men : ℕ) 
  (num_children_after : ℕ) 
  (h1 : total_guests = 80) 
  (h2 : num_men = 40) 
  (h3 : num_children_after = 30) :
  (total_guests - num_men - (num_children_after - 10)) / num_men = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_women_to_men_ratio_l1667_166705


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1667_166789

def is_valid_number (n : ℕ) : Prop :=
  (n % 10 = 6) ∧ 
  (∃ m : ℕ, m > 0 ∧ 6 * 10^m + n / 10 = 4 * n)

theorem smallest_valid_number : 
  (is_valid_number 1538466) ∧ 
  (∀ k < 1538466, ¬(is_valid_number k)) := by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1667_166789


namespace NUMINAMATH_CALUDE_lawn_width_is_30_l1667_166775

/-- Represents the dimensions and properties of a rectangular lawn with roads --/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  road_width : ℝ
  gravel_cost_per_sqm : ℝ
  total_gravel_cost : ℝ

/-- Calculates the total area of the roads on the lawn --/
def road_area (l : LawnWithRoads) : ℝ :=
  l.length * l.road_width + (l.width - l.road_width) * l.road_width

/-- Theorem stating that the width of the lawn is 30 meters --/
theorem lawn_width_is_30 (l : LawnWithRoads) 
  (h1 : l.length = 70)
  (h2 : l.road_width = 5)
  (h3 : l.gravel_cost_per_sqm = 4)
  (h4 : l.total_gravel_cost = 1900)
  : l.width = 30 := by
  sorry

end NUMINAMATH_CALUDE_lawn_width_is_30_l1667_166775


namespace NUMINAMATH_CALUDE_total_water_consumed_water_consumed_is_686_l1667_166704

/-- Represents a medication schedule --/
structure MedicationSchedule where
  name : String
  timesPerDay : Nat
  waterPerDose : Nat

/-- Represents missed doses for a medication --/
structure MissedDoses where
  medication : String
  count : Nat

/-- Calculates the total water consumed for a medication over two weeks --/
def waterConsumedForMedication (schedule : MedicationSchedule) : Nat :=
  schedule.timesPerDay * schedule.waterPerDose * 7 * 2

/-- Calculates the water missed due to skipped doses --/
def waterMissedForMedication (schedule : MedicationSchedule) (missed : Nat) : Nat :=
  schedule.waterPerDose * missed

/-- The main theorem to prove --/
theorem total_water_consumed 
  (schedules : List MedicationSchedule)
  (missedDoses : List MissedDoses) : Nat :=
  let totalWater := schedules.map waterConsumedForMedication |>.sum
  let missedWater := missedDoses.map (fun m => 
    let schedule := schedules.find? (fun s => s.name == m.medication)
    match schedule with
    | some s => waterMissedForMedication s m.count
    | none => 0
  ) |>.sum
  totalWater - missedWater

/-- The specific medication schedules --/
def medicationSchedules : List MedicationSchedule := [
  { name := "A", timesPerDay := 3, waterPerDose := 4 },
  { name := "B", timesPerDay := 4, waterPerDose := 5 },
  { name := "C", timesPerDay := 2, waterPerDose := 6 },
  { name := "D", timesPerDay := 1, waterPerDose := 8 }
]

/-- The specific missed doses --/
def missedDosesList : List MissedDoses := [
  { medication := "A", count := 3 },
  { medication := "B", count := 2 },
  { medication := "C", count := 2 },
  { medication := "D", count := 1 }
]

/-- The main theorem for this specific problem --/
theorem water_consumed_is_686 : 
  total_water_consumed medicationSchedules missedDosesList = 686 := by
  sorry

end NUMINAMATH_CALUDE_total_water_consumed_water_consumed_is_686_l1667_166704


namespace NUMINAMATH_CALUDE_project_payment_main_project_payment_l1667_166779

/-- Represents the project details and calculates the total payment -/
structure Project where
  q_wage : ℝ  -- Hourly wage of candidate q
  p_hours : ℝ  -- Hours required by candidate p to complete the project
  total_payment : ℝ  -- Total payment for the project

/-- Theorem stating the total payment for the project is $540 -/
theorem project_payment (proj : Project) : proj.total_payment = 540 :=
  by
  have h1 : proj.q_wage + proj.q_wage / 2 = proj.q_wage + 9 := by sorry
  have h2 : (proj.q_wage + proj.q_wage / 2) * proj.p_hours = proj.q_wage * (proj.p_hours + 10) := by sorry
  have h3 : proj.total_payment = (proj.q_wage + proj.q_wage / 2) * proj.p_hours := by sorry
  sorry

/-- Main theorem proving the project payment is $540 -/
theorem main_project_payment : ∃ (proj : Project), proj.total_payment = 540 :=
  by
  sorry

end NUMINAMATH_CALUDE_project_payment_main_project_payment_l1667_166779


namespace NUMINAMATH_CALUDE_probability_red_then_black_specific_l1667_166701

/-- Represents a deck of cards with red and black cards -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h1 : red + black = total)

/-- Calculates the probability of drawing a red card first and a black card second -/
def probability_red_then_black (d : Deck) : ℚ :=
  (d.red : ℚ) / d.total * (d.black : ℚ) / (d.total - 1)

/-- Theorem: The probability of drawing a red card first and a black card second
    from a deck with 20 red cards and 32 black cards (total 52 cards) is 160/663 -/
theorem probability_red_then_black_specific :
  let d : Deck := ⟨52, 20, 32, by simp⟩
  probability_red_then_black d = 160 / 663 := by sorry

end NUMINAMATH_CALUDE_probability_red_then_black_specific_l1667_166701


namespace NUMINAMATH_CALUDE_parabola_equation_l1667_166700

/-- A parabola with directrix x = -7 has the standard equation y² = 28x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ x = -7) →  -- directrix equation
  (∃ k, ∀ x y, p (x, y) ↔ y^2 = 4 * k * x ∧ k > 0) →  -- general form of parabola equation
  (∀ x y, p (x, y) ↔ y^2 = 28 * x) :=  -- standard equation to be proved
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1667_166700


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1667_166709

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l1667_166709


namespace NUMINAMATH_CALUDE_solution_difference_l1667_166788

theorem solution_difference (p q : ℝ) : 
  ((p - 4) * (p + 4) = 24 * p - 96) →
  ((q - 4) * (q + 4) = 24 * q - 96) →
  p ≠ q →
  p > q →
  p - q = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1667_166788


namespace NUMINAMATH_CALUDE_count_integers_in_range_l1667_166714

theorem count_integers_in_range : ∃ (S : Finset ℤ), 
  (∀ n : ℤ, n ∈ S ↔ -12 * Real.sqrt Real.pi ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ 15 * Real.pi) ∧ 
  Finset.card S = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l1667_166714


namespace NUMINAMATH_CALUDE_andy_final_position_l1667_166708

-- Define the direction as an enumeration
inductive Direction
  | North
  | West
  | South
  | East

-- Define the position as a pair of integers
def Position := ℤ × ℤ

-- Define the function to get the next direction after turning left
def turn_left (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East
  | Direction.East => Direction.North

-- Define the function to move in a given direction
def move (p : Position) (d : Direction) (distance : ℤ) : Position :=
  match d with
  | Direction.North => (p.1, p.2 + distance)
  | Direction.West => (p.1 - distance, p.2)
  | Direction.South => (p.1, p.2 - distance)
  | Direction.East => (p.1 + distance, p.2)

-- Define the function to perform one step of Andy's movement
def step (p : Position) (d : Direction) (n : ℕ) : Position × Direction :=
  let new_p := move p d (n^2)
  let new_d := turn_left d
  (new_p, new_d)

-- Define the function to perform multiple steps
def multi_step (initial_p : Position) (initial_d : Direction) (steps : ℕ) : Position :=
  if steps = 0 then
    initial_p
  else
    let (p, d) := (List.range steps).foldl
      (fun (acc : Position × Direction) n => step acc.1 acc.2 (n + 1))
      (initial_p, initial_d)
    p

-- Theorem statement
theorem andy_final_position :
  multi_step (10, -10) Direction.North 16 = (154, -138) :=
sorry

end NUMINAMATH_CALUDE_andy_final_position_l1667_166708


namespace NUMINAMATH_CALUDE_min_squared_distance_to_line_l1667_166703

/-- The minimum squared distance from a point on the line x - y - 1 = 0 to the point (2, 2) -/
theorem min_squared_distance_to_line (x y : ℝ) :
  x - y - 1 = 0 → (∀ x' y' : ℝ, x' - y' - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≤ (x' - 2)^2 + (y' - 2)^2) →
  (x - 2)^2 + (y - 2)^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_squared_distance_to_line_l1667_166703


namespace NUMINAMATH_CALUDE_solve_for_k_l1667_166743

theorem solve_for_k (x y k : ℝ) : 
  x = 2 → 
  y = 1 → 
  k * x - y = 3 → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_k_l1667_166743


namespace NUMINAMATH_CALUDE_remainder_2023_div_73_l1667_166798

theorem remainder_2023_div_73 : 2023 % 73 = 52 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2023_div_73_l1667_166798


namespace NUMINAMATH_CALUDE_expression_factorization_l1667_166702

theorem expression_factorization (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1667_166702


namespace NUMINAMATH_CALUDE_line_through_points_l1667_166752

/-- Given a line y = ax + b passing through points (3, 7) and (6, 19), prove that a - b = 9 -/
theorem line_through_points (a b : ℝ) : 
  (7 : ℝ) = a * 3 + b ∧ (19 : ℝ) = a * 6 + b → a - b = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1667_166752


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l1667_166742

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ 2 / p = 1 / n + 1 / m ∧ n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l1667_166742


namespace NUMINAMATH_CALUDE_inequality_range_l1667_166706

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ a) ↔ a ∈ Set.Iic 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1667_166706


namespace NUMINAMATH_CALUDE_complex_ratio_l1667_166750

theorem complex_ratio (z : ℂ) (a b : ℝ) (h1 : z = Complex.mk a b) (h2 : z * (1 - Complex.I) = Complex.I) :
  a / b = -1 := by sorry

end NUMINAMATH_CALUDE_complex_ratio_l1667_166750


namespace NUMINAMATH_CALUDE_sqrt_sum_sin_equals_sqrt_two_minus_cos_l1667_166774

theorem sqrt_sum_sin_equals_sqrt_two_minus_cos (α : Real) 
  (h : 5 * Real.pi / 2 ≤ α ∧ α ≤ 7 * Real.pi / 2) : 
  Real.sqrt (1 + Real.sin α) + Real.sqrt (1 - Real.sin α) = Real.sqrt (2 - Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_sin_equals_sqrt_two_minus_cos_l1667_166774


namespace NUMINAMATH_CALUDE_units_digit_17_39_l1667_166765

theorem units_digit_17_39 : (17^39) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_39_l1667_166765


namespace NUMINAMATH_CALUDE_min_employees_for_given_requirements_l1667_166707

/-- Represents the number of employees needed for each pollution type and their intersections -/
structure PollutionMonitoring where
  water : ℕ
  air : ℕ
  soil : ℕ
  water_air : ℕ
  air_soil : ℕ
  soil_water : ℕ
  all_three : ℕ

/-- Calculates the minimum number of employees needed given the monitoring requirements -/
def min_employees (p : PollutionMonitoring) : ℕ :=
  p.water + p.air + p.soil - p.water_air - p.air_soil - p.soil_water + p.all_three

/-- Theorem stating that given the specific monitoring requirements, 225 employees are needed -/
theorem min_employees_for_given_requirements :
  let p : PollutionMonitoring := {
    water := 115,
    air := 92,
    soil := 60,
    water_air := 32,
    air_soil := 20,
    soil_water := 10,
    all_three := 5
  }
  min_employees p = 225 := by
  sorry


end NUMINAMATH_CALUDE_min_employees_for_given_requirements_l1667_166707


namespace NUMINAMATH_CALUDE_sandwiches_problem_l1667_166712

theorem sandwiches_problem (S : ℚ) :
  (S > 0) →
  (3/4 * S - 1/8 * S - 1/4 * S - 5 = 4) →
  S = 24 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_problem_l1667_166712


namespace NUMINAMATH_CALUDE_BG_length_l1667_166737

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2)
  (AB_length : B.1 - A.1 = 6)
  (BC_length : C.2 - B.2 = 4)

-- Define point E on BC
def E (rect : Rectangle) : ℝ × ℝ := (rect.B.1, rect.B.2 + 3)

-- Define point F on AE
def F (rect : Rectangle) : ℝ × ℝ := (4, 2)

-- Define point G as intersection of DF and BC
def G (rect : Rectangle) : ℝ × ℝ := (rect.B.1, 1)

-- Theorem statement
theorem BG_length (rect : Rectangle) : (G rect).2 - rect.B.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_BG_length_l1667_166737


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l1667_166748

theorem mod_equivalence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l1667_166748


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l1667_166715

theorem nth_equation_pattern (n : ℕ) (h : n > 0) :
  9 * (n - 1) + n = 10 * (n - 1) + 1 :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l1667_166715


namespace NUMINAMATH_CALUDE_subset_implies_m_range_l1667_166792

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 3}

-- Define the range of m
def m_range : Set ℝ := {m | m < -4 ∨ m > 2}

-- Theorem statement
theorem subset_implies_m_range :
  ∀ m : ℝ, B m ⊆ A → m ∈ m_range :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_m_range_l1667_166792


namespace NUMINAMATH_CALUDE_pullup_median_is_5_point_5_l1667_166724

def pullup_counts : List ℕ := [4, 4, 5, 5, 5, 6, 6, 7, 7, 8]

def median (l : List ℝ) : ℝ := sorry

theorem pullup_median_is_5_point_5 :
  median (pullup_counts.map (λ x => (x : ℝ))) = 5.5 := by sorry

end NUMINAMATH_CALUDE_pullup_median_is_5_point_5_l1667_166724


namespace NUMINAMATH_CALUDE_composite_prime_calculation_l1667_166760

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]
def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem composite_prime_calculation :
  (((first_six_composites.prod : ℚ) / (next_six_composites.prod : ℚ)) * (first_five_primes.prod : ℚ)) = 377.55102040816324 := by
  sorry

end NUMINAMATH_CALUDE_composite_prime_calculation_l1667_166760


namespace NUMINAMATH_CALUDE_cards_distribution_l1667_166732

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 72) 
  (h2 : num_people = 10) : 
  (num_people - (total_cards % num_people)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l1667_166732


namespace NUMINAMATH_CALUDE_basketball_league_games_l1667_166740

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team, 
    the total number of games is 180 -/
theorem basketball_league_games : total_games 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l1667_166740


namespace NUMINAMATH_CALUDE_ratio_inequality_not_always_true_l1667_166729

theorem ratio_inequality_not_always_true :
  ¬ (∀ (a b c d : ℝ), (a / b = c / d) → (a > b → c > d)) := by
  sorry

end NUMINAMATH_CALUDE_ratio_inequality_not_always_true_l1667_166729


namespace NUMINAMATH_CALUDE_courier_distance_l1667_166758

/-- The total distance from A to B -/
def total_distance : ℝ := 412.5

/-- The additional distance traveled -/
def additional_distance : ℝ := 60

/-- The ratio of distance covered to remaining distance at the first point -/
def initial_ratio : ℚ := 2/3

/-- The ratio of distance covered to remaining distance after traveling the additional distance -/
def final_ratio : ℚ := 6/5

theorem courier_distance :
  ∃ (x : ℝ),
    (2 * x) / (3 * x) = initial_ratio ∧
    (2 * x + additional_distance) / (3 * x - additional_distance) = final_ratio ∧
    5 * x = total_distance :=
by sorry

end NUMINAMATH_CALUDE_courier_distance_l1667_166758


namespace NUMINAMATH_CALUDE_fifty_third_number_is_61_l1667_166710

def adjustedSequence (n : ℕ) : ℕ :=
  n + (n - 1) / 4

theorem fifty_third_number_is_61 :
  adjustedSequence 53 = 61 := by
  sorry

end NUMINAMATH_CALUDE_fifty_third_number_is_61_l1667_166710


namespace NUMINAMATH_CALUDE_max_sphere_ratio_l1667_166722

/-- Represents the configuration of spheres within two cones as described in the problem -/
structure SpheresInCones where
  r : ℝ  -- radius of the first two identical spheres
  x : ℝ  -- radius of the third sphere
  R : ℝ  -- radius of the base of the cones
  h : ℝ  -- height of each cone
  s : ℝ  -- slant height of each cone

/-- The conditions given in the problem -/
def problem_conditions (config : SpheresInCones) : Prop :=
  config.r > 0 ∧
  config.x > 0 ∧
  config.R > 0 ∧
  config.h > 0 ∧
  config.s > 0 ∧
  config.h = config.s / 2 ∧
  config.R = 3 * config.r

/-- The theorem stating the maximum ratio of the third sphere's radius to the first sphere's radius -/
theorem max_sphere_ratio (config : SpheresInCones) 
  (h : problem_conditions config) :
  ∃ (t : ℝ), t = config.x / config.r ∧ 
             t ≤ (7 - Real.sqrt 22) / 3 ∧
             ∀ (t' : ℝ), t' = config.x / config.r → t' ≤ t :=
sorry

end NUMINAMATH_CALUDE_max_sphere_ratio_l1667_166722


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l1667_166734

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l1667_166734


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1667_166751

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x < 4*x →
  x + 4*x > 20 →
  4*x + 20 > x →
  x + 4*x + 20 ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1667_166751


namespace NUMINAMATH_CALUDE_fourth_root_of_polynomial_l1667_166797

theorem fourth_root_of_polynomial (a b : ℝ) : 
  (∀ x : ℝ, a * x^4 + (a + 2*b) * x^3 + (b - 3*a) * x^2 + (2*a - 6) * x + (7 - a) = 0 ↔ 
    x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2) → 
  ∃ x : ℝ, x = -2 ∧ a * x^4 + (a + 2*b) * x^3 + (b - 3*a) * x^2 + (2*a - 6) * x + (7 - a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_of_polynomial_l1667_166797


namespace NUMINAMATH_CALUDE_line_direction_vector_c_l1667_166785

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (c : ℝ) : Prop :=
  let direction := (p2.1 - p1.1, p2.2 - p1.2)
  direction.1 = 3 ∧ direction.2 = c

/-- Theorem stating that for a line passing through (-6, 1) and (-3, 4) with direction vector (3, c), c must equal 3 -/
theorem line_direction_vector_c (c : ℝ) :
  Line (-6, 1) (-3, 4) c → c = 3 := by
  sorry


end NUMINAMATH_CALUDE_line_direction_vector_c_l1667_166785


namespace NUMINAMATH_CALUDE_centroid_tetrahedron_volume_centroid_tetrahedron_volume_54_l1667_166780

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- Represents the tetrahedron formed by the centroids of the faces of another tetrahedron -/
def centroid_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

/-- The volume of the centroid tetrahedron is 1/27 of the original tetrahedron's volume -/
theorem centroid_tetrahedron_volume (t : RegularTetrahedron) :
  (centroid_tetrahedron t).volume = t.volume / 27 :=
sorry

/-- Given a regular tetrahedron with volume 54, the volume of the tetrahedron
    formed by the centroids of its four faces is 2 -/
theorem centroid_tetrahedron_volume_54 :
  let t : RegularTetrahedron := ⟨54⟩
  (centroid_tetrahedron t).volume = 2 :=
sorry

end NUMINAMATH_CALUDE_centroid_tetrahedron_volume_centroid_tetrahedron_volume_54_l1667_166780


namespace NUMINAMATH_CALUDE_complete_square_factorization_quadratic_factorization_l1667_166738

/-- A quadratic expression ax^2 + bx + c can be factored using the complete square formula
    if and only if b = ±2√(ac) -/
theorem complete_square_factorization (a b c : ℝ) :
  (∃ (k : ℝ), b = 2 * k * Real.sqrt (a * c)) ∨ (∃ (k : ℝ), b = -2 * k * Real.sqrt (a * c)) ↔
  ∃ (p q : ℝ), a * x^2 + b * x + c = a * (x - p)^2 + q := sorry

/-- For the quadratic expression 4x^2 - (m+1)x + 9 to be factored using the complete square formula,
    m must equal 11 or -13 -/
theorem quadratic_factorization (m : ℝ) :
  (∃ (p q : ℝ), 4 * x^2 - (m + 1) * x + 9 = 4 * (x - p)^2 + q) ↔ (m = 11 ∨ m = -13) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_factorization_quadratic_factorization_l1667_166738


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1667_166745

theorem cubic_expression_value (x : ℝ) (h : x^2 - 2*x - 1 = 0) :
  x^3 - x^2 - 3*x + 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1667_166745


namespace NUMINAMATH_CALUDE_total_rainfall_sum_l1667_166790

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℝ := 0.17

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℝ := 0.42

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℝ := 0.08

/-- The total rainfall recorded over the three days -/
def total_rainfall : ℝ := monday_rainfall + tuesday_rainfall + wednesday_rainfall

/-- Theorem stating that the total rainfall is equal to 0.67 cm -/
theorem total_rainfall_sum : total_rainfall = 0.67 := by sorry

end NUMINAMATH_CALUDE_total_rainfall_sum_l1667_166790


namespace NUMINAMATH_CALUDE_road_trip_distance_l1667_166731

/-- Proves that given the conditions of the road trip, the first day's distance is 200 miles -/
theorem road_trip_distance (total_distance : ℝ) (day1 : ℝ) :
  total_distance = 525 →
  total_distance = day1 + (3/4 * day1) + (1/2 * (day1 + (3/4 * day1))) →
  day1 = 200 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distance_l1667_166731


namespace NUMINAMATH_CALUDE_reflect_2_5_across_x_axis_l1667_166759

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Theorem: Reflecting the point (2,5) across the x-axis results in (2,-5) -/
theorem reflect_2_5_across_x_axis :
  reflectAcrossXAxis { x := 2, y := 5 } = { x := 2, y := -5 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_2_5_across_x_axis_l1667_166759


namespace NUMINAMATH_CALUDE_product_with_9999_l1667_166772

theorem product_with_9999 : ∃ x : ℝ, x * 9999 = 4690910862 ∧ x = 469.1 := by
  sorry

end NUMINAMATH_CALUDE_product_with_9999_l1667_166772


namespace NUMINAMATH_CALUDE_candy_distribution_l1667_166713

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) (leftover_candies : ℕ) :
  total_candies = 67 →
  candies_per_student = 4 →
  leftover_candies = 3 →
  (total_candies - leftover_candies) / candies_per_student = 16 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1667_166713


namespace NUMINAMATH_CALUDE_largest_common_term_of_arithmetic_progressions_l1667_166753

theorem largest_common_term_of_arithmetic_progressions :
  let seq1 (n : ℕ) := 4 + 5 * n
  let seq2 (m : ℕ) := 3 + 7 * m
  ∃ (n m : ℕ), seq1 n = seq2 m ∧ seq1 n = 299 ∧
  ∀ (k l : ℕ), seq1 k = seq2 l → seq1 k ≤ 299 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_of_arithmetic_progressions_l1667_166753


namespace NUMINAMATH_CALUDE_roses_in_vase_l1667_166762

/-- The total number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: The total number of roses is 22 when there were initially 6 roses and 16 were added -/
theorem roses_in_vase : total_roses 6 16 = 22 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1667_166762


namespace NUMINAMATH_CALUDE_total_baked_goods_is_338_l1667_166766

/-- The total number of baked goods Diane makes -/
def total_baked_goods : ℕ :=
  let gingerbread_trays : ℕ := 4
  let gingerbread_per_tray : ℕ := 25
  let chocolate_chip_trays : ℕ := 3
  let chocolate_chip_per_tray : ℕ := 30
  let oatmeal_trays : ℕ := 2
  let oatmeal_per_tray : ℕ := 20
  let sugar_trays : ℕ := 6
  let sugar_per_tray : ℕ := 18
  gingerbread_trays * gingerbread_per_tray +
  chocolate_chip_trays * chocolate_chip_per_tray +
  oatmeal_trays * oatmeal_per_tray +
  sugar_trays * sugar_per_tray

theorem total_baked_goods_is_338 : total_baked_goods = 338 := by
  sorry

end NUMINAMATH_CALUDE_total_baked_goods_is_338_l1667_166766


namespace NUMINAMATH_CALUDE_tallest_player_height_l1667_166716

/-- Given a basketball team where the tallest player is 9.5 inches taller than
    the shortest player, and the shortest player is 68.25 inches tall,
    prove that the tallest player is 77.75 inches tall. -/
theorem tallest_player_height :
  let shortest_player_height : ℝ := 68.25
  let height_difference : ℝ := 9.5
  let tallest_player_height : ℝ := shortest_player_height + height_difference
  tallest_player_height = 77.75 := by sorry

end NUMINAMATH_CALUDE_tallest_player_height_l1667_166716


namespace NUMINAMATH_CALUDE_ab_range_l1667_166768

-- Define the line equation
def line_equation (a b x y : ℝ) : Prop := a * x - b * y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the property of bisecting the circumference
def bisects_circle (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), line_equation a b x y ∧ circle_equation x y

-- Theorem statement
theorem ab_range (a b : ℝ) : 
  bisects_circle a b → ab ∈ Set.Iic (1/8) :=
sorry

end NUMINAMATH_CALUDE_ab_range_l1667_166768


namespace NUMINAMATH_CALUDE_total_squares_is_86_l1667_166746

/-- The number of squares of a given size in a 6x6 grid -/
def count_squares (size : Nat) : Nat :=
  (7 - size) ^ 2

/-- The total number of squares of sizes 1x1, 2x2, 3x3, and 4x4 in a 6x6 grid -/
def total_squares : Nat :=
  count_squares 1 + count_squares 2 + count_squares 3 + count_squares 4

theorem total_squares_is_86 : total_squares = 86 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_is_86_l1667_166746
