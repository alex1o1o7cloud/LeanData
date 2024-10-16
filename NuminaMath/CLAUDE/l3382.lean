import Mathlib

namespace NUMINAMATH_CALUDE_minimum_value_problem_l3382_338220

theorem minimum_value_problem (x y : ℝ) (h1 : 5 * x - x * y - y = -6) (h2 : x > -1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≥ z := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l3382_338220


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3382_338237

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1

-- Define the points A, B, F₁ (left focus), and F₂ (right focus)
variable (A B F₁ F₂ : ℝ × ℝ)

-- Define the conditions
def chord_passes_through_left_focus : Prop :=
  ∃ t : ℝ, A = (1 - t) • F₁ + t • B ∧ 0 ≤ t ∧ t ≤ 1

def chord_length_is_6 : Prop :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6

-- Define the theorem
theorem hyperbola_triangle_perimeter
  (h1 : hyperbola F₁.1 F₁.2)
  (h2 : hyperbola F₂.1 F₂.2)
  (h3 : chord_passes_through_left_focus A B F₁)
  (h4 : chord_length_is_6 A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) +
  Real.sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2) = 28 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l3382_338237


namespace NUMINAMATH_CALUDE_value_of_x_when_sqrt_fraction_is_zero_l3382_338231

theorem value_of_x_when_sqrt_fraction_is_zero :
  ∀ x : ℝ, x ≠ 0 → (Real.sqrt (2 - x)) / x = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_when_sqrt_fraction_is_zero_l3382_338231


namespace NUMINAMATH_CALUDE_constant_term_of_equation_l3382_338203

/-- The constant term of a quadratic equation ax^2 + bx + c = 0 is c -/
def constant_term (a b c : ℝ) : ℝ := c

theorem constant_term_of_equation :
  constant_term 3 1 5 = 5 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_equation_l3382_338203


namespace NUMINAMATH_CALUDE_x_varies_as_z_l3382_338243

-- Define the variables and constants
variable (x y z : ℝ)
variable (k j : ℝ)

-- Define the conditions
axiom x_varies_as_y : ∃ k, x = k * y^3
axiom y_varies_as_z : ∃ j, y = j * z^(1/4)

-- Define the theorem to prove
theorem x_varies_as_z : ∃ m, x = m * z^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_z_l3382_338243


namespace NUMINAMATH_CALUDE_probability_is_one_third_l3382_338234

/-- A standard die with six faces -/
def StandardDie : Type := Fin 6

/-- The total number of dots on all faces of a standard die -/
def totalDots : ℕ := 21

/-- The number of favorable outcomes (faces with 1 or 2 dots) -/
def favorableOutcomes : ℕ := 2

/-- The total number of possible outcomes (total faces) -/
def totalOutcomes : ℕ := 6

/-- The probability of the sum of dots on five faces being at least 19 -/
def probability : ℚ := favorableOutcomes / totalOutcomes

theorem probability_is_one_third :
  probability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l3382_338234


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3382_338241

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3382_338241


namespace NUMINAMATH_CALUDE_steve_shared_oranges_l3382_338276

/-- The number of oranges Steve shared with Patrick -/
def oranges_shared (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem steve_shared_oranges :
  oranges_shared 46 42 = 4 := by
  sorry

end NUMINAMATH_CALUDE_steve_shared_oranges_l3382_338276


namespace NUMINAMATH_CALUDE_girls_joined_correct_l3382_338267

/-- The number of girls who joined the school -/
def girls_joined : ℕ := 465

/-- The initial number of girls in the school -/
def initial_girls : ℕ := 632

/-- The initial number of boys in the school -/
def initial_boys : ℕ := 410

/-- The difference between the number of girls and boys after some girls joined -/
def girl_boy_difference : ℕ := 687

theorem girls_joined_correct :
  initial_girls + girls_joined = initial_boys + girl_boy_difference :=
by sorry

end NUMINAMATH_CALUDE_girls_joined_correct_l3382_338267


namespace NUMINAMATH_CALUDE_bill_denomination_l3382_338298

theorem bill_denomination (num_bills : ℕ) (total_value : ℕ) (denomination : ℕ) :
  num_bills = 10 →
  total_value = 50 →
  num_bills * denomination = total_value →
  denomination = 5 := by
  sorry

end NUMINAMATH_CALUDE_bill_denomination_l3382_338298


namespace NUMINAMATH_CALUDE_naomi_bike_count_l3382_338280

theorem naomi_bike_count (total_wheels : ℕ) (childrens_bikes : ℕ) (regular_bike_wheels : ℕ) (childrens_bike_wheels : ℕ) : 
  total_wheels = 58 →
  childrens_bikes = 11 →
  regular_bike_wheels = 2 →
  childrens_bike_wheels = 4 →
  ∃ (regular_bikes : ℕ), regular_bikes = 7 ∧ 
    total_wheels = regular_bikes * regular_bike_wheels + childrens_bikes * childrens_bike_wheels :=
by
  sorry

end NUMINAMATH_CALUDE_naomi_bike_count_l3382_338280


namespace NUMINAMATH_CALUDE_range_of_a_l3382_338265

-- Define proposition P
def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition Q
def Q (a : ℝ) : Prop := a * (a - 3) < 0

-- Define the set of valid a values
def valid_a_set : Set ℝ := {a | a = 0 ∨ (3 ≤ a ∧ a < 4)}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3382_338265


namespace NUMINAMATH_CALUDE_correct_selection_methods_l3382_338248

/-- The number of ways to select students for health checks -/
def select_students (total_students : ℕ) (num_classes : ℕ) (min_per_class : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of selection methods -/
theorem correct_selection_methods :
  select_students 23 10 2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_correct_selection_methods_l3382_338248


namespace NUMINAMATH_CALUDE_extended_ohara_triple_49_64_l3382_338204

/-- Definition of an Extended O'Hara triple -/
def is_extended_ohara_triple (a b x : ℕ) : Prop :=
  2 * Real.sqrt a + Real.sqrt b = x

/-- Theorem: If (49, 64, x) is an Extended O'Hara triple, then x = 22 -/
theorem extended_ohara_triple_49_64 (x : ℕ) :
  is_extended_ohara_triple 49 64 x → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_extended_ohara_triple_49_64_l3382_338204


namespace NUMINAMATH_CALUDE_quadratic_m_gt_n_l3382_338275

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The y-value of a quadratic function at a given x -/
def eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_m_gt_n 
  (f : QuadraticFunction)
  (h1 : eval f (-1) = 0)
  (h2 : eval f 0 = 2)
  (h3 : eval f 3 = 0)
  (m n : ℝ)
  (hm : eval f 1 = m)
  (hn : eval f 2 = n) :
  m > n := by
  sorry

end NUMINAMATH_CALUDE_quadratic_m_gt_n_l3382_338275


namespace NUMINAMATH_CALUDE_min_value_expression_l3382_338279

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  4 * x^2 + 9 * y^2 + 16 / x^2 + 6 * y / x ≥ 2 * Real.sqrt 564 / 3 ∧
  ∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ y₀ ≠ 0 ∧
    4 * x₀^2 + 9 * y₀^2 + 16 / x₀^2 + 6 * y₀ / x₀ = 2 * Real.sqrt 564 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3382_338279


namespace NUMINAMATH_CALUDE_ice_melting_volume_l3382_338252

theorem ice_melting_volume (ice_volume : ℝ) (h1 : ice_volume = 2) :
  let water_volume := ice_volume * (10/11)
  water_volume = 20/11 :=
by sorry

end NUMINAMATH_CALUDE_ice_melting_volume_l3382_338252


namespace NUMINAMATH_CALUDE_problem_solution_l3382_338261

theorem problem_solution (x y z : ℝ) 
  (hx : x ≠ 0)
  (hz : z ≠ 0)
  (eq1 : x/2 = y^2 + z)
  (eq2 : x/4 = 4*y + 2*z) :
  x = 120 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3382_338261


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l3382_338219

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := (m * (x^2 - 1)) / x - 2 * Real.log x

theorem f_has_three_zeros :
  ∃ (a b c : ℝ), a < b ∧ b < c ∧ 
  (∀ x, x > 0 → f (1/2) x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry


end NUMINAMATH_CALUDE_f_has_three_zeros_l3382_338219


namespace NUMINAMATH_CALUDE_option2_saves_money_at_80_l3382_338230

/-- The total charge for Option 1 given x participants -/
def option1_charge (x : ℝ) : ℝ := 1500 + 320 * x

/-- The total charge for Option 2 given x participants -/
def option2_charge (x : ℝ) : ℝ := 360 * x - 1800

/-- The original price per person -/
def original_price : ℝ := 400

theorem option2_saves_money_at_80 :
  ∀ x : ℝ, x > 50 → option2_charge 80 < option1_charge 80 := by
  sorry

end NUMINAMATH_CALUDE_option2_saves_money_at_80_l3382_338230


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l3382_338214

theorem parabola_point_focus_distance (p : ℝ) (y : ℝ) (h1 : p > 0) :
  y^2 = 2*p*8 ∧ (8 + p/2)^2 + y^2 = 10^2 → p = 4 ∧ (y = 8 ∨ y = -8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l3382_338214


namespace NUMINAMATH_CALUDE_speed_calculation_l3382_338233

theorem speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 50 → time = 2.5 → speed = distance / time → speed = 20 := by
sorry

end NUMINAMATH_CALUDE_speed_calculation_l3382_338233


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_geq_four_l3382_338283

theorem quadratic_always_nonnegative_implies_a_geq_four (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*x + a ≥ 0) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_geq_four_l3382_338283


namespace NUMINAMATH_CALUDE_largest_option_l3382_338293

open Real

/-- Given a function f: ℝ → ℝ satisfying xf'(x) + f(x) > 0 for all x, 
    and 0 < a < b < 1, prove that log_{ba} · f(log_{ba}) is the largest among the given options -/
theorem largest_option (f : ℝ → ℝ) (f_deriv : ℝ → ℝ) (a b : ℝ) 
  (h_f : ∀ x, x * f_deriv x + f x > 0)
  (h_a : 0 < a) (h_ab : a < b) (h_b : b < 1) :
  let D := (log b / log a) * f (log b / log a)
  D > a * b * f (a * b) ∧
  D > b * a * f (b * a) ∧
  D > (log a / log b) * f (log a / log b) := by
sorry


end NUMINAMATH_CALUDE_largest_option_l3382_338293


namespace NUMINAMATH_CALUDE_lucky_larry_challenge_l3382_338250

theorem lucky_larry_challenge (a b c d e f : ℤ) :
  a = 2 ∧ b = 4 ∧ c = 6 ∧ d = 8 ∧ f = 5 →
  (a + b - c + d - e + f = a + (b - (c + (d - (e + f))))) ↔ e = 8 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_challenge_l3382_338250


namespace NUMINAMATH_CALUDE_second_sum_calculation_l3382_338258

/-- Proves that given the conditions, the second sum is 1704 --/
theorem second_sum_calculation (total : ℝ) (first_part : ℝ) (second_part : ℝ) 
  (h1 : total = 2769)
  (h2 : total = first_part + second_part)
  (h3 : (first_part * 3 * 8 / 100) = (second_part * 5 * 3 / 100)) :
  second_part = 1704 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_calculation_l3382_338258


namespace NUMINAMATH_CALUDE_quadratic_increasing_l3382_338287

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_increasing (a b c : ℝ) 
  (h1 : f a b c 0 = f a b c 6) 
  (h2 : f a b c 6 < f a b c 7) :
  ∀ x y, 3 < x → x < y → f a b c x < f a b c y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_l3382_338287


namespace NUMINAMATH_CALUDE_golden_ratio_experiment_l3382_338295

/-- The 0.618 method for finding the optimal amount in an experiment --/
def golden_ratio_method (range_start range_end good_point : ℝ) : ℝ :=
  range_end - (good_point - range_start)

/-- Theorem: The 0.618 method yields 684 as the bad point for the given range and good point --/
theorem golden_ratio_experiment :
  let range_start : ℝ := 628
  let range_end : ℝ := 774
  let good_point : ℝ := 718
  golden_ratio_method range_start range_end good_point = 684 := by
  sorry


end NUMINAMATH_CALUDE_golden_ratio_experiment_l3382_338295


namespace NUMINAMATH_CALUDE_attendance_difference_l3382_338251

def football_game_attendance (saturday monday wednesday friday thursday expected_total : ℕ) : Prop :=
  let total := saturday + monday + wednesday + friday + thursday
  saturday = 80 ∧
  monday = saturday - 20 ∧
  wednesday = monday + 50 ∧
  friday = saturday + monday ∧
  thursday = 45 ∧
  expected_total = 350 ∧
  total - expected_total = 85

theorem attendance_difference :
  ∃ (saturday monday wednesday friday thursday expected_total : ℕ),
    football_game_attendance saturday monday wednesday friday thursday expected_total :=
by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l3382_338251


namespace NUMINAMATH_CALUDE_range_of_m_l3382_338211

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3382_338211


namespace NUMINAMATH_CALUDE_surface_area_of_special_rectangular_solid_l3382_338254

/-- A function that checks if a number is prime or a square of a prime -/
def isPrimeOrSquareOfPrime (n : ℕ) : Prop :=
  Nat.Prime n ∨ ∃ p, Nat.Prime p ∧ n = p^2

/-- Definition of a rectangular solid with the given properties -/
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  length_valid : isPrimeOrSquareOfPrime length
  width_valid : isPrimeOrSquareOfPrime width
  height_valid : isPrimeOrSquareOfPrime height
  volume_is_1155 : length * width * height = 1155

/-- The theorem to be proved -/
theorem surface_area_of_special_rectangular_solid (r : RectangularSolid) :
  2 * (r.length * r.width + r.width * r.height + r.height * r.length) = 814 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_special_rectangular_solid_l3382_338254


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l3382_338226

theorem linear_equation_equivalence (x y : ℚ) :
  2 * x + 3 * y - 4 = 0 →
  (y = (4 - 2 * x) / 3 ∧ x = (4 - 3 * y) / 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l3382_338226


namespace NUMINAMATH_CALUDE_pencil_users_count_l3382_338201

/-- The number of attendants who used a pen -/
def pen_users : ℕ := 15

/-- The number of attendants who used only one type of writing tool -/
def single_tool_users : ℕ := 20

/-- The number of attendants who used both types of writing tools -/
def both_tool_users : ℕ := 10

/-- The number of attendants who used a pencil -/
def pencil_users : ℕ := single_tool_users + both_tool_users - (pen_users - both_tool_users)

theorem pencil_users_count : pencil_users = 25 := by sorry

end NUMINAMATH_CALUDE_pencil_users_count_l3382_338201


namespace NUMINAMATH_CALUDE_birthday_crayons_l3382_338249

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 291

/-- The number of crayons Paul had lost or given away -/
def crayons_lost_or_given : ℕ := 315

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost_or_given

theorem birthday_crayons : total_crayons = 606 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l3382_338249


namespace NUMINAMATH_CALUDE_carmen_pets_difference_l3382_338210

/-- Proves that Carmen has 14 fewer cats than dogs after giving up some cats for adoption -/
theorem carmen_pets_difference (initial_cats initial_dogs : ℕ) 
  (cats_given_up_round1 cats_given_up_round2 cats_given_up_round3 : ℕ) : 
  initial_cats = 48 →
  initial_dogs = 36 →
  cats_given_up_round1 = 6 →
  cats_given_up_round2 = 12 →
  cats_given_up_round3 = 8 →
  initial_cats - (cats_given_up_round1 + cats_given_up_round2 + cats_given_up_round3) = initial_dogs - 14 :=
by
  sorry

end NUMINAMATH_CALUDE_carmen_pets_difference_l3382_338210


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3382_338213

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 10 / (a 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3382_338213


namespace NUMINAMATH_CALUDE_division_problem_l3382_338246

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 23)
  (h2 : divisor = 4)
  (h3 : remainder = 3)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3382_338246


namespace NUMINAMATH_CALUDE_bus_tour_tickets_sold_l3382_338292

/-- A bus tour selling tickets to senior citizens and regular passengers -/
structure BusTour where
  senior_price : ℕ
  regular_price : ℕ
  total_sales : ℕ
  senior_tickets : ℕ

/-- The total number of tickets sold in a bus tour -/
def total_tickets (tour : BusTour) : ℕ :=
  tour.senior_tickets + (tour.total_sales - tour.senior_tickets * tour.senior_price) / tour.regular_price

/-- Theorem stating that for the given conditions, the total number of tickets sold is 65 -/
theorem bus_tour_tickets_sold (tour : BusTour)
  (h1 : tour.senior_price = 10)
  (h2 : tour.regular_price = 15)
  (h3 : tour.total_sales = 855)
  (h4 : tour.senior_tickets = 24) :
  total_tickets tour = 65 := by
  sorry

end NUMINAMATH_CALUDE_bus_tour_tickets_sold_l3382_338292


namespace NUMINAMATH_CALUDE_min_xy_value_l3382_338257

theorem min_xy_value (x y : ℝ) :
  (∃ (n : ℕ), n = 12) →
  1 + Real.cos (2 * x + 3 * y - 1) ^ 2 = (x^2 + y^2 + 2*(x+1)*(1-y)) / (x-y+1) →
  ∀ (z : ℝ), x * y ≥ 1/25 ∧ (∃ (a b : ℝ), a * b = 1/25) :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3382_338257


namespace NUMINAMATH_CALUDE_strawberry_picking_l3382_338286

/-- Given the total number of strawberries picked by three people, the number
picked by two of them together, and the number picked by one person alone,
prove the number picked by the other two together. -/
theorem strawberry_picking (total : ℕ) (matthew_and_zac : ℕ) (zac_alone : ℕ)
  (h1 : total = 550)
  (h2 : matthew_and_zac = 250)
  (h3 : zac_alone = 200) :
  total - zac_alone = 350 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picking_l3382_338286


namespace NUMINAMATH_CALUDE_fraction_value_in_system_l3382_338200

theorem fraction_value_in_system (a b x y : ℝ) (hb : b ≠ 0) 
  (eq1 : 4 * x - 2 * y = a) (eq2 : 5 * y - 10 * x = b) : a / b = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_in_system_l3382_338200


namespace NUMINAMATH_CALUDE_smallest_dual_base_number_l3382_338240

/-- Represents a number in a given base -/
def BaseRepresentation (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digit1 digit2 : ℕ), 
    n = digit1 * base + digit2 ∧
    digit1 < base ∧
    digit2 < base

/-- The smallest number representable in both base 6 and base 8 as AA and BB respectively -/
def SmallestDualBaseNumber : ℕ := 63

theorem smallest_dual_base_number :
  (BaseRepresentation SmallestDualBaseNumber 6) ∧
  (BaseRepresentation SmallestDualBaseNumber 8) ∧
  (∀ m : ℕ, m < SmallestDualBaseNumber →
    ¬(BaseRepresentation m 6 ∧ BaseRepresentation m 8)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_number_l3382_338240


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3382_338256

theorem rectangle_area_problem (x : ℝ) :
  x > 0 ∧
  (5 - (-1)) * (x - (-2)) = 66 →
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3382_338256


namespace NUMINAMATH_CALUDE_wage_decrease_hours_increase_l3382_338208

theorem wage_decrease_hours_increase 
  (original_wage : ℝ) 
  (original_hours : ℝ) 
  (wage_decrease_percent : ℝ) 
  (new_hours : ℝ) 
  (h1 : wage_decrease_percent = 20) 
  (h2 : original_wage > 0) 
  (h3 : original_hours > 0) 
  (h4 : new_hours > 0) 
  (h5 : original_wage * original_hours = (original_wage * (1 - wage_decrease_percent / 100)) * new_hours) :
  (new_hours - original_hours) / original_hours * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_wage_decrease_hours_increase_l3382_338208


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3382_338271

-- Define the inequality function
def inequality (m : ℝ) (x : ℝ) : Prop :=
  m * (x - 1) > x^2 - x

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_implies_m_value :
  ∀ m : ℝ, (∀ x : ℝ, inequality m x ↔ x ∈ solution_set m) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3382_338271


namespace NUMINAMATH_CALUDE_nonzero_term_count_correct_l3382_338232

/-- The number of nonzero terms in the expanded and simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def nonzeroTermCount : ℕ := 1010025

/-- The degree of the polynomial expression -/
def degree : ℕ := 2008

theorem nonzero_term_count_correct :
  nonzeroTermCount = (degree / 2 + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_nonzero_term_count_correct_l3382_338232


namespace NUMINAMATH_CALUDE_javier_ate_five_meat_ravioli_l3382_338282

/-- Represents the weight of each type of ravioli in ounces -/
structure RavioliWeights where
  meat : Float
  pumpkin : Float
  cheese : Float

/-- Represents the number of each type of ravioli eaten by Javier -/
structure JavierMeal where
  meat : Nat
  pumpkin : Nat
  cheese : Nat

/-- Calculates the total weight of Javier's meal -/
def mealWeight (weights : RavioliWeights) (meal : JavierMeal) : Float :=
  weights.meat * meal.meat.toFloat + weights.pumpkin * meal.pumpkin.toFloat + weights.cheese * meal.cheese.toFloat

/-- Theorem: Given the conditions, Javier ate 5 meat ravioli -/
theorem javier_ate_five_meat_ravioli (weights : RavioliWeights) (meal : JavierMeal) : 
  weights.meat = 1.5 ∧ 
  weights.pumpkin = 1.25 ∧ 
  weights.cheese = 1 ∧ 
  meal.pumpkin = 2 ∧ 
  meal.cheese = 4 ∧ 
  mealWeight weights meal = 15 → 
  meal.meat = 5 := by
  sorry


end NUMINAMATH_CALUDE_javier_ate_five_meat_ravioli_l3382_338282


namespace NUMINAMATH_CALUDE_september_reading_plan_l3382_338225

theorem september_reading_plan (total_pages : ℕ) (total_days : ℕ) (busy_days : ℕ) (flight_pages : ℕ) :
  total_pages = 600 →
  total_days = 30 →
  busy_days = 4 →
  flight_pages = 100 →
  ∃ (standard_pages : ℕ),
    standard_pages * (total_days - busy_days - 1) + flight_pages = total_pages ∧
    standard_pages = 20 := by
  sorry

end NUMINAMATH_CALUDE_september_reading_plan_l3382_338225


namespace NUMINAMATH_CALUDE_abc_inequality_l3382_338222

theorem abc_inequality : 
  let a : ℝ := Real.rpow 7 (1/3)
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3382_338222


namespace NUMINAMATH_CALUDE_solution_set_of_fraction_inequality_range_of_a_for_empty_solution_set_l3382_338215

-- Problem 1
theorem solution_set_of_fraction_inequality (x : ℝ) :
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 := by sorry

-- Problem 2
theorem range_of_a_for_empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*a*x + 4*a^2 + a > 0) → a > 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_fraction_inequality_range_of_a_for_empty_solution_set_l3382_338215


namespace NUMINAMATH_CALUDE_cos_150_plus_cos_neg_150_l3382_338299

theorem cos_150_plus_cos_neg_150 : Real.cos (150 * π / 180) + Real.cos (-150 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_plus_cos_neg_150_l3382_338299


namespace NUMINAMATH_CALUDE_triangle_third_angle_l3382_338291

theorem triangle_third_angle (a b c : ℝ) (ha : a = 40) (hb : b = 60) 
  (sum : a + b + c = 180) : c = 80 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l3382_338291


namespace NUMINAMATH_CALUDE_equation_solutions_l3382_338227

theorem equation_solutions :
  (∃ x : ℝ, (x + 2)^3 + 1 = 0 ↔ x = -3) ∧
  (∃ x : ℝ, (3*x - 2)^2 = 64 ↔ x = 10/3 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3382_338227


namespace NUMINAMATH_CALUDE_trig_ratios_on_line_l3382_338278

/-- Given an angle α whose terminal side lies on the line y = 2x, 
    prove its trigonometric ratios. -/
theorem trig_ratios_on_line (α : Real) : 
  (∃ k : Real, k ≠ 0 ∧ Real.cos α = k ∧ Real.sin α = 2 * k) → 
  (Real.sin α)^2 = 4/5 ∧ (Real.cos α)^2 = 1/5 ∧ Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratios_on_line_l3382_338278


namespace NUMINAMATH_CALUDE_property_P_for_given_numbers_l3382_338202

-- Define property P
def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3*x*y*z

-- Theorem statement
theorem property_P_for_given_numbers :
  (has_property_P 1) ∧
  (has_property_P 5) ∧
  (has_property_P 2014) ∧
  (¬ has_property_P 2013) :=
by sorry

end NUMINAMATH_CALUDE_property_P_for_given_numbers_l3382_338202


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_sum_l3382_338284

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  -- Add necessary fields here
  dummy : Unit

/-- Represents the smaller tetrahedron formed by the centers of the faces of a regular tetrahedron -/
def smaller_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

/-- The volume ratio of the smaller tetrahedron to the original tetrahedron -/
def volume_ratio (t : RegularTetrahedron) : ℚ :=
  sorry

/-- States that m and n are relatively prime positive integers -/
def are_relatively_prime (m n : ℕ) : Prop :=
  sorry

theorem tetrahedron_volume_ratio_sum (t : RegularTetrahedron) (m n : ℕ) :
  volume_ratio t = m / n →
  are_relatively_prime m n →
  m + n = 28 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_sum_l3382_338284


namespace NUMINAMATH_CALUDE_exam_marks_proof_l3382_338288

theorem exam_marks_proof (T : ℝ) 
  (h1 : 0.3 * T + 50 = 199.99999999999997) 
  (passing_mark : ℝ := 199.99999999999997) 
  (second_candidate_score : ℝ := 0.45 * T) : 
  second_candidate_score - passing_mark = 25 := by
sorry

end NUMINAMATH_CALUDE_exam_marks_proof_l3382_338288


namespace NUMINAMATH_CALUDE_horner_method_v3_l3382_338285

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_v3 (a b c d e f x : ℝ) : ℝ :=
  ((((a*x + b)*x + c)*x + d)*x + e)*x + f

theorem horner_method_v3 :
  horner_v3 2 0 (-3) 2 1 (-3) 2 = 12 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3382_338285


namespace NUMINAMATH_CALUDE_equation_solution_l3382_338223

theorem equation_solution (x : ℝ) :
  x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 1 →
  (3 * x + 2) / (x^2 + 5 * x + 6) = 3 * x / (x - 1) →
  3 * x^3 + 12 * x^2 + 19 * x + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3382_338223


namespace NUMINAMATH_CALUDE_andre_total_cost_l3382_338236

/-- Calculates the total cost of Andre's purchases including sales tax -/
def total_cost (treadmill_price : ℝ) (treadmill_discount : ℝ) 
                (plate_price : ℝ) (plate_discount : ℝ) (num_plates : ℕ)
                (sales_tax : ℝ) : ℝ :=
  let discounted_treadmill := treadmill_price * (1 - treadmill_discount)
  let discounted_plates := plate_price * num_plates * (1 - plate_discount)
  let subtotal := discounted_treadmill + discounted_plates
  subtotal * (1 + sales_tax)

/-- Theorem stating that Andre's total cost is $1120.29 -/
theorem andre_total_cost :
  total_cost 1350 0.30 60 0.15 2 0.07 = 1120.29 := by
  sorry

end NUMINAMATH_CALUDE_andre_total_cost_l3382_338236


namespace NUMINAMATH_CALUDE_param_line_point_l3382_338239

/-- A parameterized line in 2D space -/
structure ParamLine where
  /-- The vector on the line at parameter t -/
  vector : ℝ → ℝ × ℝ

/-- Theorem: Given a parameterized line with known points, we can determine another point -/
theorem param_line_point (l : ParamLine)
  (h1 : l.vector 5 = (2, 1))
  (h2 : l.vector 6 = (5, -7)) :
  l.vector 1 = (-40, 113) := by
  sorry

end NUMINAMATH_CALUDE_param_line_point_l3382_338239


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l3382_338273

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l3382_338273


namespace NUMINAMATH_CALUDE_product_of_reals_l3382_338297

theorem product_of_reals (a b : ℝ) 
  (sum_eq : a + b = 7)
  (sum_cubes_eq : a^3 + b^3 = 91) : 
  a * b = 12 := by sorry

end NUMINAMATH_CALUDE_product_of_reals_l3382_338297


namespace NUMINAMATH_CALUDE_cube_root_of_216_l3382_338294

theorem cube_root_of_216 (x : ℝ) (h : (x ^ (1/2)) ^ 3 = 216) : x = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_216_l3382_338294


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3382_338216

/-- Given a cylinder with volume 72π cm³, a cone with the same height and twice the radius
    of the cylinder has a volume of 96π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 72 * π → 
  (1/3 : ℝ) * π * (2*r)^2 * h = 96 * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3382_338216


namespace NUMINAMATH_CALUDE_shirt_price_shirt_price_is_33_l3382_338206

theorem shirt_price (pants_price : ℝ) (num_pants : ℕ) (num_shirts : ℕ) (total_payment : ℝ) (change : ℝ) : ℝ :=
  let total_spent := total_payment - change
  let pants_total := pants_price * num_pants
  let shirts_total := total_spent - pants_total
  shirts_total / num_shirts

#check shirt_price 54 2 4 250 10 = 33

-- The proof
theorem shirt_price_is_33 :
  shirt_price 54 2 4 250 10 = 33 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_shirt_price_is_33_l3382_338206


namespace NUMINAMATH_CALUDE_c_investment_is_10500_l3382_338224

/-- Represents the investment and profit distribution in a partnership business -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates C's investment given the partnership details -/
def calculate_c_investment (p : Partnership) : ℕ :=
  p.total_profit * p.a_investment / p.a_profit_share - p.a_investment - p.b_investment

/-- Theorem stating that C's investment is 10500 given the problem conditions -/
theorem c_investment_is_10500 (p : Partnership) 
  (h1 : p.a_investment = 6300)
  (h2 : p.b_investment = 4200)
  (h3 : p.total_profit = 12500)
  (h4 : p.a_profit_share = 3750) :
  calculate_c_investment p = 10500 := by
  sorry

#eval calculate_c_investment {
  a_investment := 6300, 
  b_investment := 4200, 
  c_investment := 0,  -- This value doesn't affect the calculation
  total_profit := 12500, 
  a_profit_share := 3750
}

end NUMINAMATH_CALUDE_c_investment_is_10500_l3382_338224


namespace NUMINAMATH_CALUDE_orange_face_probability_l3382_338264

theorem orange_face_probability (total_faces : ℕ) (orange_faces : ℕ) 
  (h1 : total_faces = 12) 
  (h2 : orange_faces = 4) : 
  (orange_faces : ℚ) / total_faces = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_face_probability_l3382_338264


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l3382_338290

/-- A quadratic function of the form y = (m-2)x^2 + 2x - 3 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 2 * x - 3

/-- The range of m for which the function is quadratic -/
theorem quadratic_function_m_range (m : ℝ) :
  (∃ (a : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), quadratic_function m x = a * x^2 + 2 * x - 3) ↔ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l3382_338290


namespace NUMINAMATH_CALUDE_median_line_equation_l3382_338212

/-- Triangle ABC with vertices A(1,0), B(2,-3), and C(3,3) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The median line on side AB of a triangle -/
def medianLineAB (t : Triangle) : LineEquation :=
  sorry

/-- Theorem: The equation of the median line on side AB of the given triangle is 3x - y - 6 = 0 -/
theorem median_line_equation (t : Triangle) 
  (h1 : t.A = (1, 0)) 
  (h2 : t.B = (2, -3)) 
  (h3 : t.C = (3, 3)) : 
  medianLineAB t = LineEquation.mk 3 (-1) (-6) :=
sorry

end NUMINAMATH_CALUDE_median_line_equation_l3382_338212


namespace NUMINAMATH_CALUDE_largest_remainder_209_l3382_338268

theorem largest_remainder_209 :
  (∀ n : ℕ, n < 120 → ∃ k r : ℕ, 209 = n * k + r ∧ r < n ∧ r ≤ 104) ∧
  (∃ n : ℕ, n < 120 ∧ ∃ k : ℕ, 209 = n * k + 104) ∧
  (∀ n : ℕ, n < 90 → ∃ k r : ℕ, 209 = n * k + r ∧ r < n ∧ r ≤ 69) ∧
  (∃ n : ℕ, n < 90 ∧ ∃ k : ℕ, 209 = n * k + 69) :=
by sorry

end NUMINAMATH_CALUDE_largest_remainder_209_l3382_338268


namespace NUMINAMATH_CALUDE_tangent_slope_sin_3x_l3382_338235

theorem tangent_slope_sin_3x (x : ℝ) (h : x = π / 3) : 
  HasDerivAt (fun x => Real.sin (3 * x)) (-3) x := by sorry

end NUMINAMATH_CALUDE_tangent_slope_sin_3x_l3382_338235


namespace NUMINAMATH_CALUDE_decimal_comparisons_l3382_338221

theorem decimal_comparisons :
  (9.38 > 3.98) ∧
  (0.62 > 0.23) ∧
  (2.5 > 2.05) ∧
  (53.6 > 5.36) ∧
  (9.42 > 9.377) := by
  sorry

end NUMINAMATH_CALUDE_decimal_comparisons_l3382_338221


namespace NUMINAMATH_CALUDE_project_savings_percentage_l3382_338289

theorem project_savings_percentage 
  (actual_investment : ℕ) 
  (savings : ℕ) 
  (h1 : actual_investment = 150000)
  (h2 : savings = 50000) :
  (savings : ℝ) / ((actual_investment : ℝ) + (savings : ℝ)) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_project_savings_percentage_l3382_338289


namespace NUMINAMATH_CALUDE_polynomial_value_at_five_l3382_338263

/-- Given a polynomial g(x) = ax^7 + bx^6 + cx - 3 where g(-5) = -3, prove that g(5) = 31250b - 3 -/
theorem polynomial_value_at_five (a b c : ℝ) :
  let g : ℝ → ℝ := λ x => a * x^7 + b * x^6 + c * x - 3
  g (-5) = -3 →
  g 5 = 31250 * b - 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_at_five_l3382_338263


namespace NUMINAMATH_CALUDE_power_of_fraction_to_decimal_l3382_338238

theorem power_of_fraction_to_decimal :
  (4 / 5 : ℚ) ^ 3 = 512 / 1000 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_to_decimal_l3382_338238


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3382_338209

theorem sqrt_sum_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3382_338209


namespace NUMINAMATH_CALUDE_shopping_ratio_l3382_338218

theorem shopping_ratio (emma_spent elsa_spent elizabeth_spent total_spent : ℚ) : 
  emma_spent = 58 →
  elizabeth_spent = 4 * elsa_spent →
  total_spent = 638 →
  emma_spent + elsa_spent + elizabeth_spent = total_spent →
  elsa_spent / emma_spent = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_shopping_ratio_l3382_338218


namespace NUMINAMATH_CALUDE_sum_of_first_40_digits_eq_72_l3382_338205

/-- The sum of the first 40 digits after the decimal point in the decimal representation of 1/2222 -/
def sum_of_first_40_digits : ℕ :=
  -- Define the sum here
  72

/-- Theorem stating that the sum of the first 40 digits after the decimal point
    in the decimal representation of 1/2222 is equal to 72 -/
theorem sum_of_first_40_digits_eq_72 :
  sum_of_first_40_digits = 72 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sum_of_first_40_digits_eq_72_l3382_338205


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_area_l3382_338245

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - x*y + x - 4*y = 12

/-- Point A is on the y-axis and satisfies the ellipse equation -/
def point_A : ℝ × ℝ := sorry

/-- Point C is on the y-axis and satisfies the ellipse equation -/
def point_C : ℝ × ℝ := sorry

/-- Point B is on the x-axis and satisfies the ellipse equation -/
def point_B : ℝ × ℝ := sorry

/-- Point D is on the x-axis and satisfies the ellipse equation -/
def point_D : ℝ × ℝ := sorry

/-- The area of the inscribed quadrilateral ABCD -/
def area_ABCD : ℝ := sorry

theorem inscribed_quadrilateral_area :
  ellipse_equation point_A.1 point_A.2 ∧
  ellipse_equation point_B.1 point_B.2 ∧
  ellipse_equation point_C.1 point_C.2 ∧
  ellipse_equation point_D.1 point_D.2 ∧
  point_A.1 = 0 ∧ point_C.1 = 0 ∧
  point_B.2 = 0 ∧ point_D.2 = 0 →
  area_ABCD = 28 := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_area_l3382_338245


namespace NUMINAMATH_CALUDE_point_c_coordinates_l3382_338272

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The distance between two points -/
def distance (p q : Point) : ℚ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt

/-- Check if a point is on a line segment -/
def isOnSegment (p q r : Point) : Prop :=
  distance p r + distance r q = distance p q

theorem point_c_coordinates :
  let a : Point := ⟨-2, 1⟩
  let b : Point := ⟨4, 9⟩
  let c : Point := ⟨22/7, 55/7⟩
  isOnSegment a c b ∧ distance a c = 4 * distance c b → c = ⟨22/7, 55/7⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l3382_338272


namespace NUMINAMATH_CALUDE_coupon_usage_theorem_l3382_338253

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

def isCouponDay (startDay : DayOfWeek) (n : Nat) : Prop :=
  ∃ k : Nat, k < 8 ∧ advanceDays startDay (7 * k) = DayOfWeek.Monday

theorem coupon_usage_theorem (startDay : DayOfWeek) :
  startDay = DayOfWeek.Sunday ↔
    ¬(isCouponDay startDay 8) ∧
    ∀ d : DayOfWeek, d ≠ DayOfWeek.Sunday → isCouponDay d 8 :=
by sorry

end NUMINAMATH_CALUDE_coupon_usage_theorem_l3382_338253


namespace NUMINAMATH_CALUDE_cake_mix_distribution_l3382_338274

theorem cake_mix_distribution (tray1 tray2 : ℕ) : 
  tray2 = tray1 - 20 → 
  tray1 + tray2 = 500 → 
  tray1 = 260 := by
sorry

end NUMINAMATH_CALUDE_cake_mix_distribution_l3382_338274


namespace NUMINAMATH_CALUDE_adeline_work_hours_l3382_338244

/-- Adeline's work schedule and earnings problem -/
theorem adeline_work_hours
  (hourly_rate : ℕ)
  (days_per_week : ℕ)
  (total_earnings : ℕ)
  (total_weeks : ℕ)
  (h1 : hourly_rate = 12)
  (h2 : days_per_week = 5)
  (h3 : total_earnings = 3780)
  (h4 : total_weeks = 7) :
  total_earnings / (total_weeks * days_per_week * hourly_rate) = 9 :=
by sorry

end NUMINAMATH_CALUDE_adeline_work_hours_l3382_338244


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3382_338229

/-- Given a line with equation 3x - 6y = 21, prove that any parallel line has slope 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 21) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = (1 : ℝ) / 2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3382_338229


namespace NUMINAMATH_CALUDE_skateboard_travel_distance_l3382_338262

/-- Represents the distance traveled by a skateboard in a given number of seconds -/
def skateboardDistance (initialDistance : ℕ) (firstAcceleration : ℕ) (secondAcceleration : ℕ) (totalSeconds : ℕ) : ℕ :=
  let firstPeriodDistance := (5 : ℕ) * (2 * initialDistance + 4 * firstAcceleration) / 2
  let secondPeriodInitialDistance := initialDistance + 5 * firstAcceleration
  let secondPeriodDistance := (5 : ℕ) * (2 * secondPeriodInitialDistance + 4 * secondAcceleration) / 2
  firstPeriodDistance + secondPeriodDistance

theorem skateboard_travel_distance :
  skateboardDistance 8 6 9 10 = 380 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_travel_distance_l3382_338262


namespace NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l3382_338277

def total_products : ℕ := 10

-- Define the event A
def event_A (defective : ℕ) : Prop := defective ≥ 2 ∧ defective ≤ total_products

-- Define the complementary event of A
def complement_A (defective : ℕ) : Prop := defective ≤ 1

-- Theorem statement
theorem complement_of_at_least_two_defective :
  ∀ (defective : ℕ), defective ≤ total_products →
  (¬ event_A defective ↔ complement_A defective) :=
sorry

end NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l3382_338277


namespace NUMINAMATH_CALUDE_triple_hash_twenty_l3382_338255

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_hash_twenty : hash (hash (hash 20)) = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_twenty_l3382_338255


namespace NUMINAMATH_CALUDE_sqrt_square_eq_x_for_nonneg_l3382_338270

theorem sqrt_square_eq_x_for_nonneg (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_x_for_nonneg_l3382_338270


namespace NUMINAMATH_CALUDE_product_and_quotient_of_geometric_sequences_l3382_338242

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem product_and_quotient_of_geometric_sequences
  (a b : ℕ → ℝ)
  (ha : is_geometric_sequence a)
  (hb : is_geometric_sequence b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  is_geometric_sequence (λ n => a n * b n) ∧
  is_geometric_sequence (λ n => a n / b n) :=
sorry

end NUMINAMATH_CALUDE_product_and_quotient_of_geometric_sequences_l3382_338242


namespace NUMINAMATH_CALUDE_number_calculation_l3382_338266

theorem number_calculation (n : ℚ) : (2/5 : ℚ) * n = 36 → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3382_338266


namespace NUMINAMATH_CALUDE_statue_original_cost_l3382_338260

/-- Proves that if a statue is sold for $540 with a 35% profit, then its original cost was $400. -/
theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 540 →
  profit_percentage = 0.35 →
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 400 := by
sorry

end NUMINAMATH_CALUDE_statue_original_cost_l3382_338260


namespace NUMINAMATH_CALUDE_laurent_series_expansion_l3382_338228

/-- Laurent series expansion of f(z) = 2ia / (z^2 + a^2) in the region 0 < |z - ia| < a -/
theorem laurent_series_expansion
  (a : ℝ) (z : ℂ) (ha : a > 0) (hz : 0 < Complex.abs (z - Complex.I * a) ∧ Complex.abs (z - Complex.I * a) < a) :
  (2 * Complex.I * a) / (z^2 + a^2) =
    1 / (z - Complex.I * a) - ∑' k, (z - Complex.I * a)^k / (Complex.I * a)^(k + 1) :=
by sorry

end NUMINAMATH_CALUDE_laurent_series_expansion_l3382_338228


namespace NUMINAMATH_CALUDE_identify_fake_coin_in_two_weighings_l3382_338207

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a coin -/
inductive Coin
  | A : Coin
  | B : Coin
  | C : Coin
  | D : Coin

/-- Represents a weighing operation -/
def weigh (left right : List Coin) : WeighResult :=
  sorry

/-- Represents the process of identifying the fake coin -/
def identifyFakeCoin : Coin :=
  sorry

/-- Theorem stating that it's possible to identify the fake coin in two weighings -/
theorem identify_fake_coin_in_two_weighings :
  ∃ (fakeCoin : Coin),
    (∀ (c : Coin), c ≠ fakeCoin → (weigh [c] [fakeCoin] = WeighResult.Equal ↔ c = fakeCoin)) →
    identifyFakeCoin = fakeCoin :=
  sorry

end NUMINAMATH_CALUDE_identify_fake_coin_in_two_weighings_l3382_338207


namespace NUMINAMATH_CALUDE_line_equation_l3382_338296

/-- Given a line with slope 2 and y-intercept -3, its equation is 2x - y - 3 = 0 -/
theorem line_equation (x y : ℝ) :
  (∃ (m b : ℝ), m = 2 ∧ b = -3 ∧ y = m * x + b) →
  2 * x - y - 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3382_338296


namespace NUMINAMATH_CALUDE_not_all_equilateral_triangles_congruent_l3382_338269

/-- An equilateral triangle is a triangle where all three sides are of equal length. -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two triangles are congruent if they have the same size and shape. -/
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = t2.side_length

theorem not_all_equilateral_triangles_congruent :
  ∃ t1 t2 : EquilateralTriangle, ¬(congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_not_all_equilateral_triangles_congruent_l3382_338269


namespace NUMINAMATH_CALUDE_largest_number_is_nineteen_l3382_338281

theorem largest_number_is_nineteen 
  (a b c : ℕ) 
  (sum_ab : a + b = 16)
  (sum_ac : a + c = 20)
  (sum_bc : b + c = 23) :
  max a (max b c) = 19 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_is_nineteen_l3382_338281


namespace NUMINAMATH_CALUDE_initial_volume_of_solution_l3382_338259

/-- Given a solution with initial volume V, prove that V = 40 liters -/
theorem initial_volume_of_solution (V : ℝ) : 
  (0.05 * V + 3.5 = 0.11 * (V + 10)) → V = 40 := by sorry

end NUMINAMATH_CALUDE_initial_volume_of_solution_l3382_338259


namespace NUMINAMATH_CALUDE_ruffy_is_nine_l3382_338217

/-- Ruffy's current age -/
def ruffy_age : ℕ := 9

/-- Orlie's current age -/
def orlie_age : ℕ := 12

/-- Relation between Ruffy's and Orlie's current ages -/
axiom current_age_relation : ruffy_age = (3 * orlie_age) / 4

/-- Relation between Ruffy's and Orlie's ages four years ago -/
axiom past_age_relation : ruffy_age - 4 = (orlie_age - 4) / 2 + 1

/-- Theorem: Ruffy's current age is 9 years -/
theorem ruffy_is_nine : ruffy_age = 9 := by sorry

end NUMINAMATH_CALUDE_ruffy_is_nine_l3382_338217


namespace NUMINAMATH_CALUDE_total_material_ordered_l3382_338247

def concrete : ℝ := 0.17
def bricks : ℝ := 0.17
def stone : ℝ := 0.5

theorem total_material_ordered : concrete + bricks + stone = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_total_material_ordered_l3382_338247
