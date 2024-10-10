import Mathlib

namespace system_solution_existence_l1983_198323

/-- The system of equations has at least one solution for some 'a' if and only if 'b' is in [-11, 2) -/
theorem system_solution_existence (b : ℝ) : 
  (∃ a x y : ℝ, x^2 + y^2 + 2*b*(b - x + y) = 4 ∧ y = 9 / ((x + a)^2 + 1)) ↔ 
  -11 ≤ b ∧ b < 2 := by sorry

end system_solution_existence_l1983_198323


namespace total_go_stones_l1983_198341

theorem total_go_stones (white_stones black_stones : ℕ) : 
  white_stones = 954 →
  white_stones = black_stones + 468 →
  white_stones + black_stones = 1440 :=
by
  sorry

end total_go_stones_l1983_198341


namespace smallest_triangle_angle_function_range_l1983_198368

theorem smallest_triangle_angle_function_range :
  ∀ x : Real,
  0 < x → x ≤ Real.pi / 3 →
  let y := (Real.sin x * Real.cos x + 1) / (Real.sin x + Real.cos x)
  ∃ (a b : Real), a = 3/2 ∧ b = 3 * Real.sqrt 2 / 4 ∧
  (∀ z, y = z → a < z ∧ z ≤ b) ∧
  (∀ ε > 0, ∃ z, y = z ∧ z < a + ε) ∧
  (∃ z, y = z ∧ z = b) :=
by sorry

end smallest_triangle_angle_function_range_l1983_198368


namespace polygon_side_theorem_l1983_198310

def polygon_side_proof (total_area : ℝ) (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) (unknown_side_min unknown_side_max : ℝ) : Prop :=
  let rect1_area := rect1_length * rect1_width
  let rect2_area := rect2_length * rect2_width
  let unknown_rect_area := total_area - rect1_area - rect2_area
  ∃ (x : ℝ), (x = 7 ∨ x = 6) ∧ 
             unknown_rect_area = x * (unknown_rect_area / x) ∧
             x > unknown_side_min ∧ x < unknown_side_max

theorem polygon_side_theorem : 
  polygon_side_proof 72 10 1 5 4 5 10 := by
  sorry

end polygon_side_theorem_l1983_198310


namespace square_difference_div_product_equals_four_l1983_198321

theorem square_difference_div_product_equals_four :
  ((0.137 + 0.098)^2 - (0.137 - 0.098)^2) / (0.137 * 0.098) = 4 := by
  sorry

end square_difference_div_product_equals_four_l1983_198321


namespace servant_served_nine_months_l1983_198344

/-- Represents the employment contract and service details of a servant -/
structure ServantContract where
  fullYearSalary : ℕ  -- Salary for a full year in rupees
  uniformPrice : ℕ    -- Price of the uniform in rupees
  receivedSalary : ℕ  -- Salary actually received in rupees
  fullYearMonths : ℕ  -- Number of months in a full year

/-- Calculates the number of months served by the servant -/
def monthsServed (contract : ServantContract) : ℕ :=
  (contract.receivedSalary + contract.uniformPrice) * contract.fullYearMonths 
    / (contract.fullYearSalary + contract.uniformPrice)

/-- Theorem stating that the servant served for 9 months -/
theorem servant_served_nine_months :
  let contract : ServantContract := {
    fullYearSalary := 500,
    uniformPrice := 500,
    receivedSalary := 250,
    fullYearMonths := 12
  }
  monthsServed contract = 9 := by sorry

end servant_served_nine_months_l1983_198344


namespace max_value_of_largest_integer_l1983_198380

theorem max_value_of_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 60 →
  e.val - a.val = 10 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  e.val ≤ 290 :=
by sorry

end max_value_of_largest_integer_l1983_198380


namespace problem_solution_l1983_198373

/-- The set A as defined in the problem -/
def A : Set ℝ := {x | 12 - 5*x - 2*x^2 > 0}

/-- The set B as defined in the problem -/
def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b ≤ 0}

/-- The theorem statement -/
theorem problem_solution :
  ∃ (a b : ℝ),
    (A ∩ B a b = ∅) ∧
    (A ∪ B a b = Set.Ioo (-4) 8) ∧
    (a = 19/2) ∧
    (b = 12) := by
  sorry

end problem_solution_l1983_198373


namespace square_sequence_properties_l1983_198398

/-- A quadratic sequence of unit squares -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The theorem stating the properties of the sequence -/
theorem square_sequence_properties :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 ∧ f 150 = 67951 := by
  sorry

#check square_sequence_properties

end square_sequence_properties_l1983_198398


namespace value_of_x_l1983_198392

theorem value_of_x (w y z : ℚ) (h1 : w = 45) (h2 : z = 2 * w) (h3 : y = (1 / 6) * z) : (1 / 3) * y = 5 := by
  sorry

end value_of_x_l1983_198392


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l1983_198342

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 3*x + 2 - 12
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 3) :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l1983_198342


namespace average_equation_solution_l1983_198306

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 8) + (5*x + 3) + (3*x + 9)) = 5*x - 10 → x = 10 := by
sorry

end average_equation_solution_l1983_198306


namespace point_b_coordinates_l1983_198334

/-- Given a circle with center (0,0) and radius 2, points A(2,2) and B(a,b),
    if for any point P on the circle, |PA|/|PB| = √2, then B = (1,1) -/
theorem point_b_coordinates (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → 
    ((x - 2)^2 + (y - 2)^2) / ((x - a)^2 + (y - b)^2) = 2) → 
  a = 1 ∧ b = 1 := by sorry

end point_b_coordinates_l1983_198334


namespace tommys_profit_l1983_198358

/-- Represents a type of crate --/
structure Crate where
  capacity : ℕ
  quantity : ℕ
  cost : ℕ
  rotten : ℕ
  price : ℕ

/-- Calculates the profit from selling tomatoes --/
def calculateProfit (crateA crateB crateC : Crate) : ℕ :=
  let totalCost := crateA.cost + crateB.cost + crateC.cost
  let revenueA := (crateA.capacity * crateA.quantity - crateA.rotten) * crateA.price
  let revenueB := (crateB.capacity * crateB.quantity - crateB.rotten) * crateB.price
  let revenueC := (crateC.capacity * crateC.quantity - crateC.rotten) * crateC.price
  let totalRevenue := revenueA + revenueB + revenueC
  totalRevenue - totalCost

/-- Tommy's profit from selling tomatoes is $14 --/
theorem tommys_profit :
  let crateA : Crate := ⟨20, 2, 220, 4, 5⟩
  let crateB : Crate := ⟨25, 3, 375, 5, 6⟩
  let crateC : Crate := ⟨30, 1, 180, 3, 7⟩
  calculateProfit crateA crateB crateC = 14 := by
  sorry


end tommys_profit_l1983_198358


namespace consecutive_integers_around_sqrt3_l1983_198376

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end consecutive_integers_around_sqrt3_l1983_198376


namespace joan_picked_37_oranges_l1983_198309

/-- The number of oranges picked by Sara -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := 47

/-- The number of oranges picked by Joan -/
def joan_oranges : ℕ := total_oranges - sara_oranges

theorem joan_picked_37_oranges : joan_oranges = 37 := by
  sorry

end joan_picked_37_oranges_l1983_198309


namespace decagon_area_ratio_l1983_198352

theorem decagon_area_ratio (decagon_area : ℝ) (below_PQ_square_area : ℝ) (triangle_base : ℝ) (XQ QY : ℝ) :
  decagon_area = 12 →
  below_PQ_square_area = 1 →
  triangle_base = 6 →
  XQ + QY = 6 →
  (decagon_area / 2 = below_PQ_square_area + (1/2 * triangle_base * ((decagon_area / 2) - below_PQ_square_area) / triangle_base)) →
  XQ / QY = 2 := by
  sorry

end decagon_area_ratio_l1983_198352


namespace base_equation_solution_l1983_198369

/-- Represents a number in a given base -/
def toBase (n : ℕ) (base : ℕ) : ℕ → ℕ 
| 0 => 0
| (d+1) => (toBase n base d) * base + n % base

/-- The main theorem -/
theorem base_equation_solution (A B : ℕ) (h1 : B = A + 2) 
  (h2 : toBase 216 A 3 + toBase 52 B 2 = toBase 75 (A + B + 1) 2) : 
  A + B + 1 = 15 := by
  sorry

#eval toBase 216 6 3  -- Should output 90
#eval toBase 52 8 2   -- Should output 42
#eval toBase 75 15 2  -- Should output 132

end base_equation_solution_l1983_198369


namespace square_side_lengths_average_l1983_198327

theorem square_side_lengths_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end square_side_lengths_average_l1983_198327


namespace trader_profit_l1983_198305

theorem trader_profit (C : ℝ) (C_pos : C > 0) : 
  let markup := 0.12
  let discount := 0.09821428571428571
  let marked_price := C * (1 + markup)
  let final_price := marked_price * (1 - discount)
  (final_price - C) / C = 0.01 := by
sorry

end trader_profit_l1983_198305


namespace function_inequality_l1983_198350

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -2 * Real.log x + a / x^2 + 1

theorem function_inequality (a : ℝ) (x₁ x₂ x₀ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₀ : x₀ > 0)
  (hz₁ : f a x₁ = 0) (hz₂ : f a x₂ = 0) (hx₁₂ : x₁ ≠ x₂)
  (hextremum : ∀ x > 0, f a x₀ ≥ f a x) :
  1 / x₁^2 + 1 / x₂^2 > 2 * f a x₀ :=
sorry

end function_inequality_l1983_198350


namespace tomato_field_area_l1983_198371

theorem tomato_field_area (length : ℝ) (width : ℝ) (tomato_area : ℝ) : 
  length = 3.6 →
  width = 2.5 * length →
  tomato_area = (length * width) / 2 →
  tomato_area = 16.2 := by
  sorry

end tomato_field_area_l1983_198371


namespace specific_convention_handshakes_l1983_198319

/-- The number of handshakes in a convention with multiple companies -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes in the specific convention scenario -/
theorem specific_convention_handshakes :
  convention_handshakes 5 5 = 250 := by
  sorry

#eval convention_handshakes 5 5

end specific_convention_handshakes_l1983_198319


namespace four_balls_three_boxes_l1983_198346

/-- The number of ways to put n different balls into k boxes -/
def ways_to_put_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: Putting 4 different balls into 3 boxes results in 81 different ways -/
theorem four_balls_three_boxes : ways_to_put_balls 4 3 = 81 := by
  sorry

end four_balls_three_boxes_l1983_198346


namespace cube_root_nested_expression_l1983_198364

theorem cube_root_nested_expression : 
  (2 * (2 * 8^(1/3))^(1/3))^(1/3) = 2^(5/9) := by sorry

end cube_root_nested_expression_l1983_198364


namespace xyz_sum_l1983_198388

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : x * z + y = 47) : 
  x + y + z = 48 := by
sorry

end xyz_sum_l1983_198388


namespace machinery_expense_l1983_198391

/-- Proves that the amount spent on machinery is $1000 --/
theorem machinery_expense (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 5714.29 →
  raw_materials = 3000 →
  cash_percentage = 0.30 →
  ∃ (machinery : ℝ),
    machinery = 1000 ∧
    total = raw_materials + machinery + (cash_percentage * total) :=
by
  sorry


end machinery_expense_l1983_198391


namespace reverse_clock_theorem_l1983_198326

/-- Represents a clock with a reverse-moving minute hand -/
structure ReverseClock :=
  (hour : ℝ)
  (minute : ℝ)

/-- Converts a ReverseClock time to a standard clock time -/
def to_standard_time (c : ReverseClock) : ℝ := sorry

/-- Checks if the hands of a ReverseClock coincide -/
def hands_coincide (c : ReverseClock) : Prop := sorry

theorem reverse_clock_theorem :
  ∀ (c : ReverseClock),
    4 < c.hour ∧ c.hour < 5 →
    hands_coincide c →
    to_standard_time c = 4 + 36 / 60 + 12 / (13 * 60) :=
by sorry

end reverse_clock_theorem_l1983_198326


namespace ellipse_foci_distance_l1983_198366

/-- The distance between the foci of the ellipse x^2 + 9y^2 = 324 is 24√2 -/
theorem ellipse_foci_distance : 
  let ellipse_equation := fun (x y : ℝ) => x^2 + 9*y^2 = 324
  ∃ f₁ f₂ : ℝ × ℝ, 
    (∀ x y, ellipse_equation x y → ((x - f₁.1)^2 + (y - f₁.2)^2) + ((x - f₂.1)^2 + (y - f₂.2)^2) = 2 * 324) ∧ 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (24 * Real.sqrt 2)^2 :=
by sorry

end ellipse_foci_distance_l1983_198366


namespace coefficient_sum_after_shift_l1983_198313

def original_function (x : ℝ) : ℝ := 2 * x^2 - x + 7

def shifted_function (x : ℝ) : ℝ := original_function (x - 4)

def quadratic_form (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem coefficient_sum_after_shift :
  ∃ (a b c : ℝ), (∀ x, shifted_function x = quadratic_form a b c x) ∧ a + b + c = 28 := by
  sorry

end coefficient_sum_after_shift_l1983_198313


namespace white_pairs_coincide_l1983_198357

/-- Represents the number of triangles of each color in each half of the figure -/
structure HalfFigure where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

/-- The main theorem stating that 5 white pairs coincide -/
theorem white_pairs_coincide (half : HalfFigure) (pairs : CoincidingPairs) :
  half.red = 4 ∧ half.blue = 6 ∧ half.white = 10 ∧
  pairs.red_red = 3 ∧ pairs.blue_blue = 4 ∧ pairs.red_white = 3 →
  pairs.white_white = 5 := by
  sorry

end white_pairs_coincide_l1983_198357


namespace larger_number_problem_l1983_198389

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 15) : L = 1635 := by
  sorry

end larger_number_problem_l1983_198389


namespace integer_factorization_l1983_198362

theorem integer_factorization (a b c d : ℤ) (h : a * b = c * d) :
  ∃ (w x y z : ℤ), a = w * x ∧ b = y * z ∧ c = w * y ∧ d = x * z := by
  sorry

end integer_factorization_l1983_198362


namespace inscribed_circle_radius_rhombus_l1983_198396

theorem inscribed_circle_radius_rhombus (d₁ d₂ : ℝ) (h₁ : d₁ = 8) (h₂ : d₂ = 30) :
  let a := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  let r := (d₁ * d₂) / (8 * a)
  r = 30 / Real.sqrt 241 := by sorry

end inscribed_circle_radius_rhombus_l1983_198396


namespace perpendicular_lines_a_values_l1983_198361

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∀ x y : ℝ, (2*a + 5)*x + (a - 2)*y + 4 = 0 ∧ (2 - a)*x + (a + 3)*y - 1 = 0 → 
    ((2*a + 5)*(2 - a) + (a - 2)*(a + 3) = 0)) → 
  (a = 2 ∨ a = -2) :=
by sorry

end perpendicular_lines_a_values_l1983_198361


namespace complex_number_quadrant_z_in_second_quadrant_l1983_198337

theorem complex_number_quadrant : Complex → Prop :=
  fun z => ∃ (a b : ℝ), z = Complex.mk a b ∧ a < 0 ∧ b > 0

def i : Complex := Complex.I

def z : Complex := (1 + 2 * i) * i

theorem z_in_second_quadrant : complex_number_quadrant z := by
  sorry

end complex_number_quadrant_z_in_second_quadrant_l1983_198337


namespace kite_side_lengths_l1983_198399

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length. -/
structure Kite where
  a : ℝ  -- First diagonal
  b : ℝ  -- Second diagonal
  k : ℝ  -- Half of the perimeter
  x : ℝ  -- Length of one side
  y : ℝ  -- Length of the other side

/-- Properties of the kite based on the given conditions -/
def kite_properties (q : Kite) : Prop :=
  q.a = 6 ∧ q.b = 25/4 ∧ q.k = 35/4 ∧ q.x + q.y = q.k

/-- The theorem stating the side lengths of the kite -/
theorem kite_side_lengths (q : Kite) (h : kite_properties q) :
  q.x = 5 ∧ q.y = 15/4 :=
sorry

end kite_side_lengths_l1983_198399


namespace fraction_addition_l1983_198367

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l1983_198367


namespace gas_station_candy_boxes_l1983_198372

theorem gas_station_candy_boxes : 
  let chocolate_boxes : ℕ := 2
  let sugar_boxes : ℕ := 5
  let gum_boxes : ℕ := 2
  chocolate_boxes + sugar_boxes + gum_boxes = 9 :=
by
  sorry

end gas_station_candy_boxes_l1983_198372


namespace painters_rooms_theorem_l1983_198375

/-- Given that 3 painters can complete 3 rooms in 3 hours, 
    prove that 9 painters can complete 27 rooms in 9 hours. -/
theorem painters_rooms_theorem (painters_rate : ℕ → ℕ → ℕ → ℕ) 
  (h : painters_rate 3 3 3 = 3) : painters_rate 9 9 9 = 27 := by
  sorry

end painters_rooms_theorem_l1983_198375


namespace solution_count_l1983_198386

theorem solution_count (S : Finset ℝ) (p : ℝ) : 
  S.card = 12 → p = 1/6 → ∃ n : ℕ, n = 2 ∧ n = (S.card : ℝ) * p := by
  sorry

end solution_count_l1983_198386


namespace rational_sequence_to_integer_l1983_198303

theorem rational_sequence_to_integer (x : ℚ) : 
  ∃ (f : ℕ → ℚ), 
    f 0 = x ∧ 
    (∀ n : ℕ, n ≥ 1 → (f n = 2 * f (n - 1) ∨ f n = 2 * f (n - 1) + 1 / n)) ∧
    (∃ k : ℕ, ∃ m : ℤ, f k = m) := by
  sorry

end rational_sequence_to_integer_l1983_198303


namespace cos_cubed_minus_sin_cubed_l1983_198349

theorem cos_cubed_minus_sin_cubed (θ : ℝ) :
  Real.cos θ ^ 3 - Real.sin θ ^ 3 = (Real.cos θ - Real.sin θ) * (1 + Real.cos θ * Real.sin θ) := by
  sorry

end cos_cubed_minus_sin_cubed_l1983_198349


namespace original_cost_price_l1983_198317

/-- Calculates the original cost price given a series of transactions and the final price --/
theorem original_cost_price 
  (profit_ab profit_bc discount_cd profit_de final_price : ℝ) :
  let original_price := 
    final_price / ((1 + profit_ab/100) * (1 + profit_bc/100) * (1 - discount_cd/100) * (1 + profit_de/100))
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    (profit_ab = 20 ∧ 
     profit_bc = 25 ∧ 
     discount_cd = 15 ∧ 
     profit_de = 30 ∧ 
     final_price = 289.1) →
    (142.8 - ε ≤ original_price ∧ original_price ≤ 142.8 + ε) :=
by sorry

end original_cost_price_l1983_198317


namespace triangle_inequality_third_stick_length_l1983_198316

theorem triangle_inequality (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c → (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (a < b + c ∧ b < c + a ∧ c < a + b) :=
sorry

theorem third_stick_length (a b : ℝ) (ha : a = 20) (hb : b = 30) :
  ∃ c, c = 30 ∧ 
       (a + b > c ∧ b + c > a ∧ c + a > b) ∧
       ¬(a + b > 10 ∧ b + 10 > a ∧ 10 + a > b) ∧
       ¬(a + b > 50 ∧ b + 50 > a ∧ 50 + a > b) ∧
       ¬(a + b > 70 ∧ b + 70 > a ∧ 70 + a > b) :=
sorry

end triangle_inequality_third_stick_length_l1983_198316


namespace trigonometric_identities_l1983_198314

theorem trigonometric_identities (α : Real) (h : Real.tan α = 3) :
  ((Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6/11) ∧
  (Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 6) := by
  sorry

end trigonometric_identities_l1983_198314


namespace original_number_proof_l1983_198324

theorem original_number_proof : ∃ (n : ℕ), n ≥ 129 ∧ (n - 30) % 99 = 0 ∧ ∀ (m : ℕ), m < 129 → (m - 30) % 99 ≠ 0 :=
by sorry

end original_number_proof_l1983_198324


namespace intersection_of_A_and_B_l1983_198385

def A : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l1983_198385


namespace unique_six_digit_number_l1983_198395

/-- A six-digit number is between 100000 and 999999 -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- Function to reduce the first digit of a number by 3 and append 3 at the end -/
def transform (n : ℕ) : ℕ := (n - 300000) * 10 + 3

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_six_digit n ∧ 3 * n = transform n ∧ n = 428571 := by sorry

end unique_six_digit_number_l1983_198395


namespace count_odd_sum_numbers_l1983_198307

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A function that checks if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function that returns the sum of digits of a three-digit number -/
def digitSum (n : Nat) : Nat :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- The set of all three-digit numbers formed by the given digits without repetition -/
def threeDigitNumbers : Finset Nat :=
  Finset.filter (fun n => n ≥ 100 ∧ n < 1000 ∧ (Finset.card (Finset.filter (fun d => d ∈ digits) (Finset.range 10))) = 3) (Finset.range 1000)

theorem count_odd_sum_numbers :
  Finset.card (Finset.filter (fun n => isOdd (digitSum n)) threeDigitNumbers) = 24 := by sorry

end count_odd_sum_numbers_l1983_198307


namespace certain_number_proof_l1983_198338

theorem certain_number_proof (h : 2994 / 14.5 = 171) : 
  ∃ x : ℝ, x / 1.45 = 17.1 ∧ x = 24.795 := by
sorry

end certain_number_proof_l1983_198338


namespace bowling_ball_weighs_16_pounds_l1983_198322

/-- The weight of a single bowling ball in pounds. -/
def bowling_ball_weight : ℝ := sorry

/-- The weight of a single canoe in pounds. -/
def canoe_weight : ℝ := sorry

/-- Theorem stating that a bowling ball weighs 16 pounds under given conditions. -/
theorem bowling_ball_weighs_16_pounds : bowling_ball_weight = 16 := by
  have h1 : 8 * bowling_ball_weight = 4 * canoe_weight := by sorry
  have h2 : 2 * canoe_weight = 64 := by sorry
  sorry

end bowling_ball_weighs_16_pounds_l1983_198322


namespace two_number_problem_l1983_198383

theorem two_number_problem (A B n : ℕ) : 
  B > 0 → 
  A > B → 
  A = 10 * B + n → 
  0 ≤ n → 
  n ≤ 9 → 
  A + B = 2022 → 
  A = 1839 ∧ B = 183 := by
sorry

end two_number_problem_l1983_198383


namespace rabbit_count_l1983_198308

/-- Calculates the number of rabbits given land dimensions and clearing rates -/
theorem rabbit_count (land_width : ℝ) (land_length : ℝ) (rabbit_clear_rate : ℝ) (days_to_clear : ℝ) : 
  land_width = 200 ∧ 
  land_length = 900 ∧ 
  rabbit_clear_rate = 10 ∧ 
  days_to_clear = 20 → 
  (land_width * land_length) / 9 / (rabbit_clear_rate * days_to_clear) = 100 := by
  sorry

#check rabbit_count

end rabbit_count_l1983_198308


namespace algebraic_expression_value_l1983_198360

theorem algebraic_expression_value (a b c : ℝ) : 
  (∀ x, (x - 1) * (x + 2) = a * x^2 + b * x + c) → 
  4 * a - 2 * b + c = 0 := by
sorry

end algebraic_expression_value_l1983_198360


namespace increase_by_fifty_percent_l1983_198363

theorem increase_by_fifty_percent : 
  let initial : ℝ := 100
  let percentage : ℝ := 50
  let increase : ℝ := initial * (percentage / 100)
  let final : ℝ := initial + increase
  final = 150
  := by sorry

end increase_by_fifty_percent_l1983_198363


namespace unique_solution_cube_difference_prime_l1983_198390

theorem unique_solution_cube_difference_prime (x y z : ℕ+) : 
  Nat.Prime y.val ∧ 
  ¬(3 ∣ z.val) ∧ 
  ¬(y.val ∣ z.val) ∧ 
  x.val^3 - y.val^3 = z.val^2 →
  x = 8 ∧ y = 7 ∧ z = 13 :=
sorry

end unique_solution_cube_difference_prime_l1983_198390


namespace bake_sale_chips_l1983_198325

/-- The number of cups of chocolate chips needed for one recipe -/
def chips_per_recipe : ℕ := 2

/-- The number of recipes to be made -/
def num_recipes : ℕ := 23

/-- The total number of cups of chocolate chips needed -/
def total_chips : ℕ := chips_per_recipe * num_recipes

theorem bake_sale_chips : total_chips = 46 := by
  sorry

end bake_sale_chips_l1983_198325


namespace locus_is_two_ellipses_l1983_198343

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the locus of points
def LocusOfPoints (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs (dist p c1.center - c1.radius) = abs (dist p c2.center - c2.radius)}

-- Define the ellipse
def Ellipse (f1 f2 : ℝ × ℝ) (major_axis : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p f1 + dist p f2 = major_axis}

-- Theorem statement
theorem locus_is_two_ellipses (c1 c2 : Circle) 
  (h1 : c1.radius > c2.radius) 
  (h2 : dist c1.center c2.center < c1.radius - c2.radius) :
  LocusOfPoints c1 c2 = 
    Ellipse c1.center c2.center (c1.radius + c2.radius) ∪
    Ellipse c1.center c2.center (c1.radius - c2.radius) := by
  sorry


end locus_is_two_ellipses_l1983_198343


namespace transformation_result_l1983_198315

/-- Rotation of 180 degrees counterclockwise around a point -/
def rotate180 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

/-- Reflection about the line y = -x -/
def reflectAboutNegativeXEqualsY (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.2, point.1)

/-- The main theorem -/
theorem transformation_result (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let center : ℝ × ℝ := (1, 5)
  let transformed := reflectAboutNegativeXEqualsY (rotate180 center P)
  transformed = (7, -3) → b - a = -2 := by
  sorry

end transformation_result_l1983_198315


namespace overall_gain_percentage_l1983_198374

/-- Calculate the overall gain percentage for three articles -/
theorem overall_gain_percentage
  (cost_A cost_B cost_C : ℝ)
  (sell_A sell_B sell_C : ℝ)
  (h_cost_A : cost_A = 100)
  (h_cost_B : cost_B = 200)
  (h_cost_C : cost_C = 300)
  (h_sell_A : sell_A = 110)
  (h_sell_B : sell_B = 250)
  (h_sell_C : sell_C = 330) :
  (((sell_A + sell_B + sell_C) - (cost_A + cost_B + cost_C)) / (cost_A + cost_B + cost_C)) * 100 = 15 := by
  sorry

end overall_gain_percentage_l1983_198374


namespace john_rejection_rate_proof_l1983_198333

/-- The percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.7

/-- The total percentage of products rejected -/
def total_rejection_rate : ℝ := 0.75

/-- The ratio of products Jane inspected compared to John -/
def jane_inspection_ratio : ℝ := 1.25

/-- John's rejection rate -/
def john_rejection_rate : ℝ := 0.8125

theorem john_rejection_rate_proof :
  let total_products := 1 + jane_inspection_ratio
  jane_rejection_rate * jane_inspection_ratio + john_rejection_rate = total_rejection_rate * total_products :=
by sorry

end john_rejection_rate_proof_l1983_198333


namespace debby_ate_nine_candies_l1983_198354

/-- Represents the number of candy pieces Debby ate -/
def candy_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proves that Debby ate 9 pieces of candy -/
theorem debby_ate_nine_candies : candy_eaten 12 3 = 9 := by
  sorry

end debby_ate_nine_candies_l1983_198354


namespace ellipse_major_axis_length_l1983_198339

/-- The length of the major axis of an ellipse with given foci and tangent line -/
theorem ellipse_major_axis_length : ∀ (F₁ F₂ : ℝ × ℝ) (y₀ : ℝ),
  F₁ = (4, 10) →
  F₂ = (34, 40) →
  y₀ = -5 →
  ∃ (X : ℝ × ℝ), X.2 = y₀ ∧ 
    (∀ (P : ℝ × ℝ), P.2 = y₀ → 
      Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
      Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2) ≥ 
      Real.sqrt ((X.1 - F₁.1)^2 + (X.2 - F₁.2)^2) + 
      Real.sqrt ((X.1 - F₂.1)^2 + (X.2 - F₂.2)^2)) →
  30 * Real.sqrt 5 = 
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - (2 * y₀ - F₁.2))^2) := by
  sorry

#check ellipse_major_axis_length

end ellipse_major_axis_length_l1983_198339


namespace complex_modulus_equality_l1983_198379

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → (Complex.abs (4 + n * Complex.I) = 4 * Real.sqrt 13 ↔ n = 8 * Real.sqrt 3) := by
  sorry

end complex_modulus_equality_l1983_198379


namespace complex_number_in_second_quadrant_l1983_198301

def complex_number : ℂ := Complex.I + Complex.I^2

theorem complex_number_in_second_quadrant :
  complex_number.re < 0 ∧ complex_number.im > 0 :=
by sorry

end complex_number_in_second_quadrant_l1983_198301


namespace largest_four_digit_sum_18_l1983_198347

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem largest_four_digit_sum_18 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 18 → n ≤ 9720 :=
by sorry

end largest_four_digit_sum_18_l1983_198347


namespace biased_coin_theorem_l1983_198330

def biased_coin_prob (h : ℚ) : Prop :=
  (15 : ℚ) * h^2 * (1 - h)^4 = (20 : ℚ) * h^3 * (1 - h)^3

theorem biased_coin_theorem :
  ∀ h : ℚ, 0 < h → h < 1 → biased_coin_prob h →
  (15 : ℚ) * h^4 * (1 - h)^2 = 40 / 243 :=
by sorry

end biased_coin_theorem_l1983_198330


namespace rectangle_with_hole_area_l1983_198397

/-- The area of a rectangle with dimensions (2x+14) and (2x+10), minus the area of a rectangular hole
    with dimensions y and x, where (y+1) = (x-2) and x = (2y+3), is equal to 2x^2 + 57x + 131. -/
theorem rectangle_with_hole_area (x y : ℝ) : 
  (y + 1 = x - 2) → 
  (x = 2*y + 3) → 
  (2*x + 14) * (2*x + 10) - y * x = 2*x^2 + 57*x + 131 := by
sorry

end rectangle_with_hole_area_l1983_198397


namespace total_vehicles_l1983_198345

-- Define the number of trucks
def num_trucks : ℕ := 20

-- Define the number of tanks as a function of the number of trucks
def num_tanks : ℕ := 5 * num_trucks

-- Theorem to prove
theorem total_vehicles : num_tanks + num_trucks = 120 := by
  sorry

end total_vehicles_l1983_198345


namespace tan_ratio_theorem_l1983_198336

theorem tan_ratio_theorem (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 8) :
  (Real.tan x * Real.tan y) / (Real.tan x / Real.tan y + Real.tan y / Real.tan x) = 31 / 13 := by
  sorry

end tan_ratio_theorem_l1983_198336


namespace max_value_quadratic_inequality_l1983_198335

theorem max_value_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -2 * x^2 + 9 * x - 7
  ∃ (max_x : ℝ), max_x = 3.5 ∧
    (∀ x : ℝ, f x ≤ 0 → x ≤ max_x) ∧
    f max_x ≤ 0 :=
by sorry

end max_value_quadratic_inequality_l1983_198335


namespace expression_evaluation_l1983_198311

theorem expression_evaluation : |-3| - 2 * Real.tan (π / 3) + (1 / 2)⁻¹ + Real.sqrt 12 = 5 := by
  sorry

end expression_evaluation_l1983_198311


namespace smallest_n_for_2005_angles_l1983_198393

/-- A function that, given a natural number n, returns the number of angles not exceeding 120° 
    between pairs of points when n points are placed on a circle. -/
def anglesNotExceeding120 (n : ℕ) : ℕ := sorry

/-- The proposition that 91 is the smallest natural number satisfying the condition -/
theorem smallest_n_for_2005_angles : 
  (∀ n : ℕ, n < 91 → anglesNotExceeding120 n < 2005) ∧ 
  (anglesNotExceeding120 91 ≥ 2005) :=
sorry

end smallest_n_for_2005_angles_l1983_198393


namespace sequence_properties_l1983_198359

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (is_arithmetic_sequence a ∧ 
   (is_geometric_sequence (a 4) (a 7) (a 9) → 
    ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78)) :=
by sorry

end sequence_properties_l1983_198359


namespace remainder_x_plus_one_2025_mod_x_squared_plus_one_l1983_198355

theorem remainder_x_plus_one_2025_mod_x_squared_plus_one (x : ℤ) :
  (x + 1) ^ 2025 % (x^2 + 1) = 0 := by
sorry

end remainder_x_plus_one_2025_mod_x_squared_plus_one_l1983_198355


namespace divisor_problem_l1983_198331

theorem divisor_problem (initial_number : ℕ) (added_number : ℕ) (divisor : ℕ) : 
  initial_number = 8679921 →
  added_number = 72 →
  divisor = 69 →
  (initial_number + added_number) % divisor = 0 :=
by sorry

end divisor_problem_l1983_198331


namespace dunbar_bouquets_l1983_198304

/-- The number of table decorations needed --/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each table decoration --/
def roses_per_table_decoration : ℕ := 12

/-- The number of white roses used in each bouquet --/
def roses_per_bouquet : ℕ := 5

/-- The total number of white roses needed for all bouquets and table decorations --/
def total_roses : ℕ := 109

/-- The number of bouquets Mrs. Dunbar needs to make --/
def num_bouquets : ℕ := (total_roses - num_table_decorations * roses_per_table_decoration) / roses_per_bouquet

theorem dunbar_bouquets : num_bouquets = 5 := by
  sorry

end dunbar_bouquets_l1983_198304


namespace mean_age_of_friends_l1983_198340

theorem mean_age_of_friends (age_group1 : ℕ) (age_group2 : ℕ) 
  (h1 : age_group1 = 12 * 12 + 3)  -- 12 years and 3 months in months
  (h2 : age_group2 = 13 * 12 + 5)  -- 13 years and 5 months in months
  : (3 * age_group1 + 4 * age_group2) / 7 = 155 := by
  sorry

end mean_age_of_friends_l1983_198340


namespace password_probability_l1983_198382

/-- The set of possible first characters in the password -/
def first_char : Finset Char := {'M', 'I', 'N'}

/-- The set of possible second characters in the password -/
def second_char : Finset Char := {'1', '2', '3', '4', '5'}

/-- The type representing a two-character password -/
def Password := Char × Char

/-- The set of all possible passwords -/
def all_passwords : Finset Password :=
  first_char.product second_char

theorem password_probability :
  (Finset.card all_passwords : ℚ) = 15 ∧
  (1 : ℚ) / (Finset.card all_passwords : ℚ) = 1 / 15 := by
  sorry

end password_probability_l1983_198382


namespace square_rectangle_area_ratio_l1983_198318

theorem square_rectangle_area_ratio :
  let rectangle_width : ℝ := 3
  let rectangle_length : ℝ := 5
  let square_side : ℝ := 1
  let square_area := square_side ^ 2
  let rectangle_area := rectangle_width * rectangle_length
  square_area / rectangle_area = 1 / 15 := by
sorry

end square_rectangle_area_ratio_l1983_198318


namespace selene_purchase_total_l1983_198312

/-- The price of an instant camera -/
def camera_price : ℝ := 110

/-- The price of a digital photo frame -/
def frame_price : ℝ := 120

/-- The number of cameras purchased -/
def num_cameras : ℕ := 2

/-- The number of frames purchased -/
def num_frames : ℕ := 3

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.05

/-- The total amount Selene pays -/
def total_paid : ℝ := 551

theorem selene_purchase_total :
  (camera_price * num_cameras + frame_price * num_frames) * (1 - discount_rate) = total_paid := by
  sorry

end selene_purchase_total_l1983_198312


namespace gold_bar_worth_l1983_198300

/-- Proves that the worth of each gold bar is $20,000 given the specified conditions -/
theorem gold_bar_worth (rows : ℕ) (bars_per_row : ℕ) (total_worth : ℕ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : rows = 4 := by sorry
  have h2 : bars_per_row = 20 := by sorry
  have h3 : total_worth = 1600000 := by sorry

  -- Calculate the total number of gold bars
  let total_bars := rows * bars_per_row

  -- Calculate the worth of each gold bar
  let bar_worth := total_worth / total_bars

  -- Prove that bar_worth equals 20000
  sorry

-- The theorem statement
#check gold_bar_worth

end gold_bar_worth_l1983_198300


namespace product_of_positive_reals_l1983_198329

theorem product_of_positive_reals (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 15/8) : r * s = Real.sqrt 17 / 4 := by
  sorry

end product_of_positive_reals_l1983_198329


namespace max_product_distances_l1983_198332

/-- Given two perpendicular lines passing through fixed points A and B,
    prove that the maximum value of the product of distances from their
    intersection point P to A and B is |AB|²/2 -/
theorem max_product_distances (m : ℝ) : ∃ (P : ℝ × ℝ),
  (P.1 + m * P.2 = 0) ∧
  (m * P.1 - P.2 - m + 3 = 0) →
  ∀ (Q : ℝ × ℝ),
    (Q.1 + m * Q.2 = 0) ∧
    (m * Q.1 - Q.2 - m + 3 = 0) →
    (Q.1 - 0)^2 + (Q.2 - 0)^2 * ((Q.1 - 1)^2 + (Q.2 - 3)^2) ≤ 25 :=
sorry

end max_product_distances_l1983_198332


namespace michael_matchsticks_l1983_198378

theorem michael_matchsticks (total : ℕ) (houses : ℕ) (sticks_per_house : ℕ) : 
  houses = 30 →
  sticks_per_house = 10 →
  houses * sticks_per_house = total / 2 →
  total = 600 := by
  sorry

end michael_matchsticks_l1983_198378


namespace frank_to_betty_bill_ratio_l1983_198387

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := 12

/-- The number of seeds Frank planted from each orange -/
def seeds_per_orange : ℕ := 2

/-- The number of oranges on each tree -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_oranges : ℕ := 810

/-- Theorem stating the ratio of Frank's oranges to Betty and Bill's combined oranges -/
theorem frank_to_betty_bill_ratio :
  ∃ (frank_oranges : ℕ),
    frank_oranges > 0 ∧
    philip_oranges = frank_oranges * seeds_per_orange * oranges_per_tree ∧
    frank_oranges = 3 * (betty_oranges + bill_oranges) := by
  sorry

end frank_to_betty_bill_ratio_l1983_198387


namespace unique_player_count_l1983_198353

/-- Given a total number of socks and the fact that each player contributes two socks,
    proves that there is only one possible number of players. -/
theorem unique_player_count (total_socks : ℕ) (h : total_socks = 22) :
  ∃! n : ℕ, n * 2 = total_socks := by sorry

end unique_player_count_l1983_198353


namespace cube_edge_length_l1983_198377

theorem cube_edge_length (V : ℝ) (h : V = 32 / 3 * Real.pi) :
  ∃ (a : ℝ), a > 0 ∧ a = 4 * Real.sqrt 3 / 3 ∧
  V = 4 / 3 * Real.pi * (3 * a^2 / 4) ^ (3/2) :=
sorry

end cube_edge_length_l1983_198377


namespace batsman_average_after_19th_inning_l1983_198370

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRunsBefore : ℕ
  scoreInLastInning : ℕ
  averageIncrease : ℚ

/-- Calculates the new average of a batsman after their latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRunsBefore + b.scoreInLastInning : ℚ) / b.innings

theorem batsman_average_after_19th_inning 
  (b : Batsman) 
  (h1 : b.innings = 19) 
  (h2 : b.scoreInLastInning = 100) 
  (h3 : b.averageIncrease = 2) :
  newAverage b = 64 := by
  sorry

end batsman_average_after_19th_inning_l1983_198370


namespace arithmetic_calculation_l1983_198384

theorem arithmetic_calculation : 80 + 5 * 12 / (180 / 3) = 81 := by
  sorry

end arithmetic_calculation_l1983_198384


namespace tank_depth_l1983_198351

theorem tank_depth (length width : ℝ) (cost_per_sqm total_cost : ℝ) (d : ℝ) :
  length = 25 →
  width = 12 →
  cost_per_sqm = 0.3 →
  total_cost = 223.2 →
  cost_per_sqm * (length * width + 2 * (length * d) + 2 * (width * d)) = total_cost →
  d = 6 := by
sorry

end tank_depth_l1983_198351


namespace smallest_number_is_2544_l1983_198365

def is_smallest_number (x : ℕ) : Prop :=
  (x - 24) % 5 = 0 ∧
  (x - 24) % 10 = 0 ∧
  (x - 24) % 15 = 0 ∧
  (x - 24) / (Nat.lcm 5 (Nat.lcm 10 15)) = 84 ∧
  ∀ y, y < x → ¬(is_smallest_number y)

theorem smallest_number_is_2544 :
  is_smallest_number 2544 :=
sorry

end smallest_number_is_2544_l1983_198365


namespace rectangle_construction_l1983_198320

/-- Given a length b and a sum s, prove the existence of a rectangle with side lengths a and b,
    such that s equals the sum of the diagonal and side b. -/
theorem rectangle_construction (b : ℝ) (s : ℝ) (h_pos : b > 0 ∧ s > b) :
  ∃ (a : ℝ), a > 0 ∧ s = a + (a^2 + b^2).sqrt := by
  sorry

#check rectangle_construction

end rectangle_construction_l1983_198320


namespace not_possible_N_l1983_198328

-- Define the set M
def M : Set ℝ := {x | x^2 - 6*x - 16 < 0}

-- Define the theorem
theorem not_possible_N (N : Set ℝ) (h1 : M ∩ N = N) : N ≠ Set.Icc (-1 : ℝ) 8 := by
  sorry

end not_possible_N_l1983_198328


namespace yura_catches_lena_l1983_198356

/-- The time it takes for Yura to catch up with Lena -/
def catchUpTime : ℝ := 5

/-- Lena's walking speed -/
def lenaSpeed : ℝ := 1

/-- The time difference between Lena and Yura's start -/
def timeDifference : ℝ := 5

theorem yura_catches_lena :
  ∀ (t : ℝ),
  t = catchUpTime →
  (lenaSpeed * (t + timeDifference)) = (2 * lenaSpeed * t) :=
by sorry

end yura_catches_lena_l1983_198356


namespace one_and_one_third_problem_l1983_198302

theorem one_and_one_third_problem :
  ∀ x : ℝ, (4/3 : ℝ) * x = 45 ↔ x = 33.75 := by sorry

end one_and_one_third_problem_l1983_198302


namespace rescue_possible_l1983_198381

/-- Represents the rescue mission parameters --/
structure RescueMission where
  distance : ℝ
  rover_air : ℝ
  ponchik_extra_air : ℝ
  dunno_tank_air : ℝ
  max_tanks : ℕ
  speed : ℝ

/-- Represents a rescue strategy --/
structure RescueStrategy where
  trips : ℕ
  air_drops : List ℝ
  meeting_point : ℝ

/-- Checks if a rescue strategy is valid for a given mission --/
def is_valid_strategy (mission : RescueMission) (strategy : RescueStrategy) : Prop :=
  -- Define the conditions for a valid strategy
  sorry

/-- Theorem stating that a valid rescue strategy exists --/
theorem rescue_possible (mission : RescueMission) 
  (h1 : mission.distance = 18)
  (h2 : mission.rover_air = 3)
  (h3 : mission.ponchik_extra_air = 1)
  (h4 : mission.dunno_tank_air = 2)
  (h5 : mission.max_tanks = 2)
  (h6 : mission.speed = 6) :
  ∃ (strategy : RescueStrategy), is_valid_strategy mission strategy :=
sorry

end rescue_possible_l1983_198381


namespace g_composition_of_2_l1983_198348

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 1

theorem g_composition_of_2 : g (g (g (g 2))) = 1406 := by
  sorry

end g_composition_of_2_l1983_198348


namespace coplanar_condition_l1983_198394

open Vector

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define the points
variable (O E F G H : V)

-- Define the condition for coplanarity
def are_coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (B - A) + b • (C - A) + c • (D - A) = 0

-- Define the theorem
theorem coplanar_condition (m : ℝ) :
  (4 • (E - O) - 3 • (F - O) + 6 • (G - O) + m • (H - O) = 0) →
  (are_coplanar E F G H ↔ m = -7) :=
by sorry

end coplanar_condition_l1983_198394
