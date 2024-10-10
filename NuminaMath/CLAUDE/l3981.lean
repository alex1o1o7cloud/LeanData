import Mathlib

namespace distinct_distances_lower_bound_l3981_398145

/-- Given n points on a plane, where n ≥ 2, the number of distinct distances k
    between these points satisfies k ≥ √(n - 3/4) - 1/2. -/
theorem distinct_distances_lower_bound (n : ℕ) (k : ℕ) (h : n ≥ 2) :
  k ≥ Real.sqrt (n - 3/4) - 1/2 :=
by sorry

end distinct_distances_lower_bound_l3981_398145


namespace half_dollar_and_dollar_heads_probability_l3981_398190

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the result of flipping four coins -/
structure FourCoinFlip :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (halfDollar : CoinFlip)
  (oneDollar : CoinFlip)

/-- The set of all possible outcomes when flipping four coins -/
def allOutcomes : Finset FourCoinFlip := sorry

/-- The set of favorable outcomes (half-dollar and one-dollar are both heads) -/
def favorableOutcomes : Finset FourCoinFlip := sorry

/-- The probability of an event occurring -/
def probability (event : Finset FourCoinFlip) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

theorem half_dollar_and_dollar_heads_probability :
  probability favorableOutcomes = 1/4 := by sorry

end half_dollar_and_dollar_heads_probability_l3981_398190


namespace consecutive_odd_product_ends_09_l3981_398124

theorem consecutive_odd_product_ends_09 (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, (10*n - 3) * (10*n - 1) * (10*n + 1) * (10*n + 3) = 100 * k + 9 :=
sorry

end consecutive_odd_product_ends_09_l3981_398124


namespace f_odd_and_decreasing_l3981_398171

def f (x : ℝ) : ℝ := -x * abs x

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end f_odd_and_decreasing_l3981_398171


namespace sheets_in_box_l3981_398176

/-- The number of sheets needed per printer -/
def sheets_per_printer : ℕ := 7

/-- The number of printers that can be filled -/
def num_printers : ℕ := 31

/-- The total number of sheets in the box -/
def total_sheets : ℕ := sheets_per_printer * num_printers

theorem sheets_in_box : total_sheets = 217 := by
  sorry

end sheets_in_box_l3981_398176


namespace alternative_plan_cost_is_eleven_l3981_398173

/-- The cost of Darnell's current unlimited plan -/
def current_plan_cost : ℕ := 12

/-- The difference in cost between the current plan and the alternative plan -/
def cost_difference : ℕ := 1

/-- The number of texts Darnell sends per month -/
def texts_per_month : ℕ := 60

/-- The number of minutes Darnell spends on calls per month -/
def call_minutes_per_month : ℕ := 60

/-- The cost of the alternative plan -/
def alternative_plan_cost : ℕ := current_plan_cost - cost_difference

theorem alternative_plan_cost_is_eleven :
  alternative_plan_cost = 11 :=
by sorry

end alternative_plan_cost_is_eleven_l3981_398173


namespace triangle_area_l3981_398103

def a : ℝ × ℝ := (4, -3)
def b : ℝ × ℝ := (-6, 5)
def c : ℝ × ℝ := (-12, 10)

theorem triangle_area : 
  let det := a.1 * c.2 - a.2 * c.1
  (1/2 : ℝ) * |det| = 2 := by sorry

end triangle_area_l3981_398103


namespace cylinder_radius_ratio_l3981_398137

/-- Given a right circular cylinder with initial volume 6 and final volume 186,
    prove that the ratio of the new radius to the original radius is √31. -/
theorem cylinder_radius_ratio (r R h : ℝ) : 
  r > 0 → h > 0 → 
  π * r^2 * h = 6 → 
  π * R^2 * h = 186 → 
  R / r = Real.sqrt 31 := by
  sorry

end cylinder_radius_ratio_l3981_398137


namespace barry_sitting_time_l3981_398100

/-- Calculates the sitting time between turns for Barry's head-standing routine -/
def calculate_sitting_time (total_time minutes_per_turn number_of_turns : ℕ) : ℕ :=
  let total_standing_time := minutes_per_turn * number_of_turns
  let total_sitting_time := total_time - total_standing_time
  let number_of_breaks := number_of_turns - 1
  (total_sitting_time + number_of_breaks - 1) / number_of_breaks

theorem barry_sitting_time :
  let total_time : ℕ := 120  -- 2 hours in minutes
  let minutes_per_turn : ℕ := 10
  let number_of_turns : ℕ := 8
  calculate_sitting_time total_time minutes_per_turn number_of_turns = 6 := by
  sorry

end barry_sitting_time_l3981_398100


namespace complex_transformation_l3981_398135

/-- The result of applying a 60° counter-clockwise rotation followed by a dilation 
    with scale factor 2 to the complex number -4 + 3i -/
theorem complex_transformation : 
  let z : ℂ := -4 + 3 * Complex.I
  let rotation : ℂ := Complex.exp (Complex.I * Real.pi / 3)
  let dilation : ℝ := 2
  (dilation * rotation * z) = (-4 - 3 * Real.sqrt 3) + (3 - 4 * Real.sqrt 3) * Complex.I := by
  sorry

end complex_transformation_l3981_398135


namespace pencil_packs_l3981_398199

theorem pencil_packs (pencils_per_pack : ℕ) (pencils_per_row : ℕ) (total_rows : ℕ) : 
  pencils_per_pack = 4 →
  pencils_per_row = 2 →
  total_rows = 70 →
  (total_rows * pencils_per_row) / pencils_per_pack = 35 :=
by
  sorry

end pencil_packs_l3981_398199


namespace original_pay_before_tax_l3981_398141

/-- Given a 10% tax rate and a take-home pay of $585, prove that the original pay before tax deduction is $650. -/
theorem original_pay_before_tax (tax_rate : ℝ) (take_home_pay : ℝ) (original_pay : ℝ) :
  tax_rate = 0.1 →
  take_home_pay = 585 →
  original_pay * (1 - tax_rate) = take_home_pay →
  original_pay = 650 :=
by sorry

end original_pay_before_tax_l3981_398141


namespace max_value_of_f_l3981_398101

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 8 := by
  sorry

end max_value_of_f_l3981_398101


namespace M_mod_45_l3981_398189

def M : ℕ := sorry

theorem M_mod_45 : M % 45 = 15 := by sorry

end M_mod_45_l3981_398189


namespace magpie_porridge_l3981_398158

/-- The amount of porridge given to each chick -/
def PorridgeDistribution (p₁ p₂ p₃ p₄ p₅ p₆ : ℝ) : Prop :=
  p₃ = p₁ + p₂ ∧
  p₄ = p₂ + p₃ ∧
  p₅ = p₃ + p₄ ∧
  p₆ = p₄ + p₅ ∧
  p₅ = 10

theorem magpie_porridge : 
  ∀ p₁ p₂ p₃ p₄ p₅ p₆ : ℝ, 
  PorridgeDistribution p₁ p₂ p₃ p₄ p₅ p₆ → 
  p₁ + p₂ + p₃ + p₄ + p₅ + p₆ = 40 :=
by
  sorry

end magpie_porridge_l3981_398158


namespace ellipse_properties_l3981_398138

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/5 + y^2 = 1

-- Define the right focus F
def right_focus : ℝ × ℝ := (2, 0)

-- Define the line l passing through F
def line_l (k : ℝ) (x : ℝ) : ℝ := k*(x - 2)

-- Define points A and B on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse_C x y ∧ ∃ k, y = line_l k x

-- Define point M on y-axis
def point_M (y : ℝ) : ℝ × ℝ := (0, y)

-- Define vectors MA, MB, FA, and FB
def vector_MA (x y y0 : ℝ) : ℝ × ℝ := (x, y - y0)
def vector_MB (x y y0 : ℝ) : ℝ × ℝ := (x, y - y0)
def vector_FA (x y : ℝ) : ℝ × ℝ := (x - 2, y)
def vector_FB (x y : ℝ) : ℝ × ℝ := (x - 2, y)

theorem ellipse_properties :
  ∀ (x1 y1 x2 y2 y0 m n : ℝ),
  point_on_ellipse x1 y1 →
  point_on_ellipse x2 y2 →
  vector_MA x1 y1 y0 = m • (vector_FA x1 y1) →
  vector_MB x2 y2 y0 = n • (vector_FB x2 y2) →
  m + n = 10 :=
sorry

end ellipse_properties_l3981_398138


namespace age_difference_proof_l3981_398136

theorem age_difference_proof (younger_age elder_age : ℕ) 
  (h1 : elder_age > younger_age)
  (h2 : elder_age - 4 = 5 * (younger_age - 4))
  (h3 : younger_age = 29)
  (h4 : elder_age = 49) :
  elder_age - younger_age = 20 := by
sorry

end age_difference_proof_l3981_398136


namespace peggy_total_dolls_l3981_398131

def initial_dolls : ℕ := 6
def grandmother_gift : ℕ := 30
def additional_dolls : ℕ := grandmother_gift / 2

theorem peggy_total_dolls :
  initial_dolls + grandmother_gift + additional_dolls = 51 := by
  sorry

end peggy_total_dolls_l3981_398131


namespace library_visitors_l3981_398167

theorem library_visitors (sunday_visitors : ℕ) (total_days : ℕ) (sundays : ℕ) (avg_visitors : ℕ) :
  sunday_visitors = 510 →
  total_days = 30 →
  sundays = 5 →
  avg_visitors = 285 →
  (sunday_visitors * sundays + (total_days - sundays) * 
    ((total_days * avg_visitors - sunday_visitors * sundays) / (total_days - sundays))) / total_days = avg_visitors :=
by sorry

end library_visitors_l3981_398167


namespace dress_shop_inventory_l3981_398112

def total_dresses (red : ℕ) (blue : ℕ) : ℕ := red + blue

theorem dress_shop_inventory : 
  let red : ℕ := 83
  let blue : ℕ := red + 34
  total_dresses red blue = 200 := by
sorry

end dress_shop_inventory_l3981_398112


namespace a_value_proof_l3981_398175

theorem a_value_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end a_value_proof_l3981_398175


namespace polynomial_sum_l3981_398166

/-- Given a polynomial P such that P + (x^2 - y^2) = x^2 + y^2, then P = 2y^2 -/
theorem polynomial_sum (x y : ℝ) (P : ℝ → ℝ) :
  (∀ x, P x + (x^2 - y^2) = x^2 + y^2) → P x = 2 * y^2 := by
  sorry

end polynomial_sum_l3981_398166


namespace hyperbola_asymptotes_l3981_398161

/-- Definition of a hyperbola with equation x^2 - y^2/4 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- Definition of the asymptotes y = ±2x -/
def asymptotes (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

/-- Theorem stating that the given hyperbola has the specified asymptotes -/
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ 0 ∧ asymptotes x' y') :=
by sorry


end hyperbola_asymptotes_l3981_398161


namespace strawberry_fraction_remaining_l3981_398195

theorem strawberry_fraction_remaining 
  (num_hedgehogs : ℕ) 
  (num_baskets : ℕ) 
  (strawberries_per_basket : ℕ) 
  (strawberries_eaten_per_hedgehog : ℕ) : 
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  strawberries_eaten_per_hedgehog = 1050 →
  (num_baskets * strawberries_per_basket - num_hedgehogs * strawberries_eaten_per_hedgehog : ℚ) / 
  (num_baskets * strawberries_per_basket) = 2/9 := by
  sorry

end strawberry_fraction_remaining_l3981_398195


namespace range_of_t_l3981_398177

def A : Set ℝ := {x | x^2 - 3*x + 2 ≥ 0}
def B (t : ℝ) : Set ℝ := {x | x ≥ t}

theorem range_of_t (t : ℝ) : A ∪ B t = A → t ≥ 2 := by
  sorry

end range_of_t_l3981_398177


namespace factorization_equality_l3981_398184

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end factorization_equality_l3981_398184


namespace expression_bounds_l3981_398155

theorem expression_bounds (a b c d e : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  let expr := Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
              Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
              Real.sqrt (e^2 + (1-a)^2)
  5 / Real.sqrt 2 ≤ expr ∧ expr ≤ 5 ∧ 
  ∃ (a' b' c' d' e' : Real), (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ 
    (0 ≤ c' ∧ c' ≤ 1) ∧ (0 ≤ d' ∧ d' ≤ 1) ∧ (0 ≤ e' ∧ e' ≤ 1) ∧
    let expr' := Real.sqrt (a'^2 + (1-b')^2) + Real.sqrt (b'^2 + (1-c')^2) + 
                 Real.sqrt (c'^2 + (1-d')^2) + Real.sqrt (d'^2 + (1-e')^2) + 
                 Real.sqrt (e'^2 + (1-a')^2)
    expr' = 5 / Real.sqrt 2 ∧
  ∃ (a'' b'' c'' d'' e'' : Real), (0 ≤ a'' ∧ a'' ≤ 1) ∧ (0 ≤ b'' ∧ b'' ≤ 1) ∧ 
    (0 ≤ c'' ∧ c'' ≤ 1) ∧ (0 ≤ d'' ∧ d'' ≤ 1) ∧ (0 ≤ e'' ∧ e'' ≤ 1) ∧
    let expr'' := Real.sqrt (a''^2 + (1-b'')^2) + Real.sqrt (b''^2 + (1-c'')^2) + 
                  Real.sqrt (c''^2 + (1-d'')^2) + Real.sqrt (d''^2 + (1-e'')^2) + 
                  Real.sqrt (e''^2 + (1-a'')^2)
    expr'' = 5 := by
  sorry

end expression_bounds_l3981_398155


namespace downstream_distance_l3981_398128

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (time : ℝ)
  (h1 : boat_speed = 14)
  (h2 : stream_speed = 6)
  (h3 : time = 3.6) :
  boat_speed + stream_speed * time = 72 :=
by sorry

end downstream_distance_l3981_398128


namespace number_of_binders_l3981_398126

theorem number_of_binders (total_sheets : ℕ) (sheets_per_binder : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : sheets_per_binder = 490)
  (h3 : total_sheets % sheets_per_binder = 0) :
  total_sheets / sheets_per_binder = 5 := by
  sorry

end number_of_binders_l3981_398126


namespace expression_equals_24_l3981_398139

theorem expression_equals_24 : 
  2012 * ((3.75 * 1.3 + 3 / 2.6666666666666665) / ((1+3+5+7+9) * 20 + 3)) = 24 := by
  sorry

end expression_equals_24_l3981_398139


namespace room_area_difference_l3981_398105

-- Define the dimensions of the rooms
def largest_room_width : ℕ := 45
def largest_room_length : ℕ := 30
def smallest_room_width : ℕ := 15
def smallest_room_length : ℕ := 8

-- Define the function to calculate the area of a rectangular room
def room_area (width : ℕ) (length : ℕ) : ℕ := width * length

-- Theorem statement
theorem room_area_difference :
  room_area largest_room_width largest_room_length - 
  room_area smallest_room_width smallest_room_length = 1230 := by
  sorry

end room_area_difference_l3981_398105


namespace three_person_subcommittees_l3981_398179

theorem three_person_subcommittees (n : ℕ) (k : ℕ) : n = 8 → k = 3 →
  Nat.choose n k = 56 := by sorry

end three_person_subcommittees_l3981_398179


namespace probability_statements_l3981_398129

-- Define a type for days in a year
def Day := Fin 365

-- Define a type for numbers in a drawing
def DrawNumber := Fin 10

-- Function to calculate birthday probability
def birthday_probability : ℚ :=
  1 / 365

-- Function to check if drawing method is fair
def is_fair_drawing_method (draw : DrawNumber → DrawNumber → Bool) : Prop :=
  ∀ a b : DrawNumber, (draw a b) = ¬(draw b a)

-- Theorem statement
theorem probability_statements :
  (birthday_probability = 1 / 365) ∧
  (∃ draw : DrawNumber → DrawNumber → Bool, is_fair_drawing_method draw) :=
sorry

end probability_statements_l3981_398129


namespace priyas_trip_l3981_398170

/-- Priya's trip between towns X, Y, and Z -/
theorem priyas_trip (time_x_to_z : ℝ) (speed_x_to_z : ℝ) (time_z_to_y : ℝ) :
  time_x_to_z = 5 →
  speed_x_to_z = 50 →
  time_z_to_y = 2.0833333333333335 →
  let distance_x_to_z := time_x_to_z * speed_x_to_z
  let distance_z_to_y := distance_x_to_z / 2
  let speed_z_to_y := distance_z_to_y / time_z_to_y
  speed_z_to_y = 60 := by
sorry


end priyas_trip_l3981_398170


namespace arithmetic_sequence_length_l3981_398160

/-- 
Theorem: The number of terms in an arithmetic sequence 
starting with 2, ending with 2014, and having a common difference of 4 
is equal to 504.
-/
theorem arithmetic_sequence_length : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 4   -- Common difference
  let aₙ : ℕ := 2014  -- Last term
  ∃ n : ℕ, n > 0 ∧ aₙ = a₁ + (n - 1) * d ∧ n = 504
  := by sorry

end arithmetic_sequence_length_l3981_398160


namespace investment_percentage_l3981_398140

theorem investment_percentage (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (additional_rate : ℝ) 
  (h1 : initial_investment = 1400)
  (h2 : initial_rate = 0.05)
  (h3 : additional_investment = 700)
  (h4 : additional_rate = 0.08) : 
  (initial_investment * initial_rate + additional_investment * additional_rate) / 
  (initial_investment + additional_investment) = 0.06 := by
  sorry

end investment_percentage_l3981_398140


namespace symmetric_points_range_l3981_398168

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 2 * x^2 - 3 * x else a / Real.exp x

theorem symmetric_points_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ f a x = f a (-y)) →
  -Real.exp (-1/2) ≤ a ∧ a ≤ 9 * Real.exp (-3) :=
sorry

end symmetric_points_range_l3981_398168


namespace system_solution_l3981_398191

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 7 * y = -14) ∧ 
    (5 * x + 3 * y = -13) ∧ 
    (x = -133 / 47) ∧ 
    (y = 18 / 47) := by
  sorry

end system_solution_l3981_398191


namespace solution_difference_l3981_398154

theorem solution_difference (r s : ℝ) : 
  (r - 4) * (r + 4) = 24 * r - 96 →
  (s - 4) * (s + 4) = 24 * s - 96 →
  r ≠ s →
  r > s →
  r - s = 16 := by sorry

end solution_difference_l3981_398154


namespace arithmetic_sequence_sum_l3981_398113

/-- Given an arithmetic sequence {a_n} where a_5 + a_6 + a_7 = 15,
    prove that the sum (a_3 + a_4 + ... + a_9) is equal to 35. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 + a 7 = 15 →                            -- given condition
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=    -- conclusion to prove
by sorry

end arithmetic_sequence_sum_l3981_398113


namespace stationery_cost_is_18300_l3981_398152

/-- Calculates the total amount paid for stationery given the number of pencil boxes,
    pencils per box, and the costs of pens and pencils. -/
def total_stationery_cost (pencil_boxes : ℕ) (pencils_per_box : ℕ) 
                          (pen_cost : ℕ) (pencil_cost : ℕ) : ℕ :=
  let total_pencils := pencil_boxes * pencils_per_box
  let total_pens := 2 * total_pencils + 300
  total_pens * pen_cost + total_pencils * pencil_cost

/-- Proves that the total amount paid for the stationery is $18300 -/
theorem stationery_cost_is_18300 :
  total_stationery_cost 15 80 5 4 = 18300 := by
  sorry

#eval total_stationery_cost 15 80 5 4

end stationery_cost_is_18300_l3981_398152


namespace min_sum_squares_l3981_398149

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4.8 ∧ x^2 + y^2 + z^2 ≥ m ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = m :=
sorry

end min_sum_squares_l3981_398149


namespace pollution_filtering_l3981_398133

/-- Given a pollution filtering process where P = P₀e^(-kt),
    if 10% of pollutants are eliminated in 5 hours,
    then 81% of pollutants remain after 10 hours. -/
theorem pollution_filtering (P₀ k : ℝ) (h : P₀ > 0) :
  P₀ * Real.exp (-5 * k) = P₀ * 0.9 →
  P₀ * Real.exp (-10 * k) = P₀ * 0.81 := by
sorry

end pollution_filtering_l3981_398133


namespace morning_afternoon_emails_l3981_398120

theorem morning_afternoon_emails (morning_emails afternoon_emails : ℕ) 
  (h1 : morning_emails = 5)
  (h2 : afternoon_emails = 8) :
  morning_emails + afternoon_emails = 13 := by
sorry

end morning_afternoon_emails_l3981_398120


namespace distance_to_origin_l3981_398172

/-- The distance from the point corresponding to the complex number 2i/(1+i) to the origin is √2. -/
theorem distance_to_origin : ∃ (z : ℂ), z = (2 * Complex.I) / (1 + Complex.I) ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end distance_to_origin_l3981_398172


namespace pencils_lost_l3981_398117

/-- Given an initial number of pencils and a final number of pencils,
    prove that the number of lost pencils is the difference between them. -/
theorem pencils_lost (initial final : ℕ) (h : initial ≥ final) :
  initial - final = initial - final :=
by sorry

end pencils_lost_l3981_398117


namespace canoe_kayak_difference_is_five_l3981_398116

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_kayak_ratio : ℚ
  total_revenue : ℕ

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (info : RentalInfo) : ℕ :=
  let canoes := (info.total_revenue / (3 * info.canoe_cost + 2 * info.kayak_cost)) * 3
  let kayaks := (info.total_revenue / (3 * info.canoe_cost + 2 * info.kayak_cost)) * 2
  canoes - kayaks

/-- Theorem stating the difference between canoes and kayaks rented --/
theorem canoe_kayak_difference_is_five (info : RentalInfo)
  (h1 : info.canoe_cost = 15)
  (h2 : info.kayak_cost = 18)
  (h3 : info.canoe_kayak_ratio = 3/2)
  (h4 : info.total_revenue = 405) :
  canoe_kayak_difference info = 5 := by
  sorry

end canoe_kayak_difference_is_five_l3981_398116


namespace equation_solution_l3981_398148

theorem equation_solution :
  ∃! x : ℝ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 63 / 14 := by
  sorry

end equation_solution_l3981_398148


namespace constant_pace_jogging_l3981_398156

/-- Represents the time taken to jog a certain distance at a constant pace -/
structure JoggingTime where
  distance : ℝ
  time : ℝ

/-- Given a constant jogging pace, if it takes 24 minutes to jog 3 miles,
    then it will take 12 minutes to jog 1.5 miles -/
theorem constant_pace_jogging 
  (pace : ℝ) 
  (gym : JoggingTime) 
  (park : JoggingTime) 
  (h1 : gym.distance = 3) 
  (h2 : gym.time = 24) 
  (h3 : park.distance = 1.5) 
  (h4 : pace > 0) 
  (h5 : ∀ j : JoggingTime, j.time = j.distance / pace) : 
  park.time = 12 :=
sorry

end constant_pace_jogging_l3981_398156


namespace sticker_count_after_loss_l3981_398111

/-- Given a number of stickers per page, an initial number of pages, and a number of lost pages,
    calculate the total number of remaining stickers. -/
def remaining_stickers (stickers_per_page : ℕ) (initial_pages : ℕ) (lost_pages : ℕ) : ℕ :=
  (initial_pages - lost_pages) * stickers_per_page

theorem sticker_count_after_loss :
  remaining_stickers 20 12 1 = 220 := by
  sorry

end sticker_count_after_loss_l3981_398111


namespace square_difference_equality_l3981_398194

theorem square_difference_equality (m n : ℝ) :
  9 * m^2 - (m - 2*n)^2 = 4 * (2*m - n) * (m + n) := by
  sorry

end square_difference_equality_l3981_398194


namespace unique_positive_solution_l3981_398108

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) ∧ x = 18 := by
  sorry

end unique_positive_solution_l3981_398108


namespace wheel_revolutions_l3981_398192

-- Define constants
def wheel_diameter : ℝ := 8
def distance_km : ℝ := 2
def km_to_feet : ℝ := 3280.84

-- Define the theorem
theorem wheel_revolutions :
  let wheel_circumference := π * wheel_diameter
  let distance_feet := distance_km * km_to_feet
  let revolutions := distance_feet / wheel_circumference
  revolutions = 820.21 / π := by
sorry

end wheel_revolutions_l3981_398192


namespace no_intersection_points_l3981_398182

theorem no_intersection_points : 
  ¬∃ (z : ℂ), z^4 + z = 1 ∧ Complex.abs z = 1 := by sorry

end no_intersection_points_l3981_398182


namespace ashtons_remaining_items_l3981_398123

def pencil_boxes : ℕ := 3
def pencils_per_box : ℕ := 14
def pen_boxes : ℕ := 2
def pens_per_box : ℕ := 10
def pencils_to_brother : ℕ := 6
def pencils_to_friends : ℕ := 12
def pens_to_friends : ℕ := 8

theorem ashtons_remaining_items :
  let initial_pencils := pencil_boxes * pencils_per_box
  let initial_pens := pen_boxes * pens_per_box
  let remaining_pencils := initial_pencils - pencils_to_brother - pencils_to_friends
  let remaining_pens := initial_pens - pens_to_friends
  remaining_pencils + remaining_pens = 36 := by
  sorry

end ashtons_remaining_items_l3981_398123


namespace perfect_square_digit_sum_l3981_398144

def is_valid_number (N : ℕ) : Prop :=
  ∃ k : ℕ,
    10000 ≤ N ∧ N < 100000 ∧
    N = 10000 * k + 1000 * (k + 1) + 100 * (k + 2) + 10 * (3 * k) + (k + 3)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem perfect_square_digit_sum :
  ∀ N : ℕ, is_valid_number N →
    ∃ m : ℕ, m * m = N → sum_of_digits m = 15 :=
by sorry

end perfect_square_digit_sum_l3981_398144


namespace complex_power_sum_l3981_398127

theorem complex_power_sum (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  Complex.abs (a^2020 + b^2020 + c^2020) = 3 := by
sorry

end complex_power_sum_l3981_398127


namespace inequality_proof_l3981_398114

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : d > 0) :
  d / c < (d + 4) / (c + 4) := by
  sorry

end inequality_proof_l3981_398114


namespace complex_equation_solutions_l3981_398187

theorem complex_equation_solutions (x y : ℝ) :
  x^2 - y^2 + (2*x*y : ℂ)*I = (2 : ℂ)*I →
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end complex_equation_solutions_l3981_398187


namespace peters_age_fraction_l3981_398153

/-- Proves that Peter's current age is 1/2 of his mother's age -/
theorem peters_age_fraction (peter_age harriet_age mother_age : ℕ) : 
  harriet_age = 13 →
  mother_age = 60 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  peter_age = mother_age / 2 := by
sorry

end peters_age_fraction_l3981_398153


namespace paperbacks_count_l3981_398162

/-- The number of books on the shelf -/
def total_books : ℕ := 8

/-- The number of hardback books on the shelf -/
def hardbacks : ℕ := 6

/-- The number of possible selections of 3 books that include at least one paperback -/
def selections_with_paperback : ℕ := 36

/-- The number of paperbacks on the shelf -/
def paperbacks : ℕ := total_books - hardbacks

/-- Theorem stating that the number of paperbacks is 2 -/
theorem paperbacks_count : paperbacks = 2 := by sorry

end paperbacks_count_l3981_398162


namespace yadav_savings_l3981_398193

/-- Mr. Yadav's monthly savings calculation --/
def monthly_savings (salary : ℝ) : ℝ :=
  salary * (1 - 0.6 - 0.5 * (1 - 0.6))

/-- Mr. Yadav's yearly savings calculation --/
def yearly_savings (salary : ℝ) : ℝ :=
  12 * monthly_savings salary

/-- Theorem: Mr. Yadav's yearly savings are 46800 --/
theorem yadav_savings :
  ∃ (salary : ℝ),
    salary > 0 ∧
    0.5 * (1 - 0.6) * salary = 3900 ∧
    yearly_savings salary = 46800 :=
sorry

end yadav_savings_l3981_398193


namespace rogers_coins_l3981_398119

theorem rogers_coins (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 3)
  (h2 : dime_piles = 3)
  (h3 : coins_per_pile = 7) :
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 42 :=
by sorry

end rogers_coins_l3981_398119


namespace least_common_multiple_of_first_ten_l3981_398183

/-- The least positive integer divisible by each of the first ten positive integers -/
def leastCommonMultiple : ℕ := 2520

/-- The set of the first ten positive integers -/
def firstTenIntegers : Finset ℕ := Finset.range 10

theorem least_common_multiple_of_first_ten :
  (∀ n ∈ firstTenIntegers, leastCommonMultiple % (n + 1) = 0) ∧
  (∀ m : ℕ, m > 0 → m < leastCommonMultiple →
    ∃ k ∈ firstTenIntegers, m % (k + 1) ≠ 0) :=
sorry

end least_common_multiple_of_first_ten_l3981_398183


namespace game_prep_time_calculation_l3981_398104

/-- Calculates the total time before playing the main game --/
def totalGamePrepTime (downloadTime installTime updateTime accountTime issuesTime tutorialTime : ℕ) : ℕ :=
  downloadTime + installTime + updateTime + accountTime + issuesTime + tutorialTime

theorem game_prep_time_calculation :
  let downloadTime : ℕ := 10
  let installTime : ℕ := downloadTime / 2
  let updateTime : ℕ := downloadTime * 2
  let accountTime : ℕ := 5
  let issuesTime : ℕ := 15
  let preGameTime : ℕ := downloadTime + installTime + updateTime + accountTime + issuesTime
  let tutorialTime : ℕ := preGameTime * 3
  totalGamePrepTime downloadTime installTime updateTime accountTime issuesTime tutorialTime = 220 := by
  sorry

#eval totalGamePrepTime 10 5 20 5 15 165

end game_prep_time_calculation_l3981_398104


namespace average_of_five_quantities_l3981_398118

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 24) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 12 := by
  sorry

end average_of_five_quantities_l3981_398118


namespace ashok_subjects_l3981_398180

theorem ashok_subjects (average_all : ℝ) (average_five : ℝ) (sixth_subject : ℝ) 
  (h1 : average_all = 72)
  (h2 : average_five = 74)
  (h3 : sixth_subject = 62) :
  ∃ n : ℕ, n = 6 ∧ n * average_all = 5 * average_five + sixth_subject :=
by
  sorry

end ashok_subjects_l3981_398180


namespace dress_price_calculation_l3981_398146

theorem dress_price_calculation (discount_percent : ℝ) (final_price : ℝ) : 
  discount_percent = 30 → final_price = 35 → (100 - discount_percent) / 100 * 50 = final_price := by
  sorry

end dress_price_calculation_l3981_398146


namespace ant_movement_l3981_398157

theorem ant_movement (initial_position : Int) (move_right1 move_left move_right2 : Int) :
  initial_position = -3 →
  move_right1 = 5 →
  move_left = 9 →
  move_right2 = 1 →
  initial_position + move_right1 - move_left + move_right2 = -6 :=
by sorry

end ant_movement_l3981_398157


namespace sample_size_is_200_l3981_398102

/-- The expected sample size for a school with given student counts and selection probability -/
def expected_sample_size (freshmen sophomores juniors : ℕ) (prob : ℝ) : ℝ :=
  (freshmen + sophomores + juniors : ℝ) * prob

/-- Theorem stating that the expected sample size is 200 for the given school population and selection probability -/
theorem sample_size_is_200 :
  expected_sample_size 280 320 400 0.2 = 200 := by
  sorry

end sample_size_is_200_l3981_398102


namespace parallelogram_area_example_l3981_398151

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : parallelogramArea 12 8 = 96 := by
  sorry

end parallelogram_area_example_l3981_398151


namespace one_root_in_first_quadrant_l3981_398110

def complex_equation (z : ℂ) : Prop := z^7 = -1 + Complex.I * Real.sqrt 3

def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem one_root_in_first_quadrant :
  ∃! z, complex_equation z ∧ is_in_first_quadrant z :=
sorry

end one_root_in_first_quadrant_l3981_398110


namespace ellipse_inscribed_circle_radius_specific_ellipse_inscribed_circle_radius_l3981_398174

/-- The radius of the largest circle inside an ellipse with its center at a focus --/
theorem ellipse_inscribed_circle_radius (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let c := Real.sqrt (a^2 - b^2)
  let r := c - b
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x - c)^2 + y^2 ≥ r^2) ∧
  (∃ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ∧ (x - c)^2 + y^2 = r^2) :=
by sorry

/-- The specific case for the given ellipse --/
theorem specific_ellipse_inscribed_circle_radius :
  let a : ℝ := 6
  let b : ℝ := 5
  let c := Real.sqrt (a^2 - b^2)
  let r := c - a
  r = Real.sqrt 11 - 6 :=
by sorry

end ellipse_inscribed_circle_radius_specific_ellipse_inscribed_circle_radius_l3981_398174


namespace exists_all_cards_moved_no_guarantee_ace_not_next_to_empty_l3981_398132

/-- Represents a deck of cards arranged in a circle with one empty spot. -/
structure CircularDeck :=
  (cards : Fin 52 → Option (Fin 52))
  (empty_spot : Fin 53)
  (injective : ∀ i j, i ≠ j → cards i ≠ cards j)
  (surjective : ∀ c, c ≠ empty_spot → ∃ i, cards i = some c)

/-- Represents a sequence of card namings. -/
def NamingSequence := List (Fin 52)

/-- Predicate to check if a card has moved from its original position. -/
def has_moved (deck : CircularDeck) (card : Fin 52) : Prop :=
  deck.cards card ≠ some card

/-- Predicate to check if the Ace of Spades is next to the empty spot. -/
def ace_next_to_empty (deck : CircularDeck) : Prop :=
  ∃ i, deck.cards i = some 0 ∧ 
    ((i + 1) % 53 = deck.empty_spot ∨ (i - 1 + 53) % 53 = deck.empty_spot)

/-- Theorem stating that there exists a naming sequence that moves all cards. -/
theorem exists_all_cards_moved :
  ∃ (seq : NamingSequence), ∀ (initial_deck : CircularDeck),
    ∃ (final_deck : CircularDeck),
      ∀ (card : Fin 52), has_moved final_deck card :=
sorry

/-- Theorem stating that no naming sequence can guarantee the Ace of Spades
    is not next to the empty spot. -/
theorem no_guarantee_ace_not_next_to_empty :
  ∀ (seq : NamingSequence), ∃ (initial_deck : CircularDeck),
    ∃ (final_deck : CircularDeck),
      ace_next_to_empty final_deck :=
sorry

end exists_all_cards_moved_no_guarantee_ace_not_next_to_empty_l3981_398132


namespace parallel_lines_m_values_l3981_398122

/-- Two lines in the form ax + by + c = 0 are parallel if and only if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- The first line: 2x + (m+1)y + 4 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 1) * y + 4 = 0

/-- The second line: mx + 3y - 2 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 2 = 0

theorem parallel_lines_m_values :
  ∀ m : ℝ, are_parallel 2 (m + 1) m 3 ↔ (m = -3 ∨ m = 2) :=
by sorry

end parallel_lines_m_values_l3981_398122


namespace annual_subscription_cost_is_96_l3981_398164

/-- The cost of a monthly newspaper subscription in dollars. -/
def monthly_cost : ℝ := 10

/-- The discount rate for an annual subscription. -/
def discount_rate : ℝ := 0.2

/-- The number of months in a year. -/
def months_per_year : ℕ := 12

/-- The cost of an annual newspaper subscription with a discount. -/
def annual_subscription_cost : ℝ :=
  monthly_cost * months_per_year * (1 - discount_rate)

/-- Theorem stating that the annual subscription cost is $96. -/
theorem annual_subscription_cost_is_96 :
  annual_subscription_cost = 96 := by
  sorry


end annual_subscription_cost_is_96_l3981_398164


namespace caramel_apple_cost_is_25_l3981_398197

/-- The cost of an ice cream cone in cents -/
def ice_cream_cost : ℕ := 15

/-- The additional cost of a caramel apple compared to an ice cream cone in cents -/
def apple_additional_cost : ℕ := 10

/-- The cost of a caramel apple in cents -/
def caramel_apple_cost : ℕ := ice_cream_cost + apple_additional_cost

/-- Theorem: The cost of a caramel apple is 25 cents -/
theorem caramel_apple_cost_is_25 : caramel_apple_cost = 25 := by
  sorry

end caramel_apple_cost_is_25_l3981_398197


namespace smallest_integer_l3981_398159

theorem smallest_integer (a b : ℕ) (ha : a = 36) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 20) :
  b ≥ 45 ∧ ∃ (b' : ℕ), b' = 45 ∧ Nat.lcm a b' / Nat.gcd a b' = 20 := by
  sorry

end smallest_integer_l3981_398159


namespace sarah_candy_theorem_l3981_398107

/-- The number of candy pieces Sarah received from her neighbors -/
def candy_from_neighbors : ℕ := sorry

/-- The number of candy pieces Sarah received from her older sister -/
def candy_from_sister : ℕ := 15

/-- The number of candy pieces Sarah ate per day -/
def candy_per_day : ℕ := 9

/-- The number of days the candy lasted -/
def days_lasted : ℕ := 9

/-- The total number of candy pieces Sarah had -/
def total_candy : ℕ := candy_per_day * days_lasted

theorem sarah_candy_theorem : candy_from_neighbors = 66 := by
  sorry

end sarah_candy_theorem_l3981_398107


namespace p_minus_q_plus_r_equals_two_thirds_l3981_398165

theorem p_minus_q_plus_r_equals_two_thirds
  (p q r : ℚ)
  (hp : 3 / p = 6)
  (hq : 3 / q = 18)
  (hr : 5 / r = 15) :
  p - q + r = 2 / 3 := by
  sorry

end p_minus_q_plus_r_equals_two_thirds_l3981_398165


namespace geometric_sequence_increasing_condition_l3981_398125

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  (a 1 < 0 ∧ 0 < q ∧ q < 1) →
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) ∧
  ¬(∀ n : ℕ, n > 0 → a (n + 1) > a n → (a 1 < 0 ∧ 0 < q ∧ q < 1)) :=
by sorry

end geometric_sequence_increasing_condition_l3981_398125


namespace beavers_working_on_home_l3981_398115

/-- The number of beavers initially working on their home -/
def initial_beavers : ℕ := 2

/-- The number of beavers that went for a swim -/
def swimming_beavers : ℕ := 1

/-- The number of beavers still working on their home -/
def remaining_beavers : ℕ := initial_beavers - swimming_beavers

theorem beavers_working_on_home : remaining_beavers = 1 := by
  sorry

end beavers_working_on_home_l3981_398115


namespace complex_subtraction_reciprocal_l3981_398178

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_subtraction_reciprocal : i - (1 : ℂ) / i = 2 * i := by
  sorry

end complex_subtraction_reciprocal_l3981_398178


namespace min_draw_same_number_and_suit_min_draw_consecutive_numbers_l3981_398109

/-- Represents a card in the deck -/
structure Card where
  suit : Fin 4
  number : Fin 13

/-- The deck of cards -/
def Deck : Finset Card := sorry

/-- The number of cards in the deck -/
def deck_size : Nat := 52

/-- The number of suits in the deck -/
def num_suits : Nat := 4

/-- The number of cards per suit -/
def cards_per_suit : Nat := 13

theorem min_draw_same_number_and_suit :
  ∀ (S : Finset Card), S ⊆ Deck → S.card = 27 →
    ∃ (c1 c2 : Card), c1 ∈ S ∧ c2 ∈ S ∧ c1 ≠ c2 ∧ c1.number = c2.number ∧ c1.suit = c2.suit :=
sorry

theorem min_draw_consecutive_numbers :
  ∀ (S : Finset Card), S ⊆ Deck → S.card = 37 →
    ∃ (c1 c2 c3 : Card), c1 ∈ S ∧ c2 ∈ S ∧ c3 ∈ S ∧
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      (c1.number + 1 = c2.number ∧ c2.number + 1 = c3.number ∨
       c2.number + 1 = c1.number ∧ c1.number + 1 = c3.number ∨
       c1.number + 1 = c3.number ∧ c3.number + 1 = c2.number) :=
sorry

end min_draw_same_number_and_suit_min_draw_consecutive_numbers_l3981_398109


namespace paths_to_2005_l3981_398134

/-- Represents the number of choices for each step in forming the number 2005 -/
structure PathChoices where
  first_zero : Nat
  second_zero : Nat
  final_five : Nat

/-- Calculates the total number of paths to form 2005 -/
def total_paths (choices : PathChoices) : Nat :=
  choices.first_zero * choices.second_zero * choices.final_five

/-- The given choices for each step in forming 2005 -/
def given_choices : PathChoices :=
  { first_zero := 6
  , second_zero := 2
  , final_five := 3 }

/-- Theorem stating that there are 36 different paths to form 2005 -/
theorem paths_to_2005 : total_paths given_choices = 36 := by
  sorry

#eval total_paths given_choices

end paths_to_2005_l3981_398134


namespace smallest_solution_equation_smallest_solution_l3981_398186

theorem smallest_solution_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 - Real.sqrt 2 ∨ x = 4 + Real.sqrt 2) :=
sorry

theorem smallest_solution (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ (∀ y, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_equation_smallest_solution_l3981_398186


namespace number_of_females_l3981_398121

theorem number_of_females (total : ℕ) (avg_all : ℚ) (avg_male : ℚ) (avg_female : ℚ) 
  (h1 : total = 140)
  (h2 : avg_all = 24)
  (h3 : avg_male = 21)
  (h4 : avg_female = 28) :
  ∃ (females : ℕ), females = 60 ∧ 
    avg_all * total = avg_female * females + avg_male * (total - females) :=
by sorry

end number_of_females_l3981_398121


namespace third_digit_even_l3981_398196

theorem third_digit_even (n : ℤ) : ∃ k : ℤ, (10*n + 5)^2 = 1000*k + 200*m + 25 ∧ m % 2 = 0 :=
sorry

end third_digit_even_l3981_398196


namespace roots_of_equation_l3981_398147

theorem roots_of_equation : ∃ (x₁ x₂ : ℝ), 
  (∀ x : ℝ, (x - 3)^2 = 3 - x ↔ x = x₁ ∨ x = x₂) ∧ 
  x₁ = 3 ∧ x₂ = 2 := by
sorry

end roots_of_equation_l3981_398147


namespace hexagon_segment_length_l3981_398142

/-- A regular hexagon with side length 2 inscribed in a circle -/
structure RegularHexagon :=
  (side_length : ℝ)
  (inscribed_in_circle : Bool)
  (h_side_length : side_length = 2)
  (h_inscribed : inscribed_in_circle = true)

/-- A segment connecting a vertex to the midpoint of the opposite side -/
def opposite_midpoint_segment (h : RegularHexagon) : ℝ → ℝ := sorry

/-- The total length of all segments connecting vertices to opposite midpoints -/
def total_segment_length (h : RegularHexagon) : ℝ :=
  6 * opposite_midpoint_segment h 1

theorem hexagon_segment_length (h : RegularHexagon) :
  total_segment_length h = 4 * Real.sqrt 3 := by
  sorry

end hexagon_segment_length_l3981_398142


namespace trapezoid_median_l3981_398188

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    where the base of the triangle is 24 inches and the sum of the bases of
    the trapezoid is 40 inches, the median of the trapezoid is 20 inches. -/
theorem trapezoid_median (h : ℝ) (triangle_area trapezoid_area : ℝ) 
  (triangle_base trapezoid_base_sum : ℝ) (trapezoid_median : ℝ) :
  h > 0 →
  triangle_area = trapezoid_area →
  triangle_base = 24 →
  trapezoid_base_sum = 40 →
  triangle_area = (1 / 2) * triangle_base * h →
  trapezoid_area = trapezoid_median * h →
  trapezoid_median = trapezoid_base_sum / 2 →
  trapezoid_median = 20 := by
sorry

end trapezoid_median_l3981_398188


namespace balls_remaining_l3981_398150

/-- The number of baskets --/
def num_baskets : ℕ := 5

/-- The number of tennis balls in each basket --/
def tennis_balls_per_basket : ℕ := 15

/-- The number of soccer balls in each basket --/
def soccer_balls_per_basket : ℕ := 5

/-- The number of students who removed 8 balls each --/
def students_eight : ℕ := 3

/-- The number of students who removed 10 balls each --/
def students_ten : ℕ := 2

/-- The number of balls removed by each student in the first group --/
def balls_removed_eight : ℕ := 8

/-- The number of balls removed by each student in the second group --/
def balls_removed_ten : ℕ := 10

/-- Theorem: The number of balls remaining in the baskets is 56 --/
theorem balls_remaining :
  (num_baskets * (tennis_balls_per_basket + soccer_balls_per_basket)) -
  (students_eight * balls_removed_eight + students_ten * balls_removed_ten) = 56 := by
  sorry

end balls_remaining_l3981_398150


namespace all_ciphers_are_good_l3981_398143

/-- Represents a cipher where each letter is replaced by a word. -/
structure Cipher where
  encode : Char → List Char
  decode : List Char → Option Char
  encode_length : ∀ c, (encode c).length ≤ 10

/-- A word is a list of characters. -/
def Word := List Char

/-- Encrypts a word using the given cipher. -/
def encrypt (cipher : Cipher) (word : Word) : Word :=
  word.bind cipher.encode

/-- A cipher is good if any encrypted word can be uniquely decrypted. -/
def is_good_cipher (cipher : Cipher) : Prop :=
  ∀ (w : Word), w.length ≤ 10000 → 
    ∃! (original : Word), encrypt cipher original = w

/-- Main theorem: Any cipher satisfying the given conditions is good. -/
theorem all_ciphers_are_good (cipher : Cipher) : is_good_cipher cipher := by
  sorry

end all_ciphers_are_good_l3981_398143


namespace percentage_failed_hindi_l3981_398163

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_english = 42)
  (h2 : failed_both = 28)
  (h3 : passed_both = 56) :
  ∃ (failed_hindi : ℝ), failed_hindi = 30 ∧
    failed_hindi + failed_english - failed_both = 100 - passed_both :=
by
  sorry

end percentage_failed_hindi_l3981_398163


namespace bird_eggs_problem_l3981_398169

theorem bird_eggs_problem (total_eggs : ℕ) 
  (eggs_per_nest_tree1 : ℕ) (nests_in_tree1 : ℕ) 
  (eggs_in_front_yard : ℕ) (eggs_in_tree2 : ℕ) : 
  total_eggs = 17 →
  eggs_per_nest_tree1 = 5 →
  nests_in_tree1 = 2 →
  eggs_in_front_yard = 4 →
  total_eggs = nests_in_tree1 * eggs_per_nest_tree1 + eggs_in_front_yard + eggs_in_tree2 →
  eggs_in_tree2 = 3 := by
sorry

end bird_eggs_problem_l3981_398169


namespace square_2209_identity_l3981_398185

theorem square_2209_identity (x : ℤ) (h : x^2 = 2209) : (2*x + 1) * (2*x - 1) = 8835 := by
  sorry

end square_2209_identity_l3981_398185


namespace inequality_proof_l3981_398130

theorem inequality_proof (r s : ℝ) (hr : 0 < r) (hs : 0 < s) (hrs : r + s = 1) :
  r^r * s^s + r^s * s^r ≤ 1 := by
  sorry

end inequality_proof_l3981_398130


namespace largest_value_l3981_398106

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  a^2 + b^2 = max a (max (1/2) (max (2*a*b) (a^2 + b^2))) := by
  sorry

end largest_value_l3981_398106


namespace circle_satisfies_conditions_l3981_398198

-- Define the two original circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 3 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 3 = 0

-- Define the line on which the center of the new circle should lie
def center_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the new circle
def new_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 16*y - 3 = 0

-- Theorem statement
theorem circle_satisfies_conditions :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → new_circle x y) ∧
    (∃ (h k : ℝ), center_line h k ∧ 
      ∀ (x y : ℝ), new_circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 12*h + 16*k + 3)) :=
sorry

end circle_satisfies_conditions_l3981_398198


namespace boric_acid_weight_l3981_398181

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Boron in g/mol -/
def atomic_weight_B : ℝ := 10.81

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Hydrogen atoms in Boric acid -/
def num_H : ℕ := 3

/-- The number of Boron atoms in Boric acid -/
def num_B : ℕ := 1

/-- The number of Oxygen atoms in Boric acid -/
def num_O : ℕ := 3

/-- The molecular weight of Boric acid (H3BO3) in g/mol -/
def molecular_weight_boric_acid : ℝ :=
  num_H * atomic_weight_H + num_B * atomic_weight_B + num_O * atomic_weight_O

theorem boric_acid_weight :
  molecular_weight_boric_acid = 61.834 := by sorry

end boric_acid_weight_l3981_398181
