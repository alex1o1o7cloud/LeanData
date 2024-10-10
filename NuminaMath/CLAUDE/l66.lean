import Mathlib

namespace clothing_store_inventory_l66_6603

theorem clothing_store_inventory (belts : ℕ) (black_shirts : ℕ) (white_shirts : ℕ) :
  belts = 40 →
  black_shirts = 63 →
  white_shirts = 42 →
  ∃ (ties : ℕ) (scarves : ℕ) (jeans : ℕ),
    jeans = (2 * (black_shirts + white_shirts)) / 3 ∧
    scarves = (ties + belts) / 2 ∧
    jeans = scarves + 33 ∧
    ties = 34 :=
by sorry

end clothing_store_inventory_l66_6603


namespace specific_quadrilateral_area_l66_6643

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℝ := sorry

/-- The theorem stating that the area of the specific quadrilateral is 4.5 -/
theorem specific_quadrilateral_area :
  let a : Point := ⟨0, 0⟩
  let b : Point := ⟨0, 2⟩
  let c : Point := ⟨3, 2⟩
  let d : Point := ⟨3, 3⟩
  quadrilateralArea a b c d = 4.5 := by sorry

end specific_quadrilateral_area_l66_6643


namespace complement_of_union_in_U_l66_6619

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union_in_U :
  (U \ (M ∪ N)) = {4} := by sorry

end complement_of_union_in_U_l66_6619


namespace intersection_with_complement_l66_6656

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 3}
def B : Set ℕ := {1}

theorem intersection_with_complement : A ∩ (U \ B) = {2, 3} := by
  sorry

end intersection_with_complement_l66_6656


namespace min_value_f_plus_f_l66_6614

/-- The function f(x) = -x³ + ax² - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem min_value_f_plus_f' (a : ℝ) :
  (f' a 1 = 0) →
  (∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ (m' n' : ℝ), m' ∈ Set.Icc (-1 : ℝ) 1 → n' ∈ Set.Icc (-1 : ℝ) 1 →
      f a m + f' a n ≤ f a m' + f' a n') →
  ∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f' a n = -13 :=
by sorry

end min_value_f_plus_f_l66_6614


namespace binomial_12_choose_6_l66_6692

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end binomial_12_choose_6_l66_6692


namespace square_of_complex_l66_6664

theorem square_of_complex (z : ℂ) (i : ℂ) : z = 5 + 3 * i → i^2 = -1 → z^2 = 16 + 30 * i := by
  sorry

end square_of_complex_l66_6664


namespace smaug_gold_coins_l66_6658

/-- Represents the number of coins in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Represents the value of different coin types in terms of copper coins -/
structure CoinValues where
  silver_to_copper : ℕ
  gold_to_silver : ℕ

/-- Calculates the total value of a hoard in copper coins -/
def hoard_value (hoard : DragonHoard) (values : CoinValues) : ℕ :=
  hoard.gold * values.gold_to_silver * values.silver_to_copper +
  hoard.silver * values.silver_to_copper +
  hoard.copper

/-- Theorem stating that Smaug has 100 gold coins -/
theorem smaug_gold_coins : 
  ∀ (hoard : DragonHoard) (values : CoinValues),
    hoard.silver = 60 →
    hoard.copper = 33 →
    values.silver_to_copper = 8 →
    values.gold_to_silver = 3 →
    hoard_value hoard values = 2913 →
    hoard.gold = 100 := by
  sorry

end smaug_gold_coins_l66_6658


namespace circle_radius_circumference_relation_l66_6632

theorem circle_radius_circumference_relation (r : ℝ) (h : r > 0) :
  let c₁ := 2 * Real.pi * r
  let c₂ := 2 * Real.pi * (2 * r)
  c₂ = 2 * c₁ :=
by sorry

end circle_radius_circumference_relation_l66_6632


namespace cake_eating_problem_l66_6653

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem cake_eating_problem : 
  geometric_series_sum (1/3) (1/3) 7 = 1093/2187 := by sorry

end cake_eating_problem_l66_6653


namespace sin_function_value_l66_6626

/-- Given that the terminal side of angle φ passes through point P(3, -4),
    and the distance between two adjacent symmetry axes of the graph of
    the function f(x) = sin(ωx + φ) (ω > 0) is equal to π/2,
    prove that f(π/4) = 3/5 -/
theorem sin_function_value (φ ω : ℝ) (h1 : ω > 0) 
    (h2 : (3 : ℝ) / Real.sqrt (3^2 + 4^2) = Real.cos φ)
    (h3 : (-4 : ℝ) / Real.sqrt (3^2 + 4^2) = Real.sin φ)
    (h4 : π / (2 * ω) = π / 2) :
  Real.sin (ω * (π / 4) + φ) = 3 / 5 := by
  sorry

end sin_function_value_l66_6626


namespace card_game_result_l66_6640

/-- Represents the money distribution in a card game --/
structure MoneyDistribution where
  aldo : ℚ
  bernardo : ℚ
  carlos : ℚ

/-- The card game scenario --/
def CardGame : Type :=
  { game : MoneyDistribution × MoneyDistribution // 
    (game.1.aldo : ℚ) / (game.1.bernardo : ℚ) = 7/6 ∧
    (game.1.bernardo : ℚ) / (game.1.carlos : ℚ) = 6/5 ∧
    (game.2.aldo : ℚ) / (game.2.bernardo : ℚ) = 6/5 ∧
    (game.2.bernardo : ℚ) / (game.2.carlos : ℚ) = 5/4 ∧
    (game.2.aldo - game.1.aldo : ℚ) = 1200 ∨
    (game.2.bernardo - game.1.bernardo : ℚ) = 1200 ∨
    (game.2.carlos - game.1.carlos : ℚ) = 1200 }

/-- The theorem to be proved --/
theorem card_game_result (game : CardGame) :
  game.val.2.aldo = 43200 ∧
  game.val.2.bernardo = 36000 ∧
  game.val.2.carlos = 28800 := by
  sorry


end card_game_result_l66_6640


namespace decimal_addition_subtraction_l66_6666

theorem decimal_addition_subtraction :
  (0.45 : ℚ) - 0.03 + 0.008 = 0.428 := by
  sorry

end decimal_addition_subtraction_l66_6666


namespace fraction_sum_l66_6622

theorem fraction_sum (m n : ℚ) (h : m / n = 3 / 7) : (m + n) / n = 10 / 7 := by
  sorry

end fraction_sum_l66_6622


namespace total_wall_length_l66_6696

/-- Represents the daily work rate of a bricklayer in meters per day -/
def daily_rate : ℕ := 8

/-- Represents the number of working days -/
def working_days : ℕ := 15

/-- Theorem: The total length of wall laid by a bricklayer in 15 days -/
theorem total_wall_length : daily_rate * working_days = 120 := by
  sorry

end total_wall_length_l66_6696


namespace sine_is_periodic_l66_6629

-- Define the properties
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the sine function
def sin : ℝ → ℝ := sorry

-- Theorem statement
theorem sine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric sin →
  IsPeriodic sin := by sorry

end sine_is_periodic_l66_6629


namespace technician_average_salary_l66_6604

/-- Calculates the average salary of technicians in a workshop --/
theorem technician_average_salary
  (total_workers : ℕ)
  (total_average : ℝ)
  (non_tech_average : ℝ)
  (num_technicians : ℕ)
  (h1 : total_workers = 14)
  (h2 : total_average = 10000)
  (h3 : non_tech_average = 8000)
  (h4 : num_technicians = 7) :
  (total_workers * total_average - (total_workers - num_technicians) * non_tech_average) / num_technicians = 12000 := by
sorry

end technician_average_salary_l66_6604


namespace weight_of_four_moles_l66_6630

/-- Given a compound with a molecular weight of 260, prove that 4 moles of this compound weighs 1040 grams. -/
theorem weight_of_four_moles (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 260 → moles = 4 → moles * molecular_weight = 1040 := by
  sorry

end weight_of_four_moles_l66_6630


namespace sam_oatmeal_cookies_l66_6662

/-- Given a total number of cookies and a ratio of three types of cookies,
    calculate the number of cookies of the second type. -/
def oatmealCookies (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  let totalParts := ratio1 + ratio2 + ratio3
  let cookiesPerPart := total / totalParts
  ratio2 * cookiesPerPart

/-- Theorem stating that given 36 total cookies and a ratio of 2:3:4,
    the number of oatmeal cookies is 12. -/
theorem sam_oatmeal_cookies :
  oatmealCookies 36 2 3 4 = 12 := by
  sorry

end sam_oatmeal_cookies_l66_6662


namespace symmetric_points_sum_l66_6649

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m, 5) (3, n) → m + n = -8 := by
  sorry

end symmetric_points_sum_l66_6649


namespace min_columns_for_formation_l66_6606

theorem min_columns_for_formation (n : ℕ) : n ≥ 141 → ∃ k : ℕ, 8 * n = 225 * k + 3 :=
sorry

end min_columns_for_formation_l66_6606


namespace bicycle_price_problem_l66_6636

theorem bicycle_price_problem (cp_a : ℝ) (sp_b sp_c : ℝ) : 
  sp_b = 1.5 * cp_a →
  sp_c = 1.25 * sp_b →
  sp_c = 225 →
  cp_a = 120 := by
sorry

end bicycle_price_problem_l66_6636


namespace rita_swimming_months_l66_6642

/-- The number of months Rita needs to fulfill her coach's requirements -/
def months_to_fulfill_requirement (total_required_hours : ℕ) (hours_already_completed : ℕ) (hours_per_month : ℕ) : ℕ :=
  (total_required_hours - hours_already_completed) / hours_per_month

/-- Proof that Rita needs 6 months to fulfill her coach's requirements -/
theorem rita_swimming_months : 
  months_to_fulfill_requirement 1500 180 220 = 6 := by
sorry

end rita_swimming_months_l66_6642


namespace annes_speed_l66_6623

theorem annes_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 6) 
  (h2 : time = 3) 
  (h3 : speed = distance / time) : 
  speed = 2 := by
sorry

end annes_speed_l66_6623


namespace value_of_n_l66_6651

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters on the left side of the equation -/
def left_quarters : ℕ := 15

/-- The number of nickels on the left side of the equation -/
def left_nickels : ℕ := 18

/-- The number of quarters on the right side of the equation -/
def right_quarters : ℕ := 7

/-- Theorem stating that the value of n is 58 -/
theorem value_of_n : 
  ∃ n : ℕ, 
    left_quarters * quarter_value + left_nickels * nickel_value = 
    right_quarters * quarter_value + n * nickel_value ∧ 
    n = 58 := by
  sorry

end value_of_n_l66_6651


namespace midpoint_coordinate_product_l66_6611

/-- Given a line segment with endpoints (4, -7) and (-8, 9), 
    the product of the coordinates of its midpoint is -2. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -7
  let x2 : ℝ := -8
  let y2 : ℝ := 9
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -2 := by
  sorry

end midpoint_coordinate_product_l66_6611


namespace alice_flour_measurement_l66_6697

/-- The number of times Alice needs to fill her measuring cup to get the required amount of flour -/
def number_of_fills (total_flour : ℚ) (cup_capacity : ℚ) : ℚ :=
  total_flour / cup_capacity

/-- Theorem: Alice needs to fill her ⅓ cup measuring cup 10 times to get 3⅓ cups of flour -/
theorem alice_flour_measurement :
  number_of_fills (3 + 1/3) (1/3) = 10 := by
  sorry

end alice_flour_measurement_l66_6697


namespace at_least_one_greater_than_one_l66_6680

theorem at_least_one_greater_than_one (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) : max x y > 1 := by
  sorry

end at_least_one_greater_than_one_l66_6680


namespace cube_opposite_face_l66_6600

structure Cube where
  faces : Finset Char
  adjacent : Char → Char → Prop

def opposite (c : Cube) (x y : Char) : Prop :=
  x ∈ c.faces ∧ y ∈ c.faces ∧ x ≠ y ∧ ¬c.adjacent x y

theorem cube_opposite_face (c : Cube) :
  c.faces = {'А', 'Б', 'В', 'Г', 'Д', 'Е'} →
  c.adjacent 'В' 'А' →
  c.adjacent 'В' 'Д' →
  c.adjacent 'В' 'Е' →
  opposite c 'В' 'Г' := by
  sorry

end cube_opposite_face_l66_6600


namespace sauce_per_pulled_pork_sandwich_l66_6688

/-- The amount of sauce each pulled pork sandwich takes -/
def pulled_pork_sauce : ℚ :=
  1 / 6

theorem sauce_per_pulled_pork_sandwich 
  (total_sauce : ℚ) 
  (burger_sauce : ℚ) 
  (num_burgers : ℕ) 
  (num_pulled_pork : ℕ) 
  (h1 : total_sauce = 5)
  (h2 : burger_sauce = 1 / 4)
  (h3 : num_burgers = 8)
  (h4 : num_pulled_pork = 18)
  (h5 : num_burgers * burger_sauce + num_pulled_pork * pulled_pork_sauce = total_sauce) :
  pulled_pork_sauce = 1 / 6 := by
  sorry

end sauce_per_pulled_pork_sandwich_l66_6688


namespace vector_parallel_sum_l66_6689

/-- Given vectors a and b in ℝ², if a is parallel to (a + b), then the y-coordinate of b is -1/2. -/
theorem vector_parallel_sum (a b : ℝ × ℝ) (h : a = (4, -1)) (h' : b.1 = 2) :
  (∃ (k : ℝ), k • a = a + b) → b.2 = -1/2 := by
  sorry

end vector_parallel_sum_l66_6689


namespace monotonically_decreasing_iff_a_leq_neg_three_l66_6681

/-- A function f is monotonically decreasing on an interval [a, b] if for any x₁, x₂ in [a, b] with x₁ < x₂, we have f(x₁) ≥ f(x₂) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b → f x₁ ≥ f x₂

/-- The quadratic function f(x) = x² + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotonically_decreasing_iff_a_leq_neg_three :
  ∀ a : ℝ, MonotonicallyDecreasing (f a) (-2) 4 ↔ a ≤ -3 := by sorry


end monotonically_decreasing_iff_a_leq_neg_three_l66_6681


namespace power_sum_equals_product_l66_6657

theorem power_sum_equals_product (m n : ℕ+) (a b : ℝ) 
  (h1 : 3^(m.val) = a) (h2 : 3^(n.val) = b) : 
  3^(m.val + n.val) = a * b := by sorry

end power_sum_equals_product_l66_6657


namespace abs_m_minus_n_equals_five_l66_6605

theorem abs_m_minus_n_equals_five (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end abs_m_minus_n_equals_five_l66_6605


namespace g_satisfies_equation_l66_6644

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -4 * x^5 + 7 * x^3 - 5 * x^2 - x + 6

-- State the theorem
theorem g_satisfies_equation : ∀ x : ℝ, 4 * x^5 - 3 * x^3 + x + g x = 7 * x^3 - 5 * x^2 + 6 := by
  sorry

end g_satisfies_equation_l66_6644


namespace divisibility_by_1987_l66_6685

def odd_product : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (2 * i + 1)) 1

def even_product : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (2 * i + 2)) 1

theorem divisibility_by_1987 : ∃ k : ℤ, (odd_product 993 + even_product 993 : ℤ) = k * 1987 := by
  sorry

end divisibility_by_1987_l66_6685


namespace least_addition_for_divisibility_by_nine_l66_6645

def original_number : ℕ := 228712

theorem least_addition_for_divisibility_by_nine :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬((original_number + m) % 9 = 0)) ∧
  ((original_number + n) % 9 = 0) :=
sorry

end least_addition_for_divisibility_by_nine_l66_6645


namespace zeros_after_one_in_500_to_150_l66_6655

-- Define 500 as 5 * 10^2
def five_hundred : ℕ := 5 * 10^2

-- Theorem statement
theorem zeros_after_one_in_500_to_150 :
  (∃ n : ℕ, five_hundred^150 = 10^n * (1 + 10 * m) ∧ m < 10) ∧
  (∀ k : ℕ, five_hundred^150 = 10^k * (1 + 10 * m) ∧ m < 10 → k = 300) :=
sorry

end zeros_after_one_in_500_to_150_l66_6655


namespace x_plus_y_equals_22_l66_6683

theorem x_plus_y_equals_22 (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (25 : ℝ) ^ y = 5 ^ (x - 16)) : 
  x + y = 22 := by sorry

end x_plus_y_equals_22_l66_6683


namespace cousins_ages_sum_l66_6610

def is_single_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cousins_ages_sum :
  ∀ (a b c d : ℕ),
    is_single_digit a ∧ is_single_digit b ∧ is_single_digit c ∧ is_single_digit d →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    ((a * b = 24 ∧ c * d = 30) ∨ (a * c = 24 ∧ b * d = 30) ∨ (a * d = 24 ∧ b * c = 30)) →
    a + b + c + d = 22 :=
by sorry

end cousins_ages_sum_l66_6610


namespace triangle_third_side_l66_6677

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 6 → c % 2 = 1 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (c > b - a ∧ c < b + a) →
  c = 5 ∨ c = 7 := by
sorry

end triangle_third_side_l66_6677


namespace megan_bottles_left_l66_6684

/-- Calculates the number of bottles Megan has left after drinking and giving away some bottles. -/
def bottles_left (initial : ℕ) (drank : ℕ) (given_away : ℕ) : ℕ :=
  initial - (drank + given_away)

/-- Theorem stating that Megan has 25 bottles left after starting with 45, drinking 8, and giving away 12. -/
theorem megan_bottles_left : bottles_left 45 8 12 = 25 := by
  sorry

end megan_bottles_left_l66_6684


namespace factorial_inequality_l66_6633

/-- A function satisfying the given property -/
def special_function (f : ℕ → ℕ) : Prop :=
  ∀ w x y z : ℕ, f (f (f z)) * f (w * x * f (y * f z)) = z^2 * f (x * f y) * f w

/-- The main theorem -/
theorem factorial_inequality (f : ℕ → ℕ) (h : special_function f) : 
  ∀ n : ℕ, f (n.factorial) ≥ n.factorial :=
sorry

end factorial_inequality_l66_6633


namespace line_perpendicular_to_plane_l66_6612

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel m n → perpendicular n α :=
sorry

end line_perpendicular_to_plane_l66_6612


namespace museum_ticket_cost_l66_6695

/-- The cost of tickets at a museum --/
theorem museum_ticket_cost (adult_price : ℝ) : 
  (7 * adult_price + 5 * (adult_price / 2) = 35) →
  (10 * adult_price + 8 * (adult_price / 2) = 51.58) := by
  sorry

end museum_ticket_cost_l66_6695


namespace composite_function_evaluation_l66_6682

def f (x : ℝ) : ℝ := 5 * x + 2

def g (x : ℝ) : ℝ := 3 * x + 4

theorem composite_function_evaluation :
  f (g (f 3)) = 277 :=
by sorry

end composite_function_evaluation_l66_6682


namespace solve_equation_l66_6618

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 1)

-- State the theorem
theorem solve_equation (a : ℝ) : f (f a) = f 9 + 1 → a = -1/4 := by
  sorry

end solve_equation_l66_6618


namespace union_of_A_and_B_l66_6637

def A : Set ℤ := {-1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 3} := by sorry

end union_of_A_and_B_l66_6637


namespace power_greater_than_square_plus_one_l66_6679

theorem power_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  sorry

end power_greater_than_square_plus_one_l66_6679


namespace ice_cream_flavors_l66_6641

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of new ice cream flavors -/
def new_flavors : ℕ := distribute 5 5

theorem ice_cream_flavors : new_flavors = 126 := by sorry

end ice_cream_flavors_l66_6641


namespace negation_equivalence_l66_6609

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 3 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 2*x - 3 ≤ 0) :=
by sorry

end negation_equivalence_l66_6609


namespace sixth_graders_count_l66_6654

/-- The number of fifth graders -/
def fifth_graders : ℕ := 109

/-- The number of seventh graders -/
def seventh_graders : ℕ := 118

/-- The number of teachers -/
def teachers : ℕ := 4

/-- The number of parents per grade -/
def parents_per_grade : ℕ := 2

/-- The number of buses -/
def buses : ℕ := 5

/-- The number of seats per bus -/
def seats_per_bus : ℕ := 72

/-- The total number of seats available -/
def total_seats : ℕ := buses * seats_per_bus

/-- The total number of chaperones -/
def total_chaperones : ℕ := (teachers + parents_per_grade) * 3

/-- The number of students and chaperones excluding sixth graders -/
def non_sixth_grade_total : ℕ := fifth_graders + seventh_graders + total_chaperones

theorem sixth_graders_count : total_seats - non_sixth_grade_total = 115 := by
  sorry

end sixth_graders_count_l66_6654


namespace three_numbers_sum_l66_6694

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + c = 60 →
  a + b + c = 70 := by
sorry

end three_numbers_sum_l66_6694


namespace arithmetic_sequence_2015th_term_l66_6620

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) + a n = 4 * n - 58

theorem arithmetic_sequence_2015th_term (a : ℕ → ℤ) 
  (h : arithmetic_sequence a) : a 2015 = 4000 := by
  sorry

end arithmetic_sequence_2015th_term_l66_6620


namespace modular_inverse_35_mod_37_l66_6693

theorem modular_inverse_35_mod_37 : ∃ x : ℕ, x ≤ 36 ∧ (35 * x) % 37 = 1 :=
by
  use 18
  sorry

end modular_inverse_35_mod_37_l66_6693


namespace quiz_score_problem_l66_6661

theorem quiz_score_problem (total_questions : ℕ) 
  (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 20 ∧ 
  correct_points = 7 ∧ 
  incorrect_points = -4 ∧ 
  total_score = 100 → 
  ∃ (correct incorrect blank : ℕ), 
    correct + incorrect + blank = total_questions ∧ 
    correct_points * correct + incorrect_points * incorrect = total_score ∧ 
    blank = 1 :=
by sorry

end quiz_score_problem_l66_6661


namespace correct_tile_count_l66_6673

/-- The dimensions of the room --/
def room_width : ℝ := 8
def room_height : ℝ := 12

/-- The dimensions of a tile --/
def tile_width : ℝ := 1.5
def tile_height : ℝ := 2

/-- The number of tiles needed to cover the room --/
def tiles_needed : ℕ := 32

/-- Theorem stating that the number of tiles needed is correct --/
theorem correct_tile_count : 
  (room_width * room_height) / (tile_width * tile_height) = tiles_needed := by
  sorry

end correct_tile_count_l66_6673


namespace triangle_area_with_given_base_and_height_l66_6625

/-- The area of a triangle with base 12 cm and height 15 cm is 90 cm². -/
theorem triangle_area_with_given_base_and_height :
  let base : ℝ := 12
  let height : ℝ := 15
  let area : ℝ := (1 / 2) * base * height
  area = 90 := by sorry

end triangle_area_with_given_base_and_height_l66_6625


namespace stamp_sale_value_l66_6601

def total_stamps : ℕ := 75
def stamps_of_one_kind : ℕ := 40
def value_type1 : ℚ := 5 / 100
def value_type2 : ℚ := 8 / 100

theorem stamp_sale_value :
  ∃ (type1_count type2_count : ℕ),
    type1_count + type2_count = total_stamps ∧
    (type1_count = stamps_of_one_kind ∨ type2_count = stamps_of_one_kind) ∧
    type1_count * value_type1 + type2_count * value_type2 = 48 / 10 := by
  sorry

end stamp_sale_value_l66_6601


namespace circumcircle_radius_of_three_spheres_l66_6616

/-- Given two spheres touching a plane at points B and C, with sum of radii 11 and distance between
    centers 5√17, and a third sphere of radius 8 at point A externally tangent to the other two,
    the radius of the circumcircle of triangle ABC is 2√19. -/
theorem circumcircle_radius_of_three_spheres (R1 R2 : ℝ) (d : ℝ) (R3 : ℝ) :
  R1 + R2 = 11 →
  d = 5 * Real.sqrt 17 →
  R3 = 8 →
  R1 + R2 + 2 * R3 = d →
  ∃ (R : ℝ), R = 2 * Real.sqrt 19 ∧ R = d / 2 :=
by sorry

end circumcircle_radius_of_three_spheres_l66_6616


namespace quadratic_roots_condition_l66_6691

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x - 3 = 0 ∧ a * y^2 + 2 * y - 3 = 0) → a > -1/3 :=
by sorry

end quadratic_roots_condition_l66_6691


namespace coefficient_x5_in_expansion_l66_6624

theorem coefficient_x5_in_expansion : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * (2 ^ (7 - k)) * if k == 5 then 1 else 0) = 84 := by
  sorry

end coefficient_x5_in_expansion_l66_6624


namespace probability_theorem_l66_6635

def standard_dice : ℕ := 6

def roll_count : ℕ := 4

def probability_at_least_three_distinct_with_six : ℚ :=
  360 / (standard_dice ^ roll_count)

theorem probability_theorem :
  probability_at_least_three_distinct_with_six = 5 / 18 :=
by sorry

end probability_theorem_l66_6635


namespace circle_equation_range_l66_6672

/-- A circle in the xy-plane can be represented by an equation of the form
    (x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius. -/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The equation x^2 + y^2 + 2kx + 4y + 3k + 8 = 0 represents a circle for some real k -/
def equation (k : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8

/-- The range of k for which the equation represents a circle -/
def k_range (k : ℝ) : Prop :=
  k < -1 ∨ k > 4

theorem circle_equation_range :
  ∀ k, is_circle (equation k) ↔ k_range k :=
sorry

end circle_equation_range_l66_6672


namespace smallest_right_triangle_area_l66_6652

theorem smallest_right_triangle_area :
  let a : ℝ := 6
  let b : ℝ := 8
  let area1 : ℝ := (1/2) * a * b
  let area2 : ℝ := (1/2) * a * Real.sqrt (b^2 - a^2)
  min area1 area2 = (3 : ℝ) * Real.sqrt 28 := by
sorry

end smallest_right_triangle_area_l66_6652


namespace absolute_value_square_l66_6690

theorem absolute_value_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end absolute_value_square_l66_6690


namespace right_triangle_third_side_l66_6631

theorem right_triangle_third_side
  (m n : ℝ)
  (h1 : |m - 3| + Real.sqrt (n - 4) = 0)
  (h2 : m > 0 ∧ n > 0)
  (h3 : ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ ((a = m ∧ b = n) ∨ (a = m ∧ c = n) ∨ (b = m ∧ c = n)))
  : ∃ (x : ℝ), (x = 5 ∨ x = Real.sqrt 7) ∧
    ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ ((a = m ∧ b = n ∧ c = x) ∨ (a = m ∧ c = n ∧ b = x) ∨ (b = m ∧ c = n ∧ a = x)) :=
by sorry

end right_triangle_third_side_l66_6631


namespace imaginary_part_of_inverse_one_plus_i_squared_l66_6665

theorem imaginary_part_of_inverse_one_plus_i_squared (i : ℂ) (h : i * i = -1) :
  Complex.im (1 / ((1 : ℂ) + i)^2) = -(1/2) := by sorry

end imaginary_part_of_inverse_one_plus_i_squared_l66_6665


namespace solve_bowling_problem_l66_6668

def bowling_problem (score1 score2 average : ℕ) : Prop :=
  ∃ score3 : ℕ, 
    (score1 + score2 + score3) / 3 = average ∧
    score3 = 3 * average - score1 - score2

theorem solve_bowling_problem : 
  bowling_problem 113 85 106 → ∃ score3 : ℕ, score3 = 120 := by
  sorry

#check solve_bowling_problem

end solve_bowling_problem_l66_6668


namespace managers_salary_l66_6638

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) : 
  num_employees = 18 →
  avg_salary = 2000 →
  salary_increase = 200 →
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 5800 := by
sorry

end managers_salary_l66_6638


namespace number_problem_l66_6676

theorem number_problem (x : ℝ) : (36 / 100 * x = 129.6) → x = 360 := by
  sorry

end number_problem_l66_6676


namespace partial_fraction_decomposition_l66_6621

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 9) (h2 : x ≠ -4) :
  (5 * x + 7) / (x^2 - 5*x - 36) = 4 / (x - 9) + 1 / (x + 4) := by
  sorry

#check partial_fraction_decomposition

end partial_fraction_decomposition_l66_6621


namespace largest_five_digit_multiple_of_3_and_4_l66_6617

theorem largest_five_digit_multiple_of_3_and_4 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ 3 ∣ n ∧ 4 ∣ n → n ≤ 99996 :=
by sorry

end largest_five_digit_multiple_of_3_and_4_l66_6617


namespace point_coordinates_l66_6699

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (p : Point) 
  (h1 : SecondQuadrant p) 
  (h2 : DistanceToXAxis p = 3) 
  (h3 : DistanceToYAxis p = 1) : 
  p = Point.mk (-1) 3 := by
  sorry

end point_coordinates_l66_6699


namespace coffee_table_books_l66_6628

/-- Represents the number of books Henry has in different locations and actions he takes. -/
structure HenryBooks where
  total : ℕ
  boxed : ℕ
  boxCount : ℕ
  roomDonate : ℕ
  kitchenDonate : ℕ
  newPickup : ℕ
  finalCount : ℕ

/-- Calculates the number of books on Henry's coffee table. -/
def booksOnCoffeeTable (h : HenryBooks) : ℕ :=
  h.total - (h.boxed * h.boxCount + h.roomDonate + h.kitchenDonate) - (h.finalCount - h.newPickup)

/-- Theorem stating that the number of books on Henry's coffee table is 4. -/
theorem coffee_table_books :
  let h : HenryBooks := {
    total := 99,
    boxed := 15,
    boxCount := 3,
    roomDonate := 21,
    kitchenDonate := 18,
    newPickup := 12,
    finalCount := 23
  }
  booksOnCoffeeTable h = 4 := by
  sorry

end coffee_table_books_l66_6628


namespace soccer_match_players_l66_6634

theorem soccer_match_players (total_socks : ℕ) (socks_per_player : ℕ) : 
  total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end soccer_match_players_l66_6634


namespace bird_count_l66_6648

theorem bird_count (total_wings : ℕ) (wings_per_bird : ℕ) (h1 : total_wings = 20) (h2 : wings_per_bird = 2) :
  total_wings / wings_per_bird = 10 := by
sorry

end bird_count_l66_6648


namespace eccentricity_of_ellipse_l66_6602

-- Define the complex polynomial
def polynomial (z : ℂ) : ℂ := (z - 2) * (z^2 + 3*z + 5) * (z^2 + 5*z + 8)

-- Define the set of solutions
def solutions : Set ℂ := {z : ℂ | polynomial z = 0}

-- Define the ellipse centered at the origin
def ellipse (a b : ℝ) : Set ℂ := {z : ℂ | (z.re^2 / a^2) + (z.im^2 / b^2) = 1}

-- Theorem statement
theorem eccentricity_of_ellipse :
  ∀ a b : ℝ, a > 0 → b > 0 →
  (∃ e : Set ℂ, e = ellipse a b ∧ solutions ⊆ e) →
  (a^2 - b^2) / a^2 = 5/16 :=
sorry

end eccentricity_of_ellipse_l66_6602


namespace officers_count_l66_6639

/-- The number of ways to choose 4 distinct officers from a group of 15 people -/
def choose_officers (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3)

/-- Theorem: There are 32,760 ways to choose 4 distinct officers from a group of 15 people -/
theorem officers_count : choose_officers 15 = 32760 := by
  sorry

end officers_count_l66_6639


namespace cube_root_of_product_l66_6687

theorem cube_root_of_product (a b c : ℕ) : 
  (2^6 * 3^3 * 5^3 : ℝ)^(1/3) = 60 := by sorry

end cube_root_of_product_l66_6687


namespace hari_well_digging_time_l66_6698

theorem hari_well_digging_time 
  (jake_time : ℝ) 
  (paul_time : ℝ) 
  (combined_time : ℝ) 
  (h : jake_time = 16)
  (i : paul_time = 24)
  (j : combined_time = 8)
  : ∃ (hari_time : ℝ), 
    1 / jake_time + 1 / paul_time + 1 / hari_time = 1 / combined_time ∧ 
    hari_time = 48 := by
  sorry

end hari_well_digging_time_l66_6698


namespace problem_solution_l66_6678

-- Given equation
def equation (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0

-- Define the system of equations
def system (a b c x y : ℝ) : Prop :=
  a*x + b*y = 30 ∧ c*x + a*y = 28

-- Define the quadratic equation
def quadratic (a b m x : ℝ) : Prop :=
  a*x^2 + b*x + m = 0

theorem problem_solution :
  ∀ a b c : ℝ, equation a b c →
  (∃ x y : ℝ, (a = 3 ∧ b = 4 ∧ c = 5) → system a b c x y ∧ x = 2 ∧ y = 6) ∧
  (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) ∧
  (∀ m : ℝ, (∃ x : ℝ, quadratic a b m x) → m ≤ 4/3) :=
by sorry

end problem_solution_l66_6678


namespace simplify_and_evaluate_l66_6667

theorem simplify_and_evaluate (a : ℚ) (h : a = -3/2) :
  (a + 2 - 5 / (a - 2)) / ((2 * a^2 - 6 * a) / (a - 2)) = -1/2 := by
  sorry

end simplify_and_evaluate_l66_6667


namespace negative_three_triangle_four_equals_seven_l66_6669

-- Define the ▲ operation
def triangle (a b : ℚ) : ℚ := -a + b

-- Theorem statement
theorem negative_three_triangle_four_equals_seven :
  triangle (-3) 4 = 7 := by
  sorry

end negative_three_triangle_four_equals_seven_l66_6669


namespace travelers_checks_worth_l66_6663

/-- Represents the total worth of travelers checks -/
def total_worth (num_50 : ℕ) (num_100 : ℕ) : ℕ :=
  50 * num_50 + 100 * num_100

/-- Represents the average value of remaining checks after spending some $50 checks -/
def average_remaining (num_50 : ℕ) (num_100 : ℕ) (spent_50 : ℕ) : ℚ :=
  (50 * (num_50 - spent_50) + 100 * num_100) / (num_50 + num_100 - spent_50)

theorem travelers_checks_worth :
  ∀ (num_50 num_100 : ℕ),
    num_50 + num_100 = 30 →
    average_remaining num_50 num_100 15 = 70 →
    total_worth num_50 num_100 = 1800 :=
by sorry

end travelers_checks_worth_l66_6663


namespace sum_of_squares_l66_6674

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 70) → (a + b + c = 17) → (a^2 + b^2 + c^2 = 149) := by
  sorry

end sum_of_squares_l66_6674


namespace monge_point_properties_l66_6671

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The Monge point of a tetrahedron -/
def mongePoint (t : Tetrahedron) : Point3D := sorry

/-- Checks if a point lies on a plane defined by three other points -/
def isOnPlane (p q r s : Point3D) : Prop := sorry

/-- The projection of a point onto a plane -/
def projection (p : Point3D) (plane : Point3D × Point3D × Point3D) : Point3D := sorry

/-- The intersection point of the altitudes of a triangular face -/
def altitudeIntersection (face : Point3D × Point3D × Point3D) : Point3D := sorry

/-- The center of the circumscribed circle of a triangular face -/
def circumcenter (face : Point3D × Point3D × Point3D) : Point3D := sorry

/-- Checks if four points are coplanar -/
def areCoplanar (p q r s : Point3D) : Prop := sorry

theorem monge_point_properties (t : Tetrahedron) : 
  isOnPlane (mongePoint t) t.A t.B t.C →
  let D1 := projection t.D (t.A, t.B, t.C)
  (areCoplanar t.D 
    (altitudeIntersection (t.D, t.A, t.B))
    (altitudeIntersection (t.D, t.B, t.C))
    (altitudeIntersection (t.D, t.A, t.C))) ∧
  (areCoplanar t.D
    (circumcenter (t.D, t.A, t.B))
    (circumcenter (t.D, t.B, t.C))
    (circumcenter (t.D, t.A, t.C))) := by
  sorry

end monge_point_properties_l66_6671


namespace typist_margin_width_l66_6646

/-- Proves that for a 20x30 cm sheet with 3 cm margins on top and bottom,
    if 64% is used for typing, the side margins are 2 cm wide. -/
theorem typist_margin_width (x : ℝ) : 
  x > 0 →                             -- side margin is positive
  x < 10 →                            -- side margin is less than half the sheet width
  (20 - 2*x) * 24 = 0.64 * 600 →      -- 64% of sheet is used for typing
  x = 2 := by
sorry

end typist_margin_width_l66_6646


namespace cubic_roots_determinant_l66_6670

/-- Given a cubic equation x^3 - px^2 + qx - r = 0 with roots a, b, c,
    the determinant of the matrix
    |a 0 1|
    |0 b 1|
    |1 1 c|
    is equal to r - a - b -/
theorem cubic_roots_determinant (p q r a b c : ℝ) : 
  a^3 - p*a^2 + q*a - r = 0 →
  b^3 - p*b^2 + q*b - r = 0 →
  c^3 - p*c^2 + q*c - r = 0 →
  Matrix.det !![a, 0, 1; 0, b, 1; 1, 1, c] = r - a - b :=
sorry

end cubic_roots_determinant_l66_6670


namespace odd_function_implies_a_equals_negative_one_l66_6647

def f (a : ℝ) (x : ℝ) : ℝ := x - a - 1

theorem odd_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a (-x) = -(f a x)) → a = -1 := by
  sorry

end odd_function_implies_a_equals_negative_one_l66_6647


namespace locus_of_point_P_l66_6608

/-- The locus of point P given a line and specific conditions --/
theorem locus_of_point_P (x y m n : ℝ) :
  (m / 4 + n / 3 = 1) → -- M(m, n) is on the line l
  (x - m = -2 * x) →    -- Condition from AP = 2PB
  (y = 2 * n - 2 * y) → -- Condition from AP = 2PB
  (3 * x / 4 + y / 2 = 1) := by
sorry


end locus_of_point_P_l66_6608


namespace susies_golden_comets_l66_6686

theorem susies_golden_comets (susie_rir : ℕ) (britney_total susie_total : ℕ) : ℕ :=
  let susie_gc := britney_total - susie_total - 8
  have h1 : susie_rir = 11 := by sorry
  have h2 : britney_total = susie_total + 8 := by sorry
  have h3 : britney_total = 2 * susie_rir + susie_gc / 2 := by sorry
  have h4 : susie_total = susie_rir + susie_gc := by sorry
  6

#check susies_golden_comets

end susies_golden_comets_l66_6686


namespace money_left_over_l66_6650

/-- The amount of money left over after purchasing bread, peanut butter, and honey with a discount coupon. -/
theorem money_left_over (bread_price : ℝ) (peanut_butter_price : ℝ) (honey_price : ℝ)
  (bread_quantity : ℕ) (peanut_butter_quantity : ℕ) (honey_quantity : ℕ)
  (discount : ℝ) (initial_money : ℝ) :
  bread_price = 2.35 →
  peanut_butter_price = 3.10 →
  honey_price = 4.50 →
  bread_quantity = 4 →
  peanut_butter_quantity = 2 →
  honey_quantity = 1 →
  discount = 2 →
  initial_money = 20 →
  initial_money - (bread_price * bread_quantity + peanut_butter_price * peanut_butter_quantity + 
    honey_price * honey_quantity - discount) = 1.90 := by
  sorry

end money_left_over_l66_6650


namespace sin_double_angle_special_l66_6613

/-- Given an angle θ with specific properties, prove that sin(2θ) = -√3/2 -/
theorem sin_double_angle_special (θ : Real) : 
  (∃ (x y : Real), x > 0 ∧ y = -Real.sqrt 3 * x ∧ 
    Real.cos θ = x / Real.sqrt (x^2 + y^2) ∧
    Real.sin θ = y / Real.sqrt (x^2 + y^2)) →
  Real.sin (2 * θ) = -Real.sqrt 3 / 2 := by
  sorry

end sin_double_angle_special_l66_6613


namespace combined_swimming_distance_is_1890_l66_6627

/-- Calculates the combined swimming distance for Jamir, Sarah, and Julien over a week. -/
def combinedSwimmingDistance (julienDailyDistance : ℕ) (daysInWeek : ℕ) : ℕ :=
  let sarahDailyDistance := 2 * julienDailyDistance
  let jamirDailyDistance := sarahDailyDistance + 20
  (julienDailyDistance + sarahDailyDistance + jamirDailyDistance) * daysInWeek

/-- Proves that the combined swimming distance for Jamir, Sarah, and Julien over a week is 1890 meters. -/
theorem combined_swimming_distance_is_1890 :
  combinedSwimmingDistance 50 7 = 1890 := by
  sorry

end combined_swimming_distance_is_1890_l66_6627


namespace no_three_digit_number_satisfies_conditions_l66_6615

/-- Function to check if digits are different and in ascending order -/
def digits_ascending_different (n : ℕ) : Prop := sorry

/-- Theorem stating that no three-digit number satisfies the given conditions -/
theorem no_three_digit_number_satisfies_conditions :
  ¬ ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    digits_ascending_different n ∧
    digits_ascending_different (n^2) ∧
    digits_ascending_different (n^3) := by
  sorry

end no_three_digit_number_satisfies_conditions_l66_6615


namespace larger_number_proof_l66_6659

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1355)
  (h2 : L = 6 * S + 15) : 
  L = 1623 := by
  sorry

end larger_number_proof_l66_6659


namespace gcd_b_81_is_3_l66_6675

theorem gcd_b_81_is_3 (a b : ℤ) : 
  (∃ (x : ℝ), x^2 = 2 ∧ (1 + x)^2012 = a + b * x) → Nat.gcd b.natAbs 81 = 3 := by
  sorry

end gcd_b_81_is_3_l66_6675


namespace base4_77_last_digit_l66_6660

def base4LastDigit (n : Nat) : Nat :=
  n % 4

theorem base4_77_last_digit :
  base4LastDigit 77 = 1 := by
  sorry

end base4_77_last_digit_l66_6660


namespace credit_card_problem_l66_6607

/-- Calculates the amount added to a credit card in the second month given the initial balance,
    interest rate, and final balance after two months. -/
def amount_added (initial_balance : ℚ) (interest_rate : ℚ) (final_balance : ℚ) : ℚ :=
  let first_month_balance := initial_balance * (1 + interest_rate)
  let x := (final_balance - first_month_balance * (1 + interest_rate)) / (1 + interest_rate)
  x

theorem credit_card_problem :
  let initial_balance : ℚ := 50
  let interest_rate : ℚ := 1/5
  let final_balance : ℚ := 96
  amount_added initial_balance interest_rate final_balance = 20 := by
  sorry

end credit_card_problem_l66_6607
