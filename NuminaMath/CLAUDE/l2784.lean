import Mathlib

namespace det_equals_polynomial_l2784_278478

/-- The determinant of a 3x3 matrix with polynomial entries -/
def matrix_det (y : ℝ) : ℝ :=
  let a11 := 2*y + 3
  let a12 := y - 1
  let a13 := y + 2
  let a21 := y + 1
  let a22 := 2*y
  let a23 := y
  let a31 := y
  let a32 := y
  let a33 := 2*y - 1
  a11 * (a22 * a33 - a23 * a32) - 
  a12 * (a21 * a33 - a23 * a31) + 
  a13 * (a21 * a32 - a22 * a31)

theorem det_equals_polynomial (y : ℝ) : 
  matrix_det y = 4*y^3 + 8*y^2 - 2*y - 1 := by
  sorry

end det_equals_polynomial_l2784_278478


namespace first_chapter_has_13_pages_l2784_278429

/-- Represents a book with chapters of increasing length -/
structure Book where
  num_chapters : ℕ
  total_pages : ℕ
  page_increase : ℕ

/-- Calculates the number of pages in the first chapter of a book -/
def first_chapter_pages (b : Book) : ℕ :=
  let x := (b.total_pages - (b.num_chapters * (b.num_chapters - 1) * b.page_increase / 2)) / b.num_chapters
  x

/-- Theorem stating that for a specific book, the first chapter has 13 pages -/
theorem first_chapter_has_13_pages :
  let b : Book := { num_chapters := 5, total_pages := 95, page_increase := 3 }
  first_chapter_pages b = 13 := by
  sorry


end first_chapter_has_13_pages_l2784_278429


namespace malaria_parasite_length_l2784_278426

theorem malaria_parasite_length : 0.0000015 = 1.5 * 10^(-6) := by
  sorry

end malaria_parasite_length_l2784_278426


namespace max_stamps_purchasable_l2784_278422

/-- Given a stamp price of 28 cents and a budget of 3600 cents,
    the maximum number of stamps that can be purchased is 128. -/
theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) :
  stamp_price = 28 → budget = 3600 → 
  (∃ (n : ℕ), n * stamp_price ≤ budget ∧ 
    ∀ (m : ℕ), m * stamp_price ≤ budget → m ≤ n) →
  (∃ (max_stamps : ℕ), max_stamps = 128) :=
by sorry

end max_stamps_purchasable_l2784_278422


namespace cookies_per_pack_l2784_278419

/-- Given that Candy baked four trays with 24 cookies each and divided them equally into eight packs,
    prove that the number of cookies in each pack is 12. -/
theorem cookies_per_pack :
  let num_trays : ℕ := 4
  let cookies_per_tray : ℕ := 24
  let num_packs : ℕ := 8
  let total_cookies : ℕ := num_trays * cookies_per_tray
  let cookies_per_pack : ℕ := total_cookies / num_packs
  cookies_per_pack = 12 := by sorry

end cookies_per_pack_l2784_278419


namespace solution_set_f_greater_than_2_range_of_t_for_nonempty_solution_l2784_278441

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |2*x - 4|

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = Set.Ioo (5/3) 3 := by sorry

-- Theorem 2: Range of t for non-empty solution set of f(x) > t^2 + 2t
theorem range_of_t_for_nonempty_solution :
  ∀ t : ℝ, (∃ x : ℝ, f x > t^2 + 2*t) ↔ t ∈ Set.Ioo (-3) 1 := by sorry

end solution_set_f_greater_than_2_range_of_t_for_nonempty_solution_l2784_278441


namespace square_area_on_parabola_l2784_278470

/-- The area of a square with one side on y = 8 and endpoints on y = x^2 + 4x + 3 is 36 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (8 = x₁^2 + 4*x₁ + 3) ∧
  (8 = x₂^2 + 4*x₂ + 3) ∧
  ((x₂ - x₁)^2 = 36) := by
  sorry

end square_area_on_parabola_l2784_278470


namespace f_positive_iff_f_inequality_iff_l2784_278482

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the first part of the problem
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x < -1/3 ∨ x > 3 := by sorry

-- Theorem for the second part of the problem
theorem f_inequality_iff (a : ℝ) : 
  (∀ x : ℝ, f x + 3 * |x + 2| ≥ |a - 1|) ↔ -4 ≤ a ∧ a ≤ 6 := by sorry

end f_positive_iff_f_inequality_iff_l2784_278482


namespace polar_coordinates_of_point_l2784_278400

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 2 ∧ y = -2 →
  ρ = Real.sqrt (x^2 + y^2) ∧
  θ = -π/4 ∧
  x = ρ * Real.cos θ ∧
  y = ρ * Real.sin θ →
  ρ = 2 * Real.sqrt 2 ∧ θ = -π/4 := by
  sorry

end polar_coordinates_of_point_l2784_278400


namespace matrix_sum_of_squares_l2784_278496

open Matrix

theorem matrix_sum_of_squares (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (transpose B = (2 : ℝ) • (B⁻¹)) →
  x^2 + y^2 + z^2 + w^2 = 4 := by
  sorry

end matrix_sum_of_squares_l2784_278496


namespace division_fifteen_by_negative_five_l2784_278459

theorem division_fifteen_by_negative_five : (15 : ℤ) / (-5 : ℤ) = -3 := by sorry

end division_fifteen_by_negative_five_l2784_278459


namespace system_of_equations_l2784_278412

theorem system_of_equations (x y k : ℝ) 
  (eq1 : 3 * x + 4 * y = k + 2)
  (eq2 : 2 * x + y = 4)
  (eq3 : x + y = 2) : k = 4 := by
  sorry

end system_of_equations_l2784_278412


namespace square_sum_over_sum_ge_sqrt_product_l2784_278406

theorem square_sum_over_sum_ge_sqrt_product (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y^2) / (x + y) ≥ Real.sqrt (x * y) := by sorry

end square_sum_over_sum_ge_sqrt_product_l2784_278406


namespace max_value_of_a_l2784_278421

-- Define the operation
def determinant (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem max_value_of_a :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  (∃ a_max : ℝ, a ≤ a_max ∧ a_max = 3/2) :=
by sorry

end max_value_of_a_l2784_278421


namespace triangle_formation_l2784_278465

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  can_form_triangle 3 6 8 ∧
  can_form_triangle 3 8 9 ∧
  ¬(can_form_triangle 3 6 9) ∧
  can_form_triangle 6 8 9 :=
by sorry

end triangle_formation_l2784_278465


namespace min_value_theorem_l2784_278474

theorem min_value_theorem (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + y₀ + z₀ = 1 ∧ 
    1/x₀ + 4/y₀ + 9/z₀ = 36 :=
by sorry

end min_value_theorem_l2784_278474


namespace diophantine_equation_solution_l2784_278438

theorem diophantine_equation_solution :
  ∃ (a b c d : ℕ), 5 * a + 6 * b + 7 * c + 11 * d = 1999 ∧
  a = 389 ∧ b = 2 ∧ c = 1 ∧ d = 3 := by
  sorry

end diophantine_equation_solution_l2784_278438


namespace rectangle_length_reduction_l2784_278414

theorem rectangle_length_reduction (original_length original_width : ℝ) 
  (h : original_length > 0 ∧ original_width > 0) :
  let new_width := original_width * (1 + 0.4285714285714287)
  let new_length := original_length * 0.7
  original_length * original_width = new_length * new_width :=
by
  sorry

#check rectangle_length_reduction

end rectangle_length_reduction_l2784_278414


namespace motion_equation_l2784_278448

/-- Given V = gt + V₀ and S = (1/2)gt² + V₀t + kt³, 
    prove that t = (2S(V-V₀)) / (V² - V₀² + 2k(V-V₀)²) -/
theorem motion_equation (g k V V₀ S t : ℝ) 
  (hV : V = g * t + V₀)
  (hS : S = (1/2) * g * t^2 + V₀ * t + k * t^3) :
  t = (2 * S * (V - V₀)) / (V^2 - V₀^2 + 2 * k * (V - V₀)^2) :=
by sorry

end motion_equation_l2784_278448


namespace pauls_toys_l2784_278475

theorem pauls_toys (toys_per_box : ℕ) (number_of_boxes : ℕ) (h1 : toys_per_box = 8) (h2 : number_of_boxes = 4) :
  toys_per_box * number_of_boxes = 32 := by
  sorry

end pauls_toys_l2784_278475


namespace discount_rate_inequality_l2784_278411

/-- Represents the maximum discount rate that can be offered while ensuring a profit margin of at least 5% -/
def max_discount_rate (cost_price selling_price min_profit_margin : ℝ) : Prop :=
  ∃ x : ℝ, 
    0 ≤ x ∧ x ≤ 1 ∧
    selling_price * ((1 : ℝ) / 10) * x - cost_price ≥ min_profit_margin * cost_price

theorem discount_rate_inequality 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 150)
  (h3 : min_profit_margin = 0.05) :
  max_discount_rate cost_price selling_price min_profit_margin :=
sorry

end discount_rate_inequality_l2784_278411


namespace tom_lifting_capacity_l2784_278456

def initial_capacity : ℝ := 80
def training_multiplier : ℝ := 2
def specialization_increase : ℝ := 1.1
def num_hands : ℕ := 2

theorem tom_lifting_capacity : 
  initial_capacity * training_multiplier * specialization_increase * num_hands = 352 := by
  sorry

end tom_lifting_capacity_l2784_278456


namespace inequality_proof_l2784_278405

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end inequality_proof_l2784_278405


namespace sum_of_cubes_l2784_278473

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end sum_of_cubes_l2784_278473


namespace expression_evaluation_l2784_278427

theorem expression_evaluation :
  let f (x : ℝ) := (x^2 - 5*x + 6) / (x - 2)
  f 3 = 0 := by
  sorry

end expression_evaluation_l2784_278427


namespace min_points_on_circle_l2784_278467

theorem min_points_on_circle (circle_length : ℕ) (h : circle_length = 1956) :
  let min_points := 1304
  ∀ n : ℕ, n < min_points →
    ¬(∀ p : ℕ, p < n →
      (∃! q : ℕ, q < n ∧ (q - p) % circle_length = 1) ∧
      (∃! r : ℕ, r < n ∧ (r - p) % circle_length = 2)) ∧
  (∀ p : ℕ, p < min_points →
    (∃! q : ℕ, q < min_points ∧ (q - p) % circle_length = 1) ∧
    (∃! r : ℕ, r < min_points ∧ (r - p) % circle_length = 2)) :=
by sorry

end min_points_on_circle_l2784_278467


namespace reflect_h_twice_l2784_278428

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_y_eq_x_minus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let q := (p.1, p.2 + 2)  -- Translate up by 2
  let r := (q.2, q.1)      -- Reflect across y = x
  (r.1, r.2 - 2)           -- Translate down by 2

theorem reflect_h_twice (h : ℝ × ℝ) :
  h = (5, 3) →
  reflect_y_eq_x_minus_2 (reflect_x h) = (-1, 3) := by
  sorry

end reflect_h_twice_l2784_278428


namespace inverse_of_proposition_l2784_278464

theorem inverse_of_proposition (p q : Prop) :
  (¬p → ¬q) → (¬q → ¬p) := by sorry

end inverse_of_proposition_l2784_278464


namespace solution_set_f_gt_2_range_m_common_points_l2784_278451

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Define the quadratic function g
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem 1: Solution set of f(x) > 2 when m = 5
theorem solution_set_f_gt_2 :
  {x : ℝ | f 5 x > 2} = Set.Ioo (-4/3 : ℝ) 0 := by sorry

-- Theorem 2: Range of m for which f and g always have common points
theorem range_m_common_points :
  {m : ℝ | ∀ y, ∃ x, f m x = y ∧ g x = y} = Set.Ici 4 := by sorry

end solution_set_f_gt_2_range_m_common_points_l2784_278451


namespace total_animals_equals_total_humps_l2784_278481

/-- Represents the composition of a herd of animals -/
structure Herd where
  horses : ℕ
  twoHumpedCamels : ℕ
  oneHumpedCamels : ℕ

/-- Calculates the total number of humps in the herd -/
def totalHumps (h : Herd) : ℕ :=
  2 * h.twoHumpedCamels + h.oneHumpedCamels

/-- Calculates the total number of animals in the herd -/
def totalAnimals (h : Herd) : ℕ :=
  h.horses + h.twoHumpedCamels + h.oneHumpedCamels

/-- Theorem stating that the total number of animals equals the total number of humps
    under specific conditions -/
theorem total_animals_equals_total_humps (h : Herd) :
  h.horses = h.twoHumpedCamels →
  totalHumps h = 200 →
  totalAnimals h = 200 := by
  sorry

#check total_animals_equals_total_humps

end total_animals_equals_total_humps_l2784_278481


namespace yellow_pairs_l2784_278449

theorem yellow_pairs (total_students : ℕ) (blue_students : ℕ) (yellow_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 132 →
  blue_students = 57 →
  yellow_students = 75 →
  total_pairs = 66 →
  blue_blue_pairs = 23 →
  blue_students + yellow_students = total_students →
  2 * total_pairs = total_students →
  ∃ (yellow_yellow_pairs : ℕ),
    yellow_yellow_pairs = 32 ∧
    yellow_yellow_pairs + blue_blue_pairs + (total_pairs - yellow_yellow_pairs - blue_blue_pairs) = total_pairs :=
by sorry

end yellow_pairs_l2784_278449


namespace min_value_of_f_l2784_278458

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x ∈ Set.Icc (-1) 0, f x ≥ m) ∧ (∃ x ∈ Set.Icc (-1) 0, f x = m) ∧ m = 2 := by
  sorry

end min_value_of_f_l2784_278458


namespace f_not_even_l2784_278452

def f (x : ℝ) := x^2 + x

theorem f_not_even : ¬(∀ x : ℝ, f x = f (-x)) := by
  sorry

end f_not_even_l2784_278452


namespace cubic_factorization_l2784_278407

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x - 2)^2 := by
  sorry

end cubic_factorization_l2784_278407


namespace bandage_overlap_l2784_278463

theorem bandage_overlap (n : ℕ) (l : ℝ) (total : ℝ) (h1 : n = 20) (h2 : l = 15.25) (h3 : total = 248) :
  (n * l - total) / (n - 1) = 3 := by
  sorry

end bandage_overlap_l2784_278463


namespace dans_initial_money_l2784_278436

/-- 
Given that Dan has some money, buys a candy bar for $1, and has $3 left afterwards,
prove that Dan's initial amount of money was $4.
-/
theorem dans_initial_money (money_left : ℕ) (candy_cost : ℕ) : 
  money_left = 3 → candy_cost = 1 → money_left + candy_cost = 4 :=
by sorry

end dans_initial_money_l2784_278436


namespace intersection_M_complement_N_l2784_278490

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 3*x^2 + 1}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = {x | -1 ≤ x ∧ x < 1} := by
  sorry

end intersection_M_complement_N_l2784_278490


namespace fine_payment_l2784_278430

theorem fine_payment (F : ℚ) 
  (joe_payment : ℚ) (peter_payment : ℚ) (sam_payment : ℚ)
  (h1 : joe_payment = F / 4 + 7)
  (h2 : peter_payment = F / 3 - 7)
  (h3 : sam_payment = F / 2 - 12)
  (h4 : joe_payment + peter_payment + sam_payment = F) :
  sam_payment / F = 5 / 12 := by
  sorry

end fine_payment_l2784_278430


namespace cube_root_125_times_fourth_root_256_times_sqrt_16_l2784_278492

theorem cube_root_125_times_fourth_root_256_times_sqrt_16 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (16 : ℝ) ^ (1/2) = 80 := by
  sorry

end cube_root_125_times_fourth_root_256_times_sqrt_16_l2784_278492


namespace S_union_T_eq_S_l2784_278493

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 > 0}
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

theorem S_union_T_eq_S : S ∪ T = S := by sorry

end S_union_T_eq_S_l2784_278493


namespace independence_day_absentees_l2784_278413

theorem independence_day_absentees (total_children : ℕ) (bananas : ℕ) (present_children : ℕ) : 
  total_children = 740 →
  bananas = total_children * 2 →
  bananas = present_children * 4 →
  total_children - present_children = 370 := by
sorry

end independence_day_absentees_l2784_278413


namespace value_of_a_l2784_278499

theorem value_of_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 7 * a^2 + 14 * a * b = a^3 + 2 * a^2 * b) : a = 7 := by
  sorry

end value_of_a_l2784_278499


namespace max_value_quadratic_l2784_278486

theorem max_value_quadratic (x : ℝ) : 
  (∃ (z : ℝ), z = x^2 - 14*x + 10) → 
  (∃ (max_z : ℝ), max_z = -39 ∧ ∀ (y : ℝ), y = x^2 - 14*x + 10 → y ≤ max_z) :=
sorry

end max_value_quadratic_l2784_278486


namespace coin_coverage_probability_l2784_278431

theorem coin_coverage_probability (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) :
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  let coin_radius : ℝ := coin_diameter / 2
  let landing_area : ℝ := (square_side - 2 * coin_radius) ^ 2
  let triangle_area : ℝ := 4 * (triangle_leg ^ 2 / 2 + π * coin_radius ^ 2 / 4 + triangle_leg * coin_radius)
  let diamond_area : ℝ := diamond_side ^ 2 + 4 * (π * coin_radius ^ 2 / 4 + diamond_side * coin_radius / Real.sqrt 2)
  let total_black_area : ℝ := triangle_area + diamond_area
  let probability : ℝ := total_black_area / landing_area
  probability = (1 / 225) * (900 + 300 * Real.sqrt 2 + π) := by
    sorry

end coin_coverage_probability_l2784_278431


namespace sale_to_cost_ratio_l2784_278437

/-- Given an article with a cost price, sale price, and profit, prove that if the ratio of profit to cost price is 2, then the ratio of sale price to cost price is 3. -/
theorem sale_to_cost_ratio (cost_price sale_price profit : ℝ) 
  (h_positive : cost_price > 0)
  (h_profit_ratio : profit / cost_price = 2)
  (h_profit_def : profit = sale_price - cost_price) :
  sale_price / cost_price = 3 := by
  sorry

end sale_to_cost_ratio_l2784_278437


namespace house_selling_price_l2784_278469

theorem house_selling_price 
  (original_price : ℝ)
  (profit_percentage : ℝ)
  (commission_percentage : ℝ)
  (h1 : original_price = 80000)
  (h2 : profit_percentage = 20)
  (h3 : commission_percentage = 5)
  : original_price + (profit_percentage / 100) * original_price + (commission_percentage / 100) * original_price = 100000 := by
  sorry

end house_selling_price_l2784_278469


namespace emails_left_l2784_278408

def process_emails (initial : ℕ) : ℕ :=
  let after_trash := initial / 2
  let after_work := after_trash - (after_trash * 2 / 5)
  let after_personal := after_work - (after_work / 4)
  let after_misc := after_personal - (after_personal / 10)
  let after_subfolder := after_misc - (after_misc * 3 / 10)
  after_subfolder - (after_subfolder / 5)

theorem emails_left (initial : ℕ) (h : initial = 600) : process_emails initial = 69 := by
  sorry

end emails_left_l2784_278408


namespace profit_maximized_at_95_l2784_278410

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -20 * x^2 + 200 * x + 4000

/-- Theorem stating that the profit is maximized at a selling price of 95 yuan -/
theorem profit_maximized_at_95 :
  let initial_purchase_price : ℝ := 80
  let initial_selling_price : ℝ := 90
  let initial_sales_volume : ℝ := 400
  let price_increase_rate : ℝ := 1
  let sales_decrease_rate : ℝ := 20
  ∃ (max_profit : ℝ), 
    (∀ x, profit_function x ≤ max_profit) ∧ 
    (profit_function 5 = max_profit) ∧
    (initial_selling_price + 5 = 95) := by
  sorry

#check profit_maximized_at_95

end profit_maximized_at_95_l2784_278410


namespace f_at_6_l2784_278455

/-- The polynomial f(x) = 3x^6 + 12x^5 + 8x^4 - 3.5x^3 + 7.2x^2 + 5x - 13 -/
def f (x : ℝ) : ℝ := 3*x^6 + 12*x^5 + 8*x^4 - 3.5*x^3 + 7.2*x^2 + 5*x - 13

/-- Theorem stating that f(6) = 243168.2 -/
theorem f_at_6 : f 6 = 243168.2 := by
  sorry

end f_at_6_l2784_278455


namespace min_value_expression_l2784_278453

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ≥ 11 / 16 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 ∧
    (4 / (x₀ + 2) + (3 * x₀ - 7) / (3 * y₀ + 4)) = 11 / 16 :=
by sorry

end min_value_expression_l2784_278453


namespace amy_balloon_count_l2784_278462

/-- Given that James has 1222 balloons and 709 more balloons than Amy,
    prove that Amy has 513 balloons. -/
theorem amy_balloon_count :
  ∀ (james_balloons amy_balloons : ℕ),
    james_balloons = 1222 →
    james_balloons = amy_balloons + 709 →
    amy_balloons = 513 :=
by
  sorry

end amy_balloon_count_l2784_278462


namespace min_money_for_city_l2784_278433

/-- Represents the resources needed to build a city -/
structure CityResources where
  ore : ℕ
  wheat : ℕ

/-- Represents the market prices and exchange rates -/
structure MarketPrices where
  ore_price : ℕ
  wheat_bundle_price : ℕ
  wheat_bundle_size : ℕ
  wheat_to_ore_rate : ℕ

/-- The problem setup -/
def city_building_problem (work_days : ℕ) (daily_ore_production : ℕ) 
  (city_resources : CityResources) (market_prices : MarketPrices) : Prop :=
  ∃ (initial_money : ℕ),
    initial_money = 9 ∧
    work_days * daily_ore_production + 
    (market_prices.wheat_bundle_size - city_resources.wheat) = 
    city_resources.ore ∧
    initial_money + 
    (work_days * daily_ore_production - city_resources.ore) * market_prices.ore_price = 
    (market_prices.wheat_bundle_size / city_resources.wheat) * market_prices.wheat_bundle_price

/-- The theorem to be proved -/
theorem min_money_for_city : 
  city_building_problem 3 1 
    { ore := 3, wheat := 2 } 
    { ore_price := 3, 
      wheat_bundle_price := 12, 
      wheat_bundle_size := 3, 
      wheat_to_ore_rate := 1 } :=
by
  sorry


end min_money_for_city_l2784_278433


namespace cracker_problem_l2784_278485

/-- The number of crackers Darren and Calvin bought together -/
def total_crackers (darren_boxes calvin_boxes crackers_per_box : ℕ) : ℕ :=
  (darren_boxes + calvin_boxes) * crackers_per_box

theorem cracker_problem :
  ∀ (darren_boxes calvin_boxes crackers_per_box : ℕ),
    darren_boxes = 4 →
    crackers_per_box = 24 →
    calvin_boxes = 2 * darren_boxes - 1 →
    total_crackers darren_boxes calvin_boxes crackers_per_box = 264 := by
  sorry

end cracker_problem_l2784_278485


namespace relationship_abc_l2784_278435

theorem relationship_abc (a b c : ℝ) (ha : a = Real.log 3 / Real.log 0.5)
  (hb : b = Real.sqrt 2) (hc : c = Real.sqrt 0.5) : b > c ∧ c > a := by
  sorry

end relationship_abc_l2784_278435


namespace fraction_of_juniors_studying_japanese_l2784_278420

/-- Proves that the fraction of juniors studying Japanese is 3/4 given the specified conditions. -/
theorem fraction_of_juniors_studying_japanese :
  ∀ (j s : ℕ), -- j: number of juniors, s: number of seniors
  s = 2 * j → -- senior class is twice the size of junior class
  ∃ (x : ℚ), -- x: fraction of juniors studying Japanese
  (1 / 8 : ℚ) * s + x * j = (1 / 3 : ℚ) * (j + s) ∧ -- equation based on given conditions
  x = 3 / 4 := by
  sorry

end fraction_of_juniors_studying_japanese_l2784_278420


namespace school_governor_election_votes_l2784_278416

theorem school_governor_election_votes (elvis_votes : ℕ) (elvis_percentage : ℚ) 
  (h1 : elvis_votes = 45)
  (h2 : elvis_percentage = 1/4)
  (h3 : elvis_votes = elvis_percentage * total_votes) :
  total_votes = 180 :=
by
  sorry

end school_governor_election_votes_l2784_278416


namespace distance_between_foci_l2784_278497

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 29 := by
  sorry

end distance_between_foci_l2784_278497


namespace trash_can_count_l2784_278450

theorem trash_can_count (x : ℕ) 
  (h1 : (x / 2 + 8) / 2 + x = 34) : x = 24 := by
  sorry

end trash_can_count_l2784_278450


namespace sequence_inequality_l2784_278442

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := 3 * n - 2 * n^2

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℝ := S n - S (n - 1)

theorem sequence_inequality (n : ℕ) (h : n ≥ 2) :
  n * (a 1) > S n ∧ S n > n * (a n) := by sorry

end sequence_inequality_l2784_278442


namespace max_k_value_l2784_278401

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8 = 0

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for a point to be a valid center
def valid_center (k : ℝ) (x y : ℝ) : Prop :=
  line k x y ∧ ∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 = 1

-- Theorem statement
theorem max_k_value :
  (∃ k : ℝ, ∀ k' : ℝ, (∃ x y : ℝ, valid_center k' x y) → k' ≤ k) ∧
  (∃ x y : ℝ, valid_center (12/5) x y) :=
sorry

end max_k_value_l2784_278401


namespace mechanic_worked_six_hours_l2784_278498

def mechanic_hours (total_cost parts_cost labor_rate : ℚ) : ℚ :=
  let parts_total := 2 * parts_cost
  let labor_cost := total_cost - parts_total
  let minutes_worked := labor_cost / labor_rate
  minutes_worked / 60

theorem mechanic_worked_six_hours :
  mechanic_hours 220 20 0.5 = 6 := by sorry

end mechanic_worked_six_hours_l2784_278498


namespace max_value_polynomial_l2784_278425

theorem max_value_polynomial (a b : ℝ) (h : a^2 + 4*b^2 = 4) :
  ∃ M : ℝ, M = 16 ∧ ∀ x y : ℝ, x^2 + 4*y^2 = 4 → 3*x^5*y - 40*x^3*y^3 + 48*x*y^5 ≤ M :=
by sorry

end max_value_polynomial_l2784_278425


namespace trigonometric_identities_l2784_278477

theorem trigonometric_identities (x : Real) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.tan x = -2) : 
  (Real.sin x - Real.cos x = -3 * Real.sqrt 5 / 5) ∧ 
  ((Real.sin (2 * π - x) * Real.cos (π - x) - Real.sin x ^ 2) / 
   (Real.cos (π + x) * Real.cos (π/2 - x) + Real.cos x ^ 2) = -2) := by
  sorry

end trigonometric_identities_l2784_278477


namespace cosine_arctangent_equation_solution_l2784_278491

theorem cosine_arctangent_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.cos (Real.arctan (2 * x)) = x / 2 →
  ∃ (x : ℝ), x > 0 ∧ Real.cos (Real.arctan (2 * x)) = x / 2 ∧ x^2 = (Real.sqrt 17 - 1) / 4 :=
by
  sorry

end cosine_arctangent_equation_solution_l2784_278491


namespace target_hit_probability_l2784_278466

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end target_hit_probability_l2784_278466


namespace line_slope_is_one_l2784_278461

-- Define the line using its point-slope form
def line_equation (x y : ℝ) : Prop := y + 1 = x - 2

-- State the theorem
theorem line_slope_is_one :
  ∀ x y : ℝ, line_equation x y → (y - (y + 1)) / (x - (x - 2)) = 1 := by
  sorry

end line_slope_is_one_l2784_278461


namespace city_population_problem_l2784_278403

theorem city_population_problem (p : ℝ) : 
  (0.84 * (p + 2500) + 500 = p + 2680) → p = 500 := by
  sorry

end city_population_problem_l2784_278403


namespace universally_energetic_characterization_no_specific_energetic_triplets_l2784_278483

/-- A triplet (a, b, c) is n-energetic if it satisfies the given conditions --/
def isNEnergetic (a b c n : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ Nat.gcd a (Nat.gcd b c) = 1 ∧ (a^n + b^n + c^n) % (a + b + c) = 0

/-- A triplet (a, b, c) is universally energetic if it is n-energetic for all n ≥ 1 --/
def isUniversallyEnergetic (a b c : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → isNEnergetic a b c n

/-- The set of all universally energetic triplets --/
def universallyEnergeticTriplets : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ isUniversallyEnergetic t.1 t.2.1 t.2.2}

theorem universally_energetic_characterization :
    universallyEnergeticTriplets = {(1, 1, 1), (1, 1, 4)} := by sorry

theorem no_specific_energetic_triplets :
    ∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 →
      (isNEnergetic a b c 2004 ∧ isNEnergetic a b c 2005 ∧ ¬isNEnergetic a b c 2007) → False := by sorry

end universally_energetic_characterization_no_specific_energetic_triplets_l2784_278483


namespace winter_fest_attendance_l2784_278484

theorem winter_fest_attendance (total_students : ℕ) (attending_students : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1400) 
  (h2 : attending_students = 800) (h3 : girls + boys = total_students) 
  (h4 : 3 * girls / 4 + 3 * boys / 5 = attending_students) : 
  3 * girls / 4 = 600 := by
sorry

end winter_fest_attendance_l2784_278484


namespace max_value_implies_a_equals_one_l2784_278402

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a

theorem max_value_implies_a_equals_one :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≤ 1) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 1) → a = 1 :=
by sorry

end max_value_implies_a_equals_one_l2784_278402


namespace quadratic_solution_correctness_l2784_278417

/-- Solutions to the quadratic equation (p+1)x² - 2px + p - 2 = 0 --/
def quadratic_solutions (p : ℝ) : Set ℝ :=
  if p = -1 then
    {3/2}
  else if p > -2 then
    {(p + Real.sqrt (p+2)) / (p+1), (p - Real.sqrt (p+2)) / (p+1)}
  else if p = -2 then
    {2}
  else
    ∅

/-- The quadratic equation (p+1)x² - 2px + p - 2 = 0 --/
def quadratic_equation (p x : ℝ) : Prop :=
  (p+1) * x^2 - 2*p*x + p - 2 = 0

theorem quadratic_solution_correctness (p : ℝ) :
  ∀ x, x ∈ quadratic_solutions p ↔ quadratic_equation p x :=
sorry

end quadratic_solution_correctness_l2784_278417


namespace perfect_number_examples_sum_xy_is_one_k_equals_36_when_S_is_perfect_number_l2784_278476

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Theorem 1: 11 is not a perfect number and 53 is a perfect number -/
theorem perfect_number_examples :
  (¬ is_perfect_number 11) ∧ (is_perfect_number 53) := by sorry

/-- Theorem 2: Given x^2 + y^2 - 4x + 2y + 5 = 0, prove x + y = 1 -/
theorem sum_xy_is_one (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 5 = 0) :
  x + y = 1 := by sorry

/-- Definition of S -/
def S (x y k : ℝ) : ℝ := 2*x^2 + y^2 + 2*x*y + 12*x + k

/-- Theorem 3: Given S = 2x^2 + y^2 + 2xy + 12x + k, 
    prove that k = 36 when S is a perfect number -/
theorem k_equals_36_when_S_is_perfect_number (x y : ℝ) :
  (∃ a b : ℝ, S x y 36 = a^2 + b^2) → 
  (∀ k : ℝ, (∃ a b : ℝ, S x y k = a^2 + b^2) → k = 36) := by sorry

end perfect_number_examples_sum_xy_is_one_k_equals_36_when_S_is_perfect_number_l2784_278476


namespace square_root_sum_implies_product_l2784_278415

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (9 + x) + Real.sqrt (16 - x) = 8) →
  ((9 + x) * (16 - x) = 380.25) :=
by
  sorry

end square_root_sum_implies_product_l2784_278415


namespace parallel_line_condition_perpendicular_line_condition_l2784_278488

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (3, 0)

-- Define the given lines
def line1 (x y : ℝ) : Prop := 4 * x + y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 4 * x + y - 14 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 3 = 0

-- Theorem for the first condition
theorem parallel_line_condition :
  parallel_line A.1 A.2 ∧
  ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), line1 (x + k) (y - 4 * k) :=
sorry

-- Theorem for the second condition
theorem perpendicular_line_condition :
  perpendicular_line B.1 B.2 ∧
  ∀ (x y : ℝ), perpendicular_line x y ↔ ∃ (k : ℝ), line2 (x + 2 * k) (y + k) :=
sorry

end parallel_line_condition_perpendicular_line_condition_l2784_278488


namespace converse_not_true_l2784_278404

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_perp : Plane → Plane → Prop)

-- State the theorem
theorem converse_not_true :
  ∃ (b : Line) (α β : Plane),
    (subset b β ∧ perp b α) ∧ ¬(plane_perp β α) :=
sorry

end converse_not_true_l2784_278404


namespace shaded_area_is_600_l2784_278409

-- Define the vertices of the rectangle
def rectangle_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 20), (0, 20)]

-- Define the vertices of the shaded polygon
def polygon_vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (40, 10), (40, 20), (10, 20)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem shaded_area_is_600 :
  polygon_area polygon_vertices = 600 := by sorry

end shaded_area_is_600_l2784_278409


namespace quadratic_equation_coefficients_l2784_278443

/-- 
Given a quadratic equation ax² + bx + c = 0, 
this theorem proves that for the specific equation x² - 3x - 2 = 0, 
the coefficients a, b, and c are 1, -3, and -2 respectively.
-/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 3*x - 2 = 0) ∧ 
    a = 1 ∧ b = -3 ∧ c = -2 := by
  sorry

end quadratic_equation_coefficients_l2784_278443


namespace tile_formation_theorem_l2784_278460

/-- Represents a 4x4 tile --/
def Tile := Matrix (Fin 4) (Fin 4) Bool

/-- Checks if a tile has alternating colors on its outside row and column --/
def hasAlternatingOutside (t : Tile) : Prop :=
  (∀ i, t 0 i ≠ t 0 (i + 1)) ∧
  (∀ i, t i 0 ≠ t (i + 1) 0)

/-- Represents the property that a tile can be formed by combining two pieces --/
def canBeFormedByPieces (t : Tile) : Prop :=
  hasAlternatingOutside t

theorem tile_formation_theorem (t : Tile) :
  ¬(canBeFormedByPieces t) ↔ ¬(hasAlternatingOutside t) :=
sorry

end tile_formation_theorem_l2784_278460


namespace parabola_vertex_l2784_278424

/-- The vertex of a parabola is the point where it turns. For a parabola with equation
    y² + 8y + 2x + 11 = 0, this theorem states that the vertex is (5/2, -4). -/
theorem parabola_vertex (x y : ℝ) : 
  y^2 + 8*y + 2*x + 11 = 0 → (x = 5/2 ∧ y = -4) := by
  sorry

end parabola_vertex_l2784_278424


namespace riley_mistakes_l2784_278444

theorem riley_mistakes (total_questions : Nat) (team_incorrect : Nat) (ofelia_bonus : Nat) :
  total_questions = 35 →
  team_incorrect = 17 →
  ofelia_bonus = 5 →
  ∃ (riley_mistakes : Nat),
    riley_mistakes + (35 - ((35 - riley_mistakes) / 2 + ofelia_bonus)) = team_incorrect ∧
    riley_mistakes = 3 :=
by sorry

end riley_mistakes_l2784_278444


namespace school_population_l2784_278445

/-- Given a school population where:
  * The number of boys is 4 times the number of girls
  * The number of girls is 8 times the number of teachers
This theorem proves that the total number of boys, girls, and teachers
is equal to 41/32 times the number of boys. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) :
  b + g + t = (41 * b) / 32 := by
  sorry

end school_population_l2784_278445


namespace solution_set_a_2_range_of_a_l2784_278468

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Theorem 1: Solution set for a = 2
theorem solution_set_a_2 : 
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2: Range of a for the inequality to hold
theorem range_of_a : 
  {a : ℝ | ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4} = {-1, 1} := by sorry

end solution_set_a_2_range_of_a_l2784_278468


namespace unique_a_satisfies_condition_l2784_278423

/-- Converts a base-25 number to its decimal representation modulo 12 -/
def base25ToDecimalMod12 (digits : List Nat) : Nat :=
  (digits.reverse.enum.map (fun (i, d) => d * (25^i % 12)) |>.sum) % 12

/-- The given number in base 25 -/
def number : List Nat := [3, 1, 4, 2, 6, 5, 2, 3]

theorem unique_a_satisfies_condition :
  ∃! a : ℕ, 0 ≤ a ∧ a ≤ 14 ∧ (base25ToDecimalMod12 number - a) % 12 = 0 ∧ a = 2 := by
  sorry

end unique_a_satisfies_condition_l2784_278423


namespace log_216_equals_3log2_plus_3log3_l2784_278439

theorem log_216_equals_3log2_plus_3log3 : Real.log 216 = 3 * Real.log 2 + 3 * Real.log 3 := by
  sorry

end log_216_equals_3log2_plus_3log3_l2784_278439


namespace orange_put_back_l2784_278480

theorem orange_put_back (apple_price orange_price : ℚ)
  (total_fruit : ℕ) (initial_avg_price final_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruit = 10)
  (h4 : initial_avg_price = 54/100)
  (h5 : final_avg_price = 45/100) :
  ∃ (oranges_to_put_back : ℕ),
    oranges_to_put_back = 6 ∧
    ∃ (initial_apples initial_oranges : ℕ),
      initial_apples + initial_oranges = total_fruit ∧
      (initial_apples * apple_price + initial_oranges * orange_price) / total_fruit = initial_avg_price ∧
      ∃ (final_oranges : ℕ),
        final_oranges = initial_oranges - oranges_to_put_back ∧
        (initial_apples * apple_price + final_oranges * orange_price) / (initial_apples + final_oranges) = final_avg_price :=
by
  sorry

end orange_put_back_l2784_278480


namespace lottery_winning_probability_l2784_278479

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallsDrawn : ℕ := 6

def winningProbability : ℚ :=
  1 / (megaBallCount * (winnerBallCount.choose winnerBallsDrawn))

theorem lottery_winning_probability :
  winningProbability = 1 / 476721000 := by sorry

end lottery_winning_probability_l2784_278479


namespace power_values_l2784_278471

theorem power_values (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(4*m + 3*n) = 432 ∧ a^(5*m - 2*n) = 32/9 := by
  sorry

end power_values_l2784_278471


namespace translation_result_l2784_278457

def initial_point : ℝ × ℝ := (-4, 3)
def translation : ℝ × ℝ := (-2, -2)

def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + t.1, p.2 + t.2)

theorem translation_result :
  translate initial_point translation = (-6, 1) := by sorry

end translation_result_l2784_278457


namespace unique_x_satisfying_three_inequalities_l2784_278472

theorem unique_x_satisfying_three_inequalities :
  ∃! (x : ℕ), (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37) ∧
              ¬(2 * x ≥ 21) ∧ ¬(x > 7) :=
by sorry

end unique_x_satisfying_three_inequalities_l2784_278472


namespace logarithm_inequality_l2784_278489

theorem logarithm_inequality (m n p : ℝ) 
  (hm : 0 < m ∧ m < 1) 
  (hn : 0 < n ∧ n < 1) 
  (hp : 0 < p ∧ p < 1) 
  (h_log : Real.log m / Real.log 3 = Real.log n / Real.log 5 ∧ 
           Real.log n / Real.log 5 = Real.log p / Real.log 10) : 
  m^(1/3) < n^(1/5) ∧ n^(1/5) < p^(1/10) := by
sorry

end logarithm_inequality_l2784_278489


namespace petyas_torn_sheets_l2784_278440

/-- Represents a book with consecutively numbered pages -/
structure Book where
  firstTornPage : ℕ
  lastTornPage : ℕ

/-- Checks if two numbers have the same digits -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Calculates the number of sheets torn out from a book -/
def sheetsTornOut (book : Book) : ℕ := sorry

/-- Theorem stating the number of sheets torn out by Petya -/
theorem petyas_torn_sheets (book : Book) : 
  book.firstTornPage = 185 ∧ 
  sameDigits book.firstTornPage book.lastTornPage ∧
  book.lastTornPage > book.firstTornPage ∧
  Even book.lastTornPage →
  sheetsTornOut book = 167 := by
  sorry

end petyas_torn_sheets_l2784_278440


namespace max_perimeter_after_cut_l2784_278487

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: Maximum perimeter after cutting out a smaller rectangle -/
theorem max_perimeter_after_cut (original : Rectangle) (cutout : Rectangle) :
  original.length = 20 ∧ 
  original.width = 16 ∧ 
  cutout.length = 10 ∧ 
  cutout.width = 5 →
  ∃ (remaining : Rectangle), 
    perimeter remaining = 92 ∧ 
    ∀ (other : Rectangle), perimeter other ≤ perimeter remaining :=
by sorry

end max_perimeter_after_cut_l2784_278487


namespace initial_amount_proof_l2784_278447

theorem initial_amount_proof (remaining_amount : ℝ) (spent_percentage : ℝ) (initial_amount : ℝ) : 
  remaining_amount = 3500 ∧ 
  spent_percentage = 30 ∧ 
  remaining_amount = initial_amount * (1 - spent_percentage / 100) → 
  initial_amount = 5000 := by
sorry

end initial_amount_proof_l2784_278447


namespace investment_revenue_difference_l2784_278454

def banks_investments : ℕ := 8
def banks_revenue_per_investment : ℕ := 500
def elizabeth_investments : ℕ := 5
def elizabeth_revenue_per_investment : ℕ := 900

theorem investment_revenue_difference :
  elizabeth_investments * elizabeth_revenue_per_investment - 
  banks_investments * banks_revenue_per_investment = 500 := by
sorry

end investment_revenue_difference_l2784_278454


namespace tan_alpha_equals_two_tan_pi_fifth_l2784_278418

theorem tan_alpha_equals_two_tan_pi_fifth (α : Real) 
  (h : Real.tan α = 2 * Real.tan (π / 5)) : 
  (Real.cos (α - 3 * π / 10)) / (Real.sin (α - π / 5)) = 3 := by
  sorry

end tan_alpha_equals_two_tan_pi_fifth_l2784_278418


namespace intersection_perpendicular_implies_k_l2784_278494

/-- The line l: kx - y - 2 = 0 intersects the circle O: x^2 + y^2 = 4 at points A and B. -/
def intersects (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  k * A.1 - A.2 - 2 = 0 ∧
  k * B.1 - B.2 - 2 = 0 ∧
  A.1^2 + A.2^2 = 4 ∧
  B.1^2 + B.2^2 = 4

/-- The dot product of OA and OB is zero. -/
def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Main theorem: If the line intersects the circle and the intersection points are perpendicular from the origin, then k = ±1. -/
theorem intersection_perpendicular_implies_k (k : ℝ) (A B : ℝ × ℝ) 
  (h_intersects : intersects k A B) (h_perp : perpendicular A B) : k = 1 ∨ k = -1 := by
  sorry

end intersection_perpendicular_implies_k_l2784_278494


namespace div_negative_powers_l2784_278434

theorem div_negative_powers (a : ℝ) (h : a ≠ 0) : -28 * a^3 / (7 * a) = -4 * a^2 := by
  sorry

end div_negative_powers_l2784_278434


namespace paving_rate_calculation_l2784_278446

/-- Given a rectangular room with specified dimensions and total paving cost,
    calculate the rate per square meter for paving the floor. -/
theorem paving_rate_calculation (length width total_cost : ℝ) 
    (h1 : length = 9)
    (h2 : width = 4.75)
    (h3 : total_cost = 38475) : 
  total_cost / (length * width) = 900 := by
  sorry

end paving_rate_calculation_l2784_278446


namespace robert_ate_seven_chocolates_l2784_278495

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference between Robert's and Nickel's chocolate consumption -/
def robert_nickel_difference : ℕ := 2

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := nickel_chocolates + robert_nickel_difference

theorem robert_ate_seven_chocolates : robert_chocolates = 7 := by
  sorry

end robert_ate_seven_chocolates_l2784_278495


namespace circle_max_distance_l2784_278432

/-- Given a circle with equation x^2 + y^2 + 4x - 2y - 4 = 0, 
    the maximum value of x^2 + y^2 is 14 + 6√5 -/
theorem circle_max_distance (x y : ℝ) : 
  x^2 + y^2 + 4*x - 2*y - 4 = 0 → 
  x^2 + y^2 ≤ 14 + 6 * Real.sqrt 5 := by
sorry

end circle_max_distance_l2784_278432
