import Mathlib

namespace smallest_value_of_sum_of_cubes_l4074_407411

theorem smallest_value_of_sum_of_cubes (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2) 
  (h2 : Complex.abs (u^2 + v^2) = 8) : 
  Complex.abs (u^3 + v^3) = 20 := by
sorry

end smallest_value_of_sum_of_cubes_l4074_407411


namespace cara_seating_arrangements_l4074_407422

def number_of_friends : ℕ := 7

/-- The number of ways to choose 2 people from n friends to sit next to Cara in a circular arrangement -/
def circular_seating_arrangements (n : ℕ) : ℕ := Nat.choose n 2

theorem cara_seating_arrangements :
  circular_seating_arrangements number_of_friends = 21 := by
  sorry

end cara_seating_arrangements_l4074_407422


namespace quadratic_max_sum_roots_l4074_407423

theorem quadratic_max_sum_roots (m : ℝ) :
  let f := fun x : ℝ => 2 * x^2 - 5 * x + m
  let Δ := 25 - 8 * m  -- discriminant
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0) →  -- real roots exist
  (∀ k : ℝ, (∃ y₁ y₂ : ℝ, f y₁ = 0 ∧ f y₂ = 0) → y₁ + y₂ ≤ 5/2) ∧  -- 5/2 is max sum
  (m = 25/8 → ∃ z₁ z₂ : ℝ, f z₁ = 0 ∧ f z₂ = 0 ∧ z₁ + z₂ = 5/2)  -- max sum occurs at m = 25/8
  :=
sorry

end quadratic_max_sum_roots_l4074_407423


namespace root_line_discriminant_intersection_l4074_407416

/-- The discriminant curve in the pq-plane -/
def discriminant_curve (p q : ℝ) : Prop := 4 * p^3 + 27 * q^2 = 0

/-- The root line for a given value of a -/
def root_line (a p q : ℝ) : Prop := a * p + q + a^3 = 0

/-- The intersection points of the root line and the discriminant curve -/
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {(p, q) | discriminant_curve p q ∧ root_line a p q}

theorem root_line_discriminant_intersection (a : ℝ) :
  (a ≠ 0 → intersection_points a = {(-3 * a^2, 2 * a^3), (-3 * a^2 / 4, -a^3 / 4)}) ∧
  (a = 0 → intersection_points a = {(0, 0)}) := by
  sorry

end root_line_discriminant_intersection_l4074_407416


namespace total_volume_of_boxes_l4074_407469

-- Define the number of boxes
def num_boxes : ℕ := 4

-- Define the edge length of each box in feet
def edge_length : ℝ := 6

-- Define the volume of a single box
def single_box_volume : ℝ := edge_length ^ 3

-- Theorem stating the total volume of all boxes
theorem total_volume_of_boxes : single_box_volume * num_boxes = 864 := by
  sorry

end total_volume_of_boxes_l4074_407469


namespace sum_and_equal_numbers_l4074_407452

theorem sum_and_equal_numbers (a b c : ℚ) : 
  a + b + c = 150 →
  a + 10 = b - 3 ∧ b - 3 = 4 * c →
  b = 655 / 9 := by
sorry

end sum_and_equal_numbers_l4074_407452


namespace necessary_not_sufficient_l4074_407434

/-- A curve is an ellipse if both coefficients are positive and not equal -/
def is_ellipse (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

/-- The condition that m is between 3 and 7 -/
def m_between_3_and_7 (m : ℝ) : Prop := 3 < m ∧ m < 7

/-- The curve equation in terms of m -/
def curve_equation (m : ℝ) : Prop := is_ellipse (7 - m) (m - 3)

theorem necessary_not_sufficient :
  (∀ m : ℝ, curve_equation m → m_between_3_and_7 m) ∧
  (∃ m : ℝ, m_between_3_and_7 m ∧ ¬curve_equation m) :=
sorry

end necessary_not_sufficient_l4074_407434


namespace complex_number_modulus_l4074_407415

theorem complex_number_modulus (z : ℂ) : z = 1 / (Complex.I - 1) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_number_modulus_l4074_407415


namespace compound_inequality_solution_l4074_407428

theorem compound_inequality_solution (x : ℝ) :
  (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 6) ↔ 2 < x ∧ x < 3 := by
  sorry

end compound_inequality_solution_l4074_407428


namespace yoongi_age_proof_l4074_407405

/-- Yoongi's age -/
def yoongi_age : ℕ := 8

/-- Hoseok's age -/
def hoseok_age : ℕ := yoongi_age + 2

/-- The sum of Yoongi's and Hoseok's ages -/
def total_age : ℕ := yoongi_age + hoseok_age

theorem yoongi_age_proof : yoongi_age = 8 :=
  by
    have h1 : hoseok_age = yoongi_age + 2 := rfl
    have h2 : total_age = 18 := rfl
    sorry

end yoongi_age_proof_l4074_407405


namespace inscribed_square_area_l4074_407478

/-- The parabola function --/
def f (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A point is on the parabola if its y-coordinate equals f(x) --/
def on_parabola (p : ℝ × ℝ) : Prop := p.2 = f p.1

/-- A point is on the x-axis if its y-coordinate is 0 --/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A square is defined by its center and side length --/
structure Square where
  center : ℝ × ℝ
  side : ℝ

/-- A square is inscribed if all its vertices are either on the parabola or on the x-axis --/
def inscribed (s : Square) : Prop :=
  let half_side := s.side / 2
  let left := s.center.1 - half_side
  let right := s.center.1 + half_side
  let top := s.center.2 + half_side
  let bottom := s.center.2 - half_side
  on_x_axis (left, bottom) ∧
  on_x_axis (right, bottom) ∧
  on_parabola (left, top) ∧
  on_parabola (right, top)

/-- The theorem to be proved --/
theorem inscribed_square_area :
  ∃ (s : Square), inscribed s ∧ s.side^2 = 24 - 8 * Real.sqrt 5 := by
  sorry

end inscribed_square_area_l4074_407478


namespace problem_1_l4074_407455

theorem problem_1 : -3 * (1/4) - (-1/9) + (-3/4) + 1 * (8/9) = -2 := by sorry

end problem_1_l4074_407455


namespace triangle_problem_l4074_407453

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 4 * Real.sqrt 3 →
  b = 6 →
  Real.cos A = -1/3 →
  (c = 2 ∧ Real.cos (2 * B - π/4) = (4 - Real.sqrt 2) / 6) := by
  sorry

end triangle_problem_l4074_407453


namespace soybean_oil_production_l4074_407483

/-- Represents the conversion rates and prices for soybeans, tofu, and soybean oil -/
structure SoybeanProduction where
  soybean_to_tofu : ℝ        -- kg of tofu per kg of soybeans
  soybean_to_oil : ℝ         -- kg of soybeans needed for 1 kg of oil
  tofu_price : ℝ             -- yuan per kg of tofu
  oil_price : ℝ              -- yuan per kg of oil

/-- Represents the batch of soybeans and its processing -/
structure SoybeanBatch where
  total_soybeans : ℝ         -- total kg of soybeans in the batch
  tofu_soybeans : ℝ          -- kg of soybeans used for tofu
  oil_soybeans : ℝ           -- kg of soybeans used for oil
  total_revenue : ℝ          -- total revenue in yuan

/-- Theorem stating that given the conditions, 360 kg of soybeans were used for oil production -/
theorem soybean_oil_production (prod : SoybeanProduction) (batch : SoybeanBatch) :
  prod.soybean_to_tofu = 3 ∧
  prod.soybean_to_oil = 6 ∧
  prod.tofu_price = 3 ∧
  prod.oil_price = 15 ∧
  batch.total_soybeans = 460 ∧
  batch.total_revenue = 1800 ∧
  batch.tofu_soybeans + batch.oil_soybeans = batch.total_soybeans ∧
  batch.total_revenue = (batch.tofu_soybeans * prod.soybean_to_tofu * prod.tofu_price) +
                        (batch.oil_soybeans / prod.soybean_to_oil * prod.oil_price) →
  batch.oil_soybeans = 360 := by
  sorry

end soybean_oil_production_l4074_407483


namespace remaining_ribbon_l4074_407495

/-- Calculates the remaining ribbon length after wrapping gifts -/
theorem remaining_ribbon (num_gifts : ℕ) (ribbon_per_gift : ℚ) (total_ribbon : ℚ) :
  num_gifts = 8 →
  ribbon_per_gift = 3/2 →
  total_ribbon = 15 →
  total_ribbon - (num_gifts : ℚ) * ribbon_per_gift = 3 := by
  sorry

end remaining_ribbon_l4074_407495


namespace y₁_less_than_y₂_l4074_407474

/-- A linear function y = 2x + 1 passing through points (-3, y₁) and (4, y₂) -/
def linear_function (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the y-coordinate when x = -3 -/
def y₁ : ℝ := linear_function (-3)

/-- y₂ is the y-coordinate when x = 4 -/
def y₂ : ℝ := linear_function 4

/-- Theorem stating that y₁ < y₂ -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by sorry

end y₁_less_than_y₂_l4074_407474


namespace police_officer_arrangements_l4074_407463

def num_officers : ℕ := 5
def num_intersections : ℕ := 3

def valid_distribution (d : List ℕ) : Prop :=
  d.length = num_intersections ∧
  d.sum = num_officers ∧
  ∀ x ∈ d, 1 ≤ x ∧ x ≤ 3

def arrangements (d : List ℕ) : ℕ := sorry

def arrangements_with_AB_separate : ℕ := sorry

theorem police_officer_arrangements :
  arrangements_with_AB_separate = 114 := by sorry

end police_officer_arrangements_l4074_407463


namespace inradius_properties_l4074_407492

/-- Properties of a triangle ABC --/
structure Triangle where
  /-- Inradius of the triangle --/
  r : ℝ
  /-- Circumradius of the triangle --/
  R : ℝ
  /-- Semiperimeter of the triangle --/
  s : ℝ
  /-- Angle A of the triangle --/
  A : ℝ
  /-- Angle B of the triangle --/
  B : ℝ
  /-- Angle C of the triangle --/
  C : ℝ
  /-- Exradius opposite to angle A --/
  r_a : ℝ
  /-- Exradius opposite to angle B --/
  r_b : ℝ
  /-- Exradius opposite to angle C --/
  r_c : ℝ

/-- Theorem: Properties of inradius in a triangle --/
theorem inradius_properties (t : Triangle) :
  (t.r = 4 * t.R * Real.sin (t.A / 2) * Real.sin (t.B / 2) * Real.sin (t.C / 2)) ∧
  (t.r = t.s * Real.tan (t.A / 2) * Real.tan (t.B / 2) * Real.tan (t.C / 2)) ∧
  (t.r = t.R * (Real.cos t.A + Real.cos t.B + Real.cos t.C - 1)) ∧
  (t.r = t.r_a + t.r_b + t.r_c - 4 * t.R) := by
  sorry

end inradius_properties_l4074_407492


namespace projection_matrix_values_l4074_407458

def projection_matrix (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 10/21; c, 35/63]

theorem projection_matrix_values :
  ∀ a c : ℚ, projection_matrix a c ^ 2 = projection_matrix a c → a = 2/9 ∧ c = 7/6 :=
by
  sorry

end projection_matrix_values_l4074_407458


namespace expression_two_values_l4074_407444

theorem expression_two_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (x y : ℝ), x ≠ y ∧
    ∀ (z : ℝ), z = a / abs a + b / abs b + (a * b) / abs (a * b) → z = x ∨ z = y :=
by sorry

end expression_two_values_l4074_407444


namespace subtraction_of_decimals_l4074_407412

theorem subtraction_of_decimals : 3.57 - 1.45 = 2.12 := by sorry

end subtraction_of_decimals_l4074_407412


namespace jake_sister_weight_ratio_l4074_407440

theorem jake_sister_weight_ratio (J S : ℝ) (hJ : J > 0) (hS : S > 0) 
  (h1 : J + S = 132) (h2 : J - 15 = 2 * S) : (J - 15) / S = 2 := by
  sorry

end jake_sister_weight_ratio_l4074_407440


namespace units_digit_not_four_l4074_407450

/-- The set of numbers from which a and b are chosen -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

/-- The units digit of a number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem units_digit_not_four (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) :
  unitsDigit (2^a + 5^b) ≠ 4 := by sorry

end units_digit_not_four_l4074_407450


namespace special_three_digit_numbers_l4074_407473

-- Define a three-digit number
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the sum of digits for a three-digit number
def sum_of_digits (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

-- Define the condition for the special property
def has_special_property (n : ℕ) : Prop :=
  n = sum_of_digits n + 2 * (sum_of_digits n)^2

-- Theorem statement
theorem special_three_digit_numbers : 
  ∀ n : ℕ, is_three_digit n ∧ has_special_property n ↔ n = 171 ∨ n = 465 ∨ n = 666 := by
  sorry

end special_three_digit_numbers_l4074_407473


namespace tobias_apps_downloaded_l4074_407462

/-- The number of apps downloaded by Tobias -/
def m : ℕ := 24

/-- The base cost of each app in cents -/
def base_cost : ℕ := 200

/-- The tax rate as a percentage -/
def tax_rate : ℕ := 10

/-- The total amount spent in cents -/
def total_spent : ℕ := 5280

/-- Theorem stating that m is the correct number of apps downloaded -/
theorem tobias_apps_downloaded :
  m * (base_cost + base_cost * tax_rate / 100) = total_spent :=
by sorry

end tobias_apps_downloaded_l4074_407462


namespace function_properties_l4074_407477

noncomputable def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + 1

theorem function_properties (a b : ℝ) :
  (f a b 3 = -26) ∧ 
  (3*(3^2) - 2*a*3 + b = 0) →
  (a = 3 ∧ b = -9) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f 3 (-9) x ≤ 6) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 4, f 3 (-9) x = 6) :=
by sorry

end function_properties_l4074_407477


namespace factorization_equalities_l4074_407433

theorem factorization_equalities (a b x y : ℝ) : 
  (3 * a * x^2 + 6 * a * x * y + 3 * a * y^2 = 3 * a * (x + y)^2) ∧
  (a^2 * (x - y) - b^2 * (x - y) = (x - y) * (a + b) * (a - b)) ∧
  (a^4 + 3 * a^2 - 4 = (a + 1) * (a - 1) * (a^2 + 4)) ∧
  (4 * x^2 - y^2 - 2 * y - 1 = (2 * x + y + 1) * (2 * x - y - 1)) :=
by sorry

end factorization_equalities_l4074_407433


namespace division_of_powers_l4074_407464

theorem division_of_powers (x : ℝ) : 2 * x^5 / ((-x)^3) = -2 * x^2 := by
  sorry

end division_of_powers_l4074_407464


namespace parallel_postulate_l4074_407476

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Determines if a point is on a line --/
def Point.isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Two lines are parallel if they have the same slope --/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- The statement of the parallel postulate --/
theorem parallel_postulate (L : Line) (P : Point) 
  (h : ¬ P.isOnLine L) : 
  ∃! (M : Line), M.isParallel L ∧ P.isOnLine M :=
sorry


end parallel_postulate_l4074_407476


namespace exponent_rule_product_power_l4074_407487

theorem exponent_rule_product_power (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end exponent_rule_product_power_l4074_407487


namespace march_greatest_drop_l4074_407449

/-- Represents the months of the year --/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August

/-- Represents the price change for each month --/
def price_change : Month → ℝ
  | Month.January  => -0.75
  | Month.February => 1.50
  | Month.March    => -3.00
  | Month.April    => 2.50
  | Month.May      => -0.25
  | Month.June     => 0.80
  | Month.July     => -2.75
  | Month.August   => -1.20

/-- Determines if a given month has the greatest price drop --/
def has_greatest_drop (m : Month) : Prop :=
  ∀ other : Month, price_change m ≤ price_change other

/-- Theorem stating that March has the greatest price drop --/
theorem march_greatest_drop : has_greatest_drop Month.March :=
sorry

end march_greatest_drop_l4074_407449


namespace well_diameter_proof_l4074_407447

/-- The volume of a circular well -/
def well_volume : ℝ := 175.92918860102841

/-- The depth of the circular well -/
def well_depth : ℝ := 14

/-- The diameter of the circular well -/
def well_diameter : ℝ := 4

theorem well_diameter_proof :
  well_diameter = 4 :=
by
  sorry

end well_diameter_proof_l4074_407447


namespace calvin_mistake_l4074_407491

theorem calvin_mistake (a : ℝ) : 37 + 31 * a = 37 * 31 + a ↔ a = 37 :=
  sorry

end calvin_mistake_l4074_407491


namespace expand_expression_l4074_407457

theorem expand_expression (x : ℝ) : (2*x - 3) * (4*x + 9) = 8*x^2 + 6*x - 27 := by
  sorry

end expand_expression_l4074_407457


namespace comic_reconstruction_l4074_407479

theorem comic_reconstruction (pages_per_comic : ℕ) (torn_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 45 →
  torn_pages = 2700 →
  untorn_comics = 15 →
  (torn_pages / pages_per_comic + untorn_comics : ℕ) = 75 :=
by sorry

end comic_reconstruction_l4074_407479


namespace card_count_theorem_l4074_407472

/-- Represents the number of baseball cards each person has -/
structure CardCounts where
  brandon : ℕ
  malcom : ℕ
  ella : ℕ
  lily : ℕ
  mark : ℕ

/-- Calculates the final card counts after all transactions -/
def finalCardCounts (initial : CardCounts) : CardCounts :=
  let malcomToMark := (initial.malcom * 3) / 5
  let ellaToLily := initial.ella / 4
  { brandon := initial.brandon
  , malcom := initial.malcom - malcomToMark
  , ella := initial.ella - ellaToLily
  , lily := initial.lily + ellaToLily
  , mark := malcomToMark + 6 }

/-- Theorem stating the correctness of the final card counts -/
theorem card_count_theorem (initial : CardCounts) :
  initial.brandon = 20 →
  initial.malcom = initial.brandon + 12 →
  initial.ella = initial.malcom - 5 →
  initial.lily = 2 * initial.ella →
  initial.mark = 0 →
  let final := finalCardCounts initial
  final.brandon = 20 ∧
  final.malcom = 13 ∧
  final.ella = 21 ∧
  final.lily = 60 ∧
  final.mark = 25 :=
by
  sorry


end card_count_theorem_l4074_407472


namespace one_eighth_of_2_40_l4074_407468

theorem one_eighth_of_2_40 (x : ℕ) : (1 / 8 : ℝ) * 2^40 = 2^x → x = 37 := by
  sorry

end one_eighth_of_2_40_l4074_407468


namespace associate_professor_pencils_l4074_407420

theorem associate_professor_pencils 
  (total_people : ℕ) 
  (total_pencils : ℕ) 
  (total_charts : ℕ) 
  (associate_pencils : ℕ) 
  (h1 : total_people = 6) 
  (h2 : total_pencils = 10) 
  (h3 : total_charts = 8) :
  let associate_count := total_people - (total_charts - total_people) / 2
  associate_pencils = (total_pencils - (total_people - associate_count)) / associate_count →
  associate_pencils = 2 := by
sorry

end associate_professor_pencils_l4074_407420


namespace january_employee_count_l4074_407448

/-- The number of employees in January, given the December count and percentage increase --/
def january_employees (december_count : ℕ) (percent_increase : ℚ) : ℚ :=
  (december_count : ℚ) / (1 + percent_increase)

/-- Theorem stating that given the conditions, the number of employees in January is approximately 408.7 --/
theorem january_employee_count :
  let december_count : ℕ := 470
  let percent_increase : ℚ := 15 / 100
  let january_count := january_employees december_count percent_increase
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.05 ∧ |january_count - 408.7| < ε :=
sorry

end january_employee_count_l4074_407448


namespace triangle_formation_l4074_407446

/-- Triangle Inequality Theorem: The sum of any two sides of a triangle must be greater than the third side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if a set of three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 ∧
  can_form_triangle 13 12 20 :=
sorry

end triangle_formation_l4074_407446


namespace grid_division_equal_areas_l4074_407488

-- Define the grid
def grid_size : ℕ := 6

-- Define point P
def P : ℚ × ℚ := (3, 3)

-- Define points J and T
def J : ℚ × ℚ := (0, 4)
def T : ℚ × ℚ := (6, 4)

-- Function to calculate area of a triangle
def triangle_area (a b c : ℚ × ℚ) : ℚ :=
  let (x1, y1) := a
  let (x2, y2) := b
  let (x3, y3) := c
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) : ℚ)

-- Theorem statement
theorem grid_division_equal_areas :
  let area1 := triangle_area P J (0, 0)
  let area2 := triangle_area P T (grid_size, 0)
  let area3 := (grid_size * grid_size : ℚ) - area1 - area2
  area1 = area2 ∧ area2 = area3 := by sorry

end grid_division_equal_areas_l4074_407488


namespace twenty_five_percent_of_2004_l4074_407454

theorem twenty_five_percent_of_2004 : (25 : ℚ) / 100 * 2004 = 501 := by
  sorry

end twenty_five_percent_of_2004_l4074_407454


namespace complex_pure_imaginary_l4074_407442

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (b : ℝ), (a + 3 * Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) →
  a = -6 := by
sorry

end complex_pure_imaginary_l4074_407442


namespace absolute_value_equation_l4074_407467

theorem absolute_value_equation (x : ℝ) : 
  |3990 * x + 1995| = 1995 → x = 0 ∨ x = -1 := by sorry

end absolute_value_equation_l4074_407467


namespace total_crayons_is_18_l4074_407461

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 3

/-- The number of children -/
def number_of_children : ℕ := 6

/-- The total number of crayons -/
def total_crayons : ℕ := crayons_per_child * number_of_children

theorem total_crayons_is_18 : total_crayons = 18 := by
  sorry

end total_crayons_is_18_l4074_407461


namespace equal_area_equal_intersection_l4074_407496

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A placement of a rectangle in the plane -/
structure PlacedRectangle where
  rect : Rectangle
  center : Point
  angle : ℝ  -- Rotation angle in radians

/-- A horizontal line in the plane -/
structure HorizontalLine where
  y : ℝ

/-- The intersection of a horizontal line with a placed rectangle -/
def intersection (line : HorizontalLine) (pr : PlacedRectangle) : Option ℝ :=
  sorry

theorem equal_area_equal_intersection 
  (r1 r2 : Rectangle) 
  (h : r1.area = r2.area) :
  ∃ (pr1 pr2 : PlacedRectangle),
    pr1.rect = r1 ∧ pr2.rect = r2 ∧
    ∀ (line : HorizontalLine),
      (intersection line pr1).isSome ∨ (intersection line pr2).isSome →
      (intersection line pr1).isSome ∧ (intersection line pr2).isSome ∧
      (intersection line pr1 = intersection line pr2) :=
by sorry

end equal_area_equal_intersection_l4074_407496


namespace min_value_theorem_l4074_407417

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  a + 2*b + 3*c ≥ 18 ∧ (a + 2*b + 3*c = 18 ↔ a = 3 ∧ b = 3 ∧ c = 3) :=
by sorry

end min_value_theorem_l4074_407417


namespace inequality_proof_l4074_407441

theorem inequality_proof (a b c x y z : Real) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a^x = b*c) (eq2 : b^y = c*a) (eq3 : c^z = a*b) :
  (1 / (2 + x)) + (1 / (2 + y)) + (1 / (2 + z)) ≤ 3/4 := by
  sorry

end inequality_proof_l4074_407441


namespace train_length_l4074_407436

/-- Given a train with speed 108 km/hr passing a tree in 9 seconds, its length is 270 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 108 → time = 9 → length = speed * (1000 / 3600) * time → length = 270 := by sorry

end train_length_l4074_407436


namespace quadratic_solution_difference_squared_l4074_407456

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (6 * p^2 - 7 * p - 20 = 0) →
  (6 * q^2 - 7 * q - 20 = 0) →
  p ≠ q →
  (p - q)^2 = 529 / 36 := by
sorry

end quadratic_solution_difference_squared_l4074_407456


namespace billy_sleep_theorem_l4074_407485

def night1_sleep : ℕ := 6

def night2_sleep (n1 : ℕ) : ℕ := n1 + 2

def night3_sleep (n2 : ℕ) : ℕ := n2 / 2

def night4_sleep (n3 : ℕ) : ℕ := n3 * 3

def total_sleep (n1 n2 n3 n4 : ℕ) : ℕ := n1 + n2 + n3 + n4

theorem billy_sleep_theorem :
  let n1 := night1_sleep
  let n2 := night2_sleep n1
  let n3 := night3_sleep n2
  let n4 := night4_sleep n3
  total_sleep n1 n2 n3 n4 = 30 := by sorry

end billy_sleep_theorem_l4074_407485


namespace age_difference_daughter_daughterInLaw_l4074_407482

/-- Represents the ages of family members 5 years ago -/
structure FamilyAges5YearsAgo where
  member1 : ℕ
  member2 : ℕ
  member3 : ℕ
  daughter : ℕ

/-- Represents the current ages of family members -/
structure CurrentFamilyAges where
  member1 : ℕ
  member2 : ℕ
  member3 : ℕ
  daughterInLaw : ℕ

/-- The main theorem stating the difference in ages between daughter and daughter-in-law -/
theorem age_difference_daughter_daughterInLaw 
  (ages5YearsAgo : FamilyAges5YearsAgo)
  (currentAges : CurrentFamilyAges)
  (h1 : ages5YearsAgo.member1 + ages5YearsAgo.member2 + ages5YearsAgo.member3 + ages5YearsAgo.daughter = 114)
  (h2 : currentAges.member1 + currentAges.member2 + currentAges.member3 + currentAges.daughterInLaw = 85)
  (h3 : currentAges.member1 = ages5YearsAgo.member1 + 5)
  (h4 : currentAges.member2 = ages5YearsAgo.member2 + 5)
  (h5 : currentAges.member3 = ages5YearsAgo.member3 + 5) :
  ages5YearsAgo.daughter - currentAges.daughterInLaw = 29 := by
  sorry

end age_difference_daughter_daughterInLaw_l4074_407482


namespace xy_difference_l4074_407410

theorem xy_difference (x y : ℝ) (h : 10 * x^2 - 16 * x * y + 8 * y^2 + 6 * x - 4 * y + 1 = 0) :
  x - y = -1/4 := by
  sorry

end xy_difference_l4074_407410


namespace sin_cos_225_degrees_l4074_407432

theorem sin_cos_225_degrees : 
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 ∧ 
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_cos_225_degrees_l4074_407432


namespace expression_simplification_l4074_407498

theorem expression_simplification (α : ℝ) :
  4.59 * (Real.cos (2*α) - Real.cos (6*α) + Real.cos (10*α) - Real.cos (14*α)) /
  (Real.sin (2*α) + Real.sin (6*α) + Real.sin (10*α) + Real.sin (14*α)) =
  Real.tan (2*α) := by
  sorry

end expression_simplification_l4074_407498


namespace double_burger_cost_l4074_407443

/-- Calculates the cost of a double burger given the total spent, total number of hamburgers,
    cost of a single burger, and number of double burgers. -/
def cost_of_double_burger (total_spent : ℚ) (total_burgers : ℕ) (single_burger_cost : ℚ) (double_burgers : ℕ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let total_single_cost := single_burgers * single_burger_cost
  let total_double_cost := total_spent - total_single_cost
  total_double_cost / double_burgers

/-- Theorem stating that the cost of a double burger is $1.50 given the specific conditions. -/
theorem double_burger_cost :
  cost_of_double_burger 64.5 50 1 29 = 1.5 := by
  sorry

#eval cost_of_double_burger 64.5 50 1 29

end double_burger_cost_l4074_407443


namespace heather_walking_distance_l4074_407431

theorem heather_walking_distance (car_to_entrance : ℝ) (entrance_to_rides : ℝ) (rides_to_car : ℝ) 
  (h1 : car_to_entrance = 0.33)
  (h2 : entrance_to_rides = 0.33)
  (h3 : rides_to_car = 0.08) :
  car_to_entrance + entrance_to_rides + rides_to_car = 0.74 := by
  sorry

end heather_walking_distance_l4074_407431


namespace no_linear_term_implies_m_equals_four_l4074_407438

theorem no_linear_term_implies_m_equals_four :
  ∀ m : ℝ, (∀ x : ℝ, 2*x^2 + m*x = 4*x + 2) →
  (∀ x : ℝ, ∃ a b c : ℝ, a*x^2 + c = 0 ∧ 2*x^2 + m*x = 4*x + 2) →
  m = 4 :=
by sorry

end no_linear_term_implies_m_equals_four_l4074_407438


namespace tangent_line_at_negative_two_l4074_407407

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + b

-- State the theorem
theorem tangent_line_at_negative_two (b : ℝ) :
  b = -6 →
  let x₀ := -2
  let y₀ := f b x₀
  let m := (3 * x₀^2 - 12)  -- Derivative at x₀
  ∀ x, y₀ + m * (x - x₀) = 10 := by
sorry

-- Note: The actual proof is omitted as per instructions

end tangent_line_at_negative_two_l4074_407407


namespace alex_lead_after_even_l4074_407480

/-- Represents the race between Alex and Max -/
structure Race where
  total_length : ℕ
  initial_even : ℕ
  max_ahead : ℕ
  alex_final_ahead : ℕ
  remaining : ℕ

/-- Calculates the distance Alex got ahead of Max after they were even -/
def alex_initial_lead (r : Race) : ℕ :=
  r.total_length - r.remaining - r.initial_even - (r.max_ahead + r.alex_final_ahead)

/-- The theorem stating that Alex got ahead of Max by 300 feet after they were even -/
theorem alex_lead_after_even (r : Race) 
  (h1 : r.total_length = 5000)
  (h2 : r.initial_even = 200)
  (h3 : r.max_ahead = 170)
  (h4 : r.alex_final_ahead = 440)
  (h5 : r.remaining = 3890) :
  alex_initial_lead r = 300 := by
  sorry

#eval alex_initial_lead { total_length := 5000, initial_even := 200, max_ahead := 170, alex_final_ahead := 440, remaining := 3890 }

end alex_lead_after_even_l4074_407480


namespace new_average_is_44_l4074_407497

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  lastInningRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score after a given number of innings -/
def calculateAverage (performance : BatsmanPerformance) : Nat :=
  (performance.totalRuns + performance.lastInningRuns) / (performance.innings + 1)

/-- Theorem: Given the specific performance, prove the new average is 44 -/
theorem new_average_is_44 (performance : BatsmanPerformance)
  (h1 : performance.innings = 16)
  (h2 : performance.lastInningRuns = 92)
  (h3 : performance.averageIncrease = 3)
  (h4 : calculateAverage performance = calculateAverage performance - performance.averageIncrease + 3) :
  calculateAverage performance = 44 := by
  sorry

end new_average_is_44_l4074_407497


namespace system_solution_range_l4074_407400

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = 3 * k - 1) →
  (x + 2 * y = -2) →
  (x - y ≤ 5) →
  (k ≤ 4/3) := by
  sorry

end system_solution_range_l4074_407400


namespace twenty_numbers_arrangement_exists_l4074_407401

theorem twenty_numbers_arrangement_exists : ∃ (a b : ℝ), (a + 2 * b > 0) ∧ (7 * a + 13 * b < 0) := by
  sorry

end twenty_numbers_arrangement_exists_l4074_407401


namespace equilateral_triangle_perimeter_l4074_407466

theorem equilateral_triangle_perimeter (R : ℝ) (chord_length : ℝ) (chord_distance : ℝ) :
  chord_length = 2 →
  chord_distance = 3 →
  R^2 = chord_distance^2 + (chord_length/2)^2 →
  ∃ (perimeter : ℝ), perimeter = 3 * R * Real.sqrt 3 ∧ perimeter = 3 * Real.sqrt 30 :=
by sorry

end equilateral_triangle_perimeter_l4074_407466


namespace system_solution_relation_l4074_407484

theorem system_solution_relation (a₁ a₂ c₁ c₂ : ℝ) :
  (2 * a₁ + 3 = c₁ ∧ 2 * a₂ + 3 = c₂) →
  (∃! (x y : ℝ), a₁ * x + y = a₁ - c₁ ∧ a₂ * x + y = a₂ - c₂ ∧ x = -1 ∧ y = -3) := by
  sorry

end system_solution_relation_l4074_407484


namespace largest_k_for_positive_root_l4074_407451

/-- The equation in question -/
def equation (k : ℤ) (x : ℝ) : ℝ := 3*x*(2*k*x-5) - 2*x^2 + 8

/-- Predicate for the existence of a positive root -/
def has_positive_root (k : ℤ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ equation k x = 0

/-- The main theorem -/
theorem largest_k_for_positive_root :
  (∀ k : ℤ, k > 1 → ¬(has_positive_root k)) ∧
  has_positive_root 1 := by sorry

end largest_k_for_positive_root_l4074_407451


namespace expression_evaluation_l4074_407418

theorem expression_evaluation : 4 * (8 - 3 + 2) - 7 = 21 := by
  sorry

end expression_evaluation_l4074_407418


namespace modular_inverse_7_mod_120_l4074_407460

theorem modular_inverse_7_mod_120 :
  ∃ (x : ℕ), x < 120 ∧ (7 * x) % 120 = 1 ∧ x = 103 := by
  sorry

end modular_inverse_7_mod_120_l4074_407460


namespace haircut_cost_per_year_l4074_407481

/-- Calculates the total amount spent on haircuts in a year given the specified conditions. -/
theorem haircut_cost_per_year
  (growth_rate : ℝ)
  (initial_length : ℝ)
  (cut_length : ℝ)
  (haircut_cost : ℝ)
  (tip_percentage : ℝ)
  (months_per_year : ℕ)
  (h1 : growth_rate = 1.5)
  (h2 : initial_length = 9)
  (h3 : cut_length = 6)
  (h4 : haircut_cost = 45)
  (h5 : tip_percentage = 0.2)
  (h6 : months_per_year = 12) :
  (haircut_cost * (1 + tip_percentage) * (months_per_year / ((initial_length - cut_length) / growth_rate))) = 324 :=
by sorry

end haircut_cost_per_year_l4074_407481


namespace even_function_increasing_interval_l4074_407430

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The interval (-∞, 0] -/
def NegativeRealsAndZero : Set ℝ := { x | x ≤ 0 }

/-- A function f : ℝ → ℝ is increasing on a set S if f(x) ≤ f(y) for all x, y ∈ S with x ≤ y -/
def IncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

/-- The main theorem -/
theorem even_function_increasing_interval (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (a - 2) * x^2 + (a - 1) * x + 3
  IsEven f →
  IncreasingOn f NegativeRealsAndZero ∧
  ∀ S, IncreasingOn f S → S ⊆ NegativeRealsAndZero :=
sorry

end even_function_increasing_interval_l4074_407430


namespace hannah_final_pay_l4074_407435

def calculate_final_pay (hourly_rate : ℚ) (hours_worked : ℕ) (late_penalty : ℚ) 
  (times_late : ℕ) (federal_tax_rate : ℚ) (state_tax_rate : ℚ) (bonus_per_review : ℚ) 
  (qualifying_reviews : ℕ) (total_reviews : ℕ) : ℚ :=
  let gross_pay := hourly_rate * hours_worked
  let total_late_penalty := late_penalty * times_late
  let total_bonus := bonus_per_review * qualifying_reviews
  let adjusted_gross_pay := gross_pay - total_late_penalty + total_bonus
  let federal_tax := adjusted_gross_pay * federal_tax_rate
  let state_tax := adjusted_gross_pay * state_tax_rate
  let total_taxes := federal_tax + state_tax
  adjusted_gross_pay - total_taxes

theorem hannah_final_pay : 
  calculate_final_pay 30 18 5 3 (1/10) (1/20) 15 4 6 = 497.25 := by
  sorry

end hannah_final_pay_l4074_407435


namespace complex_number_solutions_l4074_407459

theorem complex_number_solutions : 
  ∀ z : ℂ, z^2 = -45 - 28*I ∧ z^3 = 8 + 26*I →
  z = Complex.mk (Real.sqrt 10) (-Real.sqrt 140) ∨
  z = Complex.mk (-Real.sqrt 10) (Real.sqrt 140) := by
sorry

end complex_number_solutions_l4074_407459


namespace sequence_value_l4074_407408

theorem sequence_value (a : ℕ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = n / (n + 1)) :
  a 8 = 1 / 8 := by
  sorry

end sequence_value_l4074_407408


namespace first_load_pieces_l4074_407437

theorem first_load_pieces (total : ℕ) (num_small_loads : ℕ) (pieces_per_small_load : ℕ) 
    (h1 : total = 47)
    (h2 : num_small_loads = 5)
    (h3 : pieces_per_small_load = 6) : 
  total - (num_small_loads * pieces_per_small_load) = 17 := by
  sorry

end first_load_pieces_l4074_407437


namespace circle_radius_is_five_l4074_407421

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y = 0

/-- The radius of the circle defined by circle_equation -/
def circle_radius : ℝ := 5

/-- Theorem stating that the radius of the circle defined by circle_equation is 5 -/
theorem circle_radius_is_five : 
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end circle_radius_is_five_l4074_407421


namespace only_one_implies_negation_l4074_407406

theorem only_one_implies_negation (p q : Prop) : 
  (∃! x : Fin 4, match x with
    | 0 => (p ∨ q) → ¬(p ∨ q)
    | 1 => (p ∧ ¬q) → ¬(p ∨ q)
    | 2 => (¬p ∧ q) → ¬(p ∨ q)
    | 3 => (¬p ∧ ¬q) → ¬(p ∨ q)
  ) := by sorry

end only_one_implies_negation_l4074_407406


namespace expected_worth_unfair_coin_l4074_407409

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  payoff_heads : ℝ
  payoff_tails : ℝ
  prob_sum : prob_heads + prob_tails = 1

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.payoff_heads + c.prob_tails * c.payoff_tails

/-- Theorem stating the expected worth of the specific unfair coin -/
theorem expected_worth_unfair_coin :
  ∃ c : UnfairCoin, c.prob_heads = 3/4 ∧ c.prob_tails = 1/4 ∧
  c.payoff_heads = 3 ∧ c.payoff_tails = -8 ∧ expected_worth c = 1/4 := by
  sorry

end expected_worth_unfair_coin_l4074_407409


namespace triangle_property_l4074_407439

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : t.b + t.c = 2 * t.a * Real.sin (t.C + π/6)) : 
  t.A = π/3 ∧ 1 < (t.b^2 + t.c^2) / t.a^2 ∧ (t.b^2 + t.c^2) / t.a^2 ≤ 2 :=
by
  sorry


end triangle_property_l4074_407439


namespace midpoint_movement_l4074_407465

/-- Given two points A and B in a Cartesian plane, their midpoint, and their new positions after
    movement, prove the new midpoint and its distance from the original midpoint. -/
theorem midpoint_movement (a b c d m n : ℝ) :
  let M : ℝ × ℝ := (m, n)
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (c, d)
  let A' : ℝ × ℝ := (a + 3, b + 5)
  let B' : ℝ × ℝ := (c - 6, d - 3)
  let M' : ℝ × ℝ := ((a + 3 + c - 6) / 2, (b + 5 + d - 3) / 2)
  (M = ((a + c) / 2, (b + d) / 2)) →
  (M' = (m - 3 / 2, n + 1) ∧
   Real.sqrt ((m - 3 / 2 - m) ^ 2 + (n + 1 - n) ^ 2) = Real.sqrt 13 / 2) :=
by sorry

end midpoint_movement_l4074_407465


namespace sequence_limit_inequality_l4074_407419

theorem sequence_limit_inequality (a b : ℕ → ℝ) (A B : ℝ) :
  (∀ n : ℕ, a n > b n) →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - A| < ε) →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |b n - B| < ε) →
  A ≥ B := by
  sorry

end sequence_limit_inequality_l4074_407419


namespace accurate_number_range_l4074_407475

/-- The approximate number obtained by rounding -/
def approximate_number : ℝ := 0.270

/-- A number rounds to the given approximate number if it's within 0.0005 of it -/
def rounds_to (x : ℝ) : Prop :=
  x ≥ approximate_number - 0.0005 ∧ x < approximate_number + 0.0005

/-- The theorem stating the range of the accurate number -/
theorem accurate_number_range (a : ℝ) (h : rounds_to a) :
  a ≥ 0.2695 ∧ a < 0.2705 := by
  sorry

end accurate_number_range_l4074_407475


namespace equation_solution_l4074_407445

theorem equation_solution (x : ℝ) (h : x ≠ 1/3) :
  (6 * x + 1) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 1) ↔
  x = -1 + (2 * Real.sqrt 3) / 3 ∨ x = -1 - (2 * Real.sqrt 3) / 3 :=
by sorry

end equation_solution_l4074_407445


namespace order_of_xyz_l4074_407470

theorem order_of_xyz (x y z : ℝ) 
  (h : x + 2013 / 2014 = y + 2012 / 2013 ∧ y + 2012 / 2013 = z + 2014 / 2015) : 
  z < y ∧ y < x := by
sorry

end order_of_xyz_l4074_407470


namespace amc10_paths_l4074_407427

/-- The number of 'M's adjacent to the central 'A' -/
def num_m_adj_a : ℕ := 4

/-- The number of 'C's adjacent to each 'M' -/
def num_c_adj_m : ℕ := 4

/-- The number of '10's adjacent to each 'C' -/
def num_10_adj_c : ℕ := 5

/-- The total number of paths to spell "AMC10" -/
def total_paths : ℕ := num_m_adj_a * num_c_adj_m * num_10_adj_c

theorem amc10_paths : total_paths = 80 := by
  sorry

end amc10_paths_l4074_407427


namespace polygon_and_calendar_problem_l4074_407414

theorem polygon_and_calendar_problem :
  ∀ (n k : ℕ),
  -- Regular polygon with interior angles of 160°
  (180 - 160 : ℝ) * n = 360 →
  -- The n-th day of May is Friday
  n % 7 = 5 →
  -- The k-th day of May is Tuesday
  k % 7 = 2 →
  -- 20 < k < 26
  20 < k ∧ k < 26 →
  -- Prove n = 18 and k = 22
  n = 18 ∧ k = 22 :=
by sorry

end polygon_and_calendar_problem_l4074_407414


namespace gym_class_laps_l4074_407402

/-- Given a total distance to run, track length, and number of laps already run by two people,
    calculate the number of additional laps needed to reach the total distance. -/
def additional_laps_needed (total_distance : ℕ) (track_length : ℕ) (laps_run_per_person : ℕ) : ℕ :=
  let total_laps_run := 2 * laps_run_per_person
  let distance_run := total_laps_run * track_length
  let remaining_distance := total_distance - distance_run
  remaining_distance / track_length

/-- Prove that for the given conditions, the number of additional laps needed is 4. -/
theorem gym_class_laps : additional_laps_needed 2400 150 6 = 4 := by
  sorry

end gym_class_laps_l4074_407402


namespace container_volume_ratio_l4074_407403

theorem container_volume_ratio :
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (3 / 4 : ℚ) * volume_container1 = (5 / 8 : ℚ) * volume_container2 →
  volume_container1 / volume_container2 = (5 / 6 : ℚ) := by
sorry

end container_volume_ratio_l4074_407403


namespace arithmetic_mean_of_18_27_45_l4074_407489

def arithmetic_mean (a b c : ℕ) : ℚ :=
  (a + b + c : ℚ) / 3

theorem arithmetic_mean_of_18_27_45 :
  arithmetic_mean 18 27 45 = 30 := by
  sorry

end arithmetic_mean_of_18_27_45_l4074_407489


namespace factorial_ratio_l4074_407424

theorem factorial_ratio (n : ℕ) (h : n > 0) : (n.factorial) / ((n-1).factorial) = n := by
  sorry

end factorial_ratio_l4074_407424


namespace probability_two_red_shoes_l4074_407471

def total_shoes : ℕ := 10
def red_shoes : ℕ := 6
def green_shoes : ℕ := 4
def shoes_drawn : ℕ := 2

theorem probability_two_red_shoes :
  (Nat.choose red_shoes shoes_drawn) / (Nat.choose total_shoes shoes_drawn) = 1 / 3 := by
sorry

end probability_two_red_shoes_l4074_407471


namespace cricket_innings_calculation_l4074_407499

/-- Given a cricket player's performance data, calculate the number of innings played. -/
theorem cricket_innings_calculation (current_average : ℚ) (next_innings_runs : ℚ) (average_increase : ℚ) : 
  current_average = 35 →
  next_innings_runs = 79 →
  average_increase = 4 →
  ∃ n : ℕ, n > 0 ∧ (n : ℚ) * current_average + next_innings_runs = ((n + 1) : ℚ) * (current_average + average_increase) ∧ n = 10 :=
by sorry

end cricket_innings_calculation_l4074_407499


namespace gcd_105_210_l4074_407494

theorem gcd_105_210 : Nat.gcd 105 210 = 105 := by
  sorry

end gcd_105_210_l4074_407494


namespace zoo_animal_types_l4074_407490

theorem zoo_animal_types :
  let viewing_time : ℕ → ℕ := λ n => 6 * n
  let initial_types : ℕ := 5
  let added_types : ℕ := 4
  let total_time : ℕ := 54
  viewing_time (initial_types + added_types) = total_time :=
by
  sorry

#check zoo_animal_types

end zoo_animal_types_l4074_407490


namespace max_xy_given_constraint_l4074_407429

theorem max_xy_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + 9 * y = 60) :
  x * y ≤ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + 9 * y₀ = 60 ∧ x₀ * y₀ = 25 := by
  sorry

end max_xy_given_constraint_l4074_407429


namespace busy_squirrel_nuts_calculation_l4074_407486

/-- The number of nuts stockpiled per day by each busy squirrel -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of busy squirrels -/
def num_busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled per day by the sleepy squirrel -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The number of days the squirrels have been stockpiling -/
def num_days : ℕ := 40

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := 3200

theorem busy_squirrel_nuts_calculation :
  num_busy_squirrels * busy_squirrel_nuts_per_day * num_days + 
  sleepy_squirrel_nuts_per_day * num_days = total_nuts :=
by sorry

end busy_squirrel_nuts_calculation_l4074_407486


namespace distribute_10_3_1_l4074_407426

/-- The number of ways to distribute n identical objects into k identical containers,
    with each container having at least m objects. -/
def distribute (n k m : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical coins into 3 identical bags,
    with each bag having at least 1 coin. -/
theorem distribute_10_3_1 : distribute 10 3 1 = 8 := sorry

end distribute_10_3_1_l4074_407426


namespace complex_purely_imaginary_solution_l4074_407413

-- Define a complex number to be purely imaginary if its real part is zero
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_purely_imaginary_solution (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z - 3)^2 + 5*I)) : 
  z = 3*I ∨ z = -3*I := by
  sorry

end complex_purely_imaginary_solution_l4074_407413


namespace max_sum_under_constraint_l4074_407493

theorem max_sum_under_constraint (m n : ℤ) (h : 205 * m^2 + 409 * n^4 ≤ 20736) :
  m + n ≤ 12 :=
sorry

end max_sum_under_constraint_l4074_407493


namespace normal_distribution_symmetry_l4074_407404

-- Define a normally distributed random variable
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
def P (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (x : normal_dist 4 σ) 
  (h : P {y : ℝ | y > 2} = 0.6) :
  P {y : ℝ | y > 6} = 0.4 := by sorry

end normal_distribution_symmetry_l4074_407404


namespace trig_difference_equals_sqrt_three_l4074_407425

-- Define the problem
theorem trig_difference_equals_sqrt_three :
  (1 / Real.tan (20 * π / 180)) - (1 / Real.cos (10 * π / 180)) = Real.sqrt 3 := by
  sorry

end trig_difference_equals_sqrt_three_l4074_407425
