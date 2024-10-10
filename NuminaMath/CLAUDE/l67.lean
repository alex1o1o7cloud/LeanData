import Mathlib

namespace expected_value_of_three_marbles_l67_6709

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sumOfThree (s : Finset ℕ) : ℕ := s.sum id

def allCombinations : Finset (Finset ℕ) :=
  marbles.powerset.filter (λ s => s.card = 3)

def expectedValue : ℚ :=
  (allCombinations.sum sumOfThree) / allCombinations.card

theorem expected_value_of_three_marbles :
  expectedValue = 21/2 := by sorry

end expected_value_of_three_marbles_l67_6709


namespace apple_price_l67_6741

/-- The price of apples given Emmy's and Gerry's money and the total number of apples they can buy -/
theorem apple_price (emmy_money : ℝ) (gerry_money : ℝ) (total_apples : ℝ) 
  (h1 : emmy_money = 200)
  (h2 : gerry_money = 100)
  (h3 : total_apples = 150) :
  (emmy_money + gerry_money) / total_apples = 2 := by
  sorry

end apple_price_l67_6741


namespace scenic_area_ticket_sales_l67_6772

/-- Scenic area ticket sales problem -/
theorem scenic_area_ticket_sales 
  (parent_child_price : ℝ) 
  (family_price : ℝ) 
  (parent_child_presale : ℝ) 
  (family_presale : ℝ) 
  (volume_difference : ℕ) 
  (parent_child_planned : ℕ) 
  (family_planned : ℕ) 
  (a : ℝ) :
  family_price = 2 * parent_child_price →
  parent_child_presale = 21000 →
  family_presale = 10500 →
  (parent_child_presale / parent_child_price) - (family_presale / family_price) = volume_difference →
  parent_child_planned = 1600 →
  family_planned = 400 →
  (parent_child_price + 3/4 * a) * (parent_child_planned - 32 * a) + 
    (family_price + a) * family_planned = 
    parent_child_price * parent_child_planned + family_price * family_planned →
  parent_child_price = 35 ∧ a = 20 := by
  sorry

end scenic_area_ticket_sales_l67_6772


namespace mork_tax_rate_calculation_l67_6788

-- Define the variables
def mork_income : ℝ := sorry
def mork_tax_rate : ℝ := sorry
def mindy_tax_rate : ℝ := 0.25
def combined_tax_rate : ℝ := 0.28

-- Define the theorem
theorem mork_tax_rate_calculation :
  mork_tax_rate = 0.4 :=
by
  -- Assume the conditions
  have h1 : mindy_tax_rate = 0.25 := rfl
  have h2 : combined_tax_rate = 0.28 := rfl
  have h3 : mork_income > 0 := sorry
  have h4 : mork_tax_rate * mork_income + mindy_tax_rate * (4 * mork_income) = combined_tax_rate * (5 * mork_income) := sorry

  -- Proof steps would go here
  sorry

end mork_tax_rate_calculation_l67_6788


namespace quadratic_has_real_root_l67_6755

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end quadratic_has_real_root_l67_6755


namespace gcd_of_integer_differences_l67_6737

theorem gcd_of_integer_differences (a b c d : ℤ) : 
  ∃ k : ℤ, (a - b) * (b - c) * (c - d) * (d - a) * (a - c) * (b - d) = 12 * k :=
sorry

end gcd_of_integer_differences_l67_6737


namespace total_bouncy_balls_l67_6790

def red_packs : ℕ := 4
def yellow_packs : ℕ := 8
def green_packs : ℕ := 4
def balls_per_pack : ℕ := 10

theorem total_bouncy_balls :
  (red_packs + yellow_packs + green_packs) * balls_per_pack = 160 := by
  sorry

end total_bouncy_balls_l67_6790


namespace inscribing_square_area_l67_6786

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 8 * x - 12 * y + 24 = 0

/-- The square inscribing the circle -/
structure InscribingSquare where
  side : ℝ
  center_x : ℝ
  center_y : ℝ
  inscribes_circle : ∀ (x y : ℝ), circle_equation x y →
    (|x - center_x| ≤ side / 2) ∧ (|y - center_y| ≤ side / 2)
  parallel_to_axes : True  -- This condition is implicit in the structure

/-- The theorem stating that the area of the inscribing square is 4 -/
theorem inscribing_square_area :
  ∀ (s : InscribingSquare), s.side^2 = 4 := by sorry

end inscribing_square_area_l67_6786


namespace max_value_abc_fraction_l67_6784

theorem max_value_abc_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ (1 : ℝ) / 4 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) = (1 : ℝ) / 4 :=
sorry

end max_value_abc_fraction_l67_6784


namespace mango_count_l67_6783

/-- The number of mangoes in all boxes -/
def total_mangoes (boxes : ℕ) (dozen_per_box : ℕ) : ℕ :=
  boxes * dozen_per_box * 12

/-- Proof that there are 4320 mangoes in 36 boxes with 10 dozen mangoes each -/
theorem mango_count : total_mangoes 36 10 = 4320 := by
  sorry

end mango_count_l67_6783


namespace order_of_numbers_l67_6734

theorem order_of_numbers : (2 : ℝ)^24 < 10^8 ∧ 10^8 < 5^12 := by sorry

end order_of_numbers_l67_6734


namespace speed_conversion_l67_6732

/-- Conversion of speed from m/s to km/h -/
theorem speed_conversion (speed_ms : ℚ) (conversion_factor : ℚ) :
  speed_ms = 13/36 →
  conversion_factor = 36/10 →
  speed_ms * conversion_factor = 13/10 := by
  sorry

#eval (13/36 : ℚ) * (36/10 : ℚ) -- To verify the result

end speed_conversion_l67_6732


namespace second_number_in_sum_l67_6725

theorem second_number_in_sum (a b c : ℝ) : 
  a = 3.15 → c = 0.458 → a + b + c = 3.622 → b = 0.014 := by
  sorry

end second_number_in_sum_l67_6725


namespace isosceles_trapezoid_prism_height_isosceles_trapezoid_prism_height_proof_l67_6717

/-- Represents a prism with a base that is an isosceles trapezoid inscribed around a circle -/
structure IsoscelesTrapezoidPrism where
  r : ℝ  -- radius of the inscribed circle
  α : ℝ  -- acute angle of the trapezoid

/-- 
Theorem: The height of the prism is 2r tan(α) given:
- The base is an isosceles trapezoid inscribed around a circle with radius r
- The acute angle of the trapezoid is α
- A plane passing through one side of the base and the acute angle endpoint 
  of the opposite side of the top plane forms an angle α with the base plane
-/
theorem isosceles_trapezoid_prism_height 
  (prism : IsoscelesTrapezoidPrism) : ℝ :=
  2 * prism.r * Real.tan prism.α

-- Proof
theorem isosceles_trapezoid_prism_height_proof
  (prism : IsoscelesTrapezoidPrism) :
  isosceles_trapezoid_prism_height prism = 2 * prism.r * Real.tan prism.α := by
  sorry

end isosceles_trapezoid_prism_height_isosceles_trapezoid_prism_height_proof_l67_6717


namespace line_not_through_point_l67_6798

theorem line_not_through_point (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (2*m+1)*x₁ + m^2 + 4 = 0 ∧ x₂^2 + (2*m+1)*x₂ + m^2 + 4 = 0) →
  ¬((2*m-3)*(-2) - 4*m + 7 = 1) :=
by sorry

end line_not_through_point_l67_6798


namespace biancas_album_pictures_l67_6745

/-- Given that Bianca uploaded 33 pictures and put some into 3 albums with 2 pictures each,
    prove that she put 27 pictures into the first album. -/
theorem biancas_album_pictures :
  ∀ (total_pictures : ℕ) (other_albums : ℕ) (pics_per_album : ℕ),
    total_pictures = 33 →
    other_albums = 3 →
    pics_per_album = 2 →
    total_pictures - (other_albums * pics_per_album) = 27 :=
by
  sorry

end biancas_album_pictures_l67_6745


namespace quadratic_square_of_binomial_l67_6705

/-- If x^2 + 110x + d is equal to the square of a binomial, then d = 3025 -/
theorem quadratic_square_of_binomial (d : ℝ) :
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 110*x + d = (x + b)^2) → d = 3025 :=
by sorry

end quadratic_square_of_binomial_l67_6705


namespace paiges_files_l67_6710

theorem paiges_files (deleted_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) :
  deleted_files = 9 →
  files_per_folder = 6 →
  num_folders = 3 →
  deleted_files + (files_per_folder * num_folders) = 27 :=
by
  sorry

end paiges_files_l67_6710


namespace paintable_area_is_1520_l67_6723

/-- Calculates the total paintable area of walls in multiple bedrooms. -/
def total_paintable_area (num_bedrooms length width height unpaintable_area : ℝ) : ℝ :=
  num_bedrooms * ((2 * (length * height + width * height)) - unpaintable_area)

/-- Proves that the total paintable area of walls in 4 bedrooms is 1520 square feet. -/
theorem paintable_area_is_1520 :
  total_paintable_area 4 14 11 9 70 = 1520 := by
  sorry

end paintable_area_is_1520_l67_6723


namespace M_equals_P_l67_6795

/-- Set M defined as {x | x = a² + 1, a ∈ ℝ} -/
def M : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}

/-- Set P defined as {y | y = b² - 4b + 5, b ∈ ℝ} -/
def P : Set ℝ := {y | ∃ b : ℝ, y = b^2 - 4*b + 5}

/-- Theorem stating that M = P -/
theorem M_equals_P : M = P := by sorry

end M_equals_P_l67_6795


namespace sofa_price_calculation_l67_6702

def living_room_set_price (sofa_price armchair_price coffee_table_price : ℝ) : ℝ :=
  sofa_price + 2 * armchair_price + coffee_table_price

theorem sofa_price_calculation (armchair_price coffee_table_price total_price : ℝ)
  (h1 : armchair_price = 425)
  (h2 : coffee_table_price = 330)
  (h3 : total_price = 2430)
  (h4 : living_room_set_price (total_price - 2 * armchair_price - coffee_table_price) armchair_price coffee_table_price = total_price) :
  total_price - 2 * armchair_price - coffee_table_price = 1250 := by
  sorry

#check sofa_price_calculation

end sofa_price_calculation_l67_6702


namespace min_value_implies_a_eq_four_l67_6785

/-- Given a function f(x) = 4x + a²/x where x > 0 and x ∈ ℝ, 
    if f attains its minimum value at x = 2, then a = 4. -/
theorem min_value_implies_a_eq_four (a : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ (f : ℝ → ℝ), f x = 4*x + a^2/x) →
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, x > 0 → f x ≥ f 2) →
  a = 4 := by
  sorry


end min_value_implies_a_eq_four_l67_6785


namespace james_age_proof_l67_6781

/-- James' age -/
def james_age : ℝ := 47.5

/-- Mara's age -/
def mara_age : ℝ := 22.5

/-- James' age is 20 years less than three times Mara's age -/
axiom age_relation : james_age = 3 * mara_age - 20

/-- The sum of their ages is 70 -/
axiom age_sum : james_age + mara_age = 70

theorem james_age_proof : james_age = 47.5 := by
  sorry

end james_age_proof_l67_6781


namespace hexadecagon_triangles_l67_6756

/-- The number of sides in a regular hexadecagon -/
def n : ℕ := 16

/-- The number of vertices to choose for each triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular hexadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end hexadecagon_triangles_l67_6756


namespace max_gingerbread_production_l67_6743

/-- The gingerbread production function -/
def gingerbread_production (k : ℝ) (t : ℝ) : ℝ := k * t * (24 - t)

/-- Theorem stating that gingerbread production is maximized at 16 hours of work -/
theorem max_gingerbread_production (k : ℝ) (h : k > 0) :
  ∃ (t : ℝ), t = 16 ∧ ∀ (s : ℝ), 0 ≤ s ∧ s ≤ 24 → gingerbread_production k s ≤ gingerbread_production k t :=
by
  sorry

#check max_gingerbread_production

end max_gingerbread_production_l67_6743


namespace henry_lap_time_l67_6740

theorem henry_lap_time (margo_lap_time henry_lap_time meet_time : ℕ) 
  (h1 : margo_lap_time = 12)
  (h2 : meet_time = 84)
  (h3 : meet_time % margo_lap_time = 0)
  (h4 : meet_time % henry_lap_time = 0)
  (h5 : henry_lap_time < margo_lap_time)
  : henry_lap_time = 7 :=
sorry

end henry_lap_time_l67_6740


namespace quadratic_equal_roots_l67_6736

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 24 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 24 = 0 → y = x) ↔ 
  (k = 2 + 12 * Real.sqrt 2 ∨ k = 2 - 12 * Real.sqrt 2) :=
by sorry

end quadratic_equal_roots_l67_6736


namespace calculate_small_orders_l67_6729

/-- Given information about packing peanuts usage in orders, calculate the number of small orders. -/
theorem calculate_small_orders (total_peanuts : ℕ) (large_orders : ℕ) (peanuts_per_large : ℕ) (peanuts_per_small : ℕ) :
  total_peanuts = 800 →
  large_orders = 3 →
  peanuts_per_large = 200 →
  peanuts_per_small = 50 →
  (total_peanuts - large_orders * peanuts_per_large) / peanuts_per_small = 4 :=
by sorry

end calculate_small_orders_l67_6729


namespace gcd_459_357_l67_6777

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l67_6777


namespace binary_101101_is_45_l67_6774

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_is_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end binary_101101_is_45_l67_6774


namespace project_hours_difference_l67_6730

theorem project_hours_difference (total : ℕ) (k p m : ℕ) : 
  total = k + p + m →
  p = 2 * k →
  3 * p = m →
  total = 153 →
  m - k = 85 := by sorry

end project_hours_difference_l67_6730


namespace point_on_circle_l67_6708

theorem point_on_circle (t : ℝ) :
  let x := (3 * t^2 - 1) / (t^2 + 3)
  let y := 6 * t / (t^2 + 3)
  x^2 + y^2 = 1 := by
sorry

end point_on_circle_l67_6708


namespace point_b_position_l67_6728

theorem point_b_position (a b : ℝ) : 
  a = -2 → (b - a = 4 ∨ a - b = 4) → (b = 2 ∨ b = -6) := by
  sorry

end point_b_position_l67_6728


namespace quadratic_equation_roots_l67_6793

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end quadratic_equation_roots_l67_6793


namespace max_servings_jordan_l67_6735

/-- Represents the recipe for hot chocolate -/
structure Recipe where
  servings : ℚ
  chocolate : ℚ
  sugar : ℚ
  water : ℚ
  milk : ℚ

/-- Represents the available ingredients -/
structure Ingredients where
  chocolate : ℚ
  sugar : ℚ
  milk : ℚ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (recipe : Recipe) (ingredients : Ingredients) : ℚ :=
  min (ingredients.chocolate / recipe.chocolate * recipe.servings)
      (min (ingredients.sugar / recipe.sugar * recipe.servings)
           (ingredients.milk / recipe.milk * recipe.servings))

theorem max_servings_jordan :
  let recipe : Recipe := ⟨5, 2, 1/4, 1, 4⟩
  let ingredients : Ingredients := ⟨5, 2, 7⟩
  maxServings recipe ingredients = 35/4 := by
  sorry

end max_servings_jordan_l67_6735


namespace project_completion_l67_6700

theorem project_completion 
  (a b c d e : ℕ) 
  (f g : ℝ) 
  (h₁ : a > 0) 
  (h₂ : c > 0) 
  (h₃ : f > 0) 
  (h₄ : g > 0) :
  (d : ℝ) * (b : ℝ) * g * (e : ℝ) / ((c : ℝ) * (a : ℝ)) = 
  (b : ℝ) * (d : ℝ) * g * (e : ℝ) / ((c : ℝ) * (a : ℝ)) :=
by sorry

#check project_completion

end project_completion_l67_6700


namespace cubic_expression_equals_zero_l67_6720

theorem cubic_expression_equals_zero (k : ℝ) (h : k = 2) : (k^3 - 8) * (k + 1) = 0 := by
  sorry

end cubic_expression_equals_zero_l67_6720


namespace als_original_portion_l67_6721

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  a - 150 + 3 * b + 3 * c = 1800 →
  c = 2 * b →
  a = 825 :=
by sorry

end als_original_portion_l67_6721


namespace hidden_square_exists_l67_6773

theorem hidden_square_exists (ℓ : ℕ) : ∃ (x y : ℤ) (p : Fin ℓ → Fin ℓ → ℕ), 
  (∀ (i j : Fin ℓ), Nat.Prime (p i j)) ∧ 
  (∀ (i j k m : Fin ℓ), i ≠ k ∨ j ≠ m → p i j ≠ p k m) ∧
  (∀ (i j : Fin ℓ), x ≡ -i.val [ZMOD (p i j)] ∧ y ≡ -j.val [ZMOD (p i j)]) :=
sorry

end hidden_square_exists_l67_6773


namespace two_numbers_difference_l67_6706

theorem two_numbers_difference (x y : ℝ) (h_sum : x + y = 25) (h_product : x * y = 144) :
  |x - y| = 7 := by sorry

end two_numbers_difference_l67_6706


namespace average_birds_per_site_l67_6751

-- Define the data for each day
def monday_sites : ℕ := 5
def monday_avg : ℕ := 7
def tuesday_sites : ℕ := 5
def tuesday_avg : ℕ := 5
def wednesday_sites : ℕ := 10
def wednesday_avg : ℕ := 8

-- Define the total number of sites
def total_sites : ℕ := monday_sites + tuesday_sites + wednesday_sites

-- Define the total number of birds
def total_birds : ℕ := monday_sites * monday_avg + tuesday_sites * tuesday_avg + wednesday_sites * wednesday_avg

-- Theorem to prove
theorem average_birds_per_site :
  total_birds / total_sites = 7 := by
  sorry

end average_birds_per_site_l67_6751


namespace sqrt_n_squared_minus_np_integer_l67_6754

theorem sqrt_n_squared_minus_np_integer (p : ℕ) (hp : Prime p) (hodd : Odd p) :
  ∃! n : ℕ, n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n^2 - n*p = k^2 ∧ n = ((p + 1)^2) / 4 := by
  sorry

end sqrt_n_squared_minus_np_integer_l67_6754


namespace perfect_linearity_implies_R_squared_one_l67_6718

/-- A scatter plot is perfectly linear if all its points fall on a straight line with non-zero slope -/
def is_perfectly_linear (scatter_plot : Set (ℝ × ℝ)) : Prop :=
  ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ scatter_plot → y = m * x + b

/-- The coefficient of determination (R²) for a scatter plot -/
def R_squared (scatter_plot : Set (ℝ × ℝ)) : ℝ := sorry

theorem perfect_linearity_implies_R_squared_one (scatter_plot : Set (ℝ × ℝ)) :
  is_perfectly_linear scatter_plot → R_squared scatter_plot = 1 := by sorry

end perfect_linearity_implies_R_squared_one_l67_6718


namespace rounded_product_less_than_original_l67_6761

theorem rounded_product_less_than_original
  (x y z : ℝ)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (hz_pos : z > 0)
  (hxy : x > 2*y) :
  (x + z) * (y - z) < x * y :=
by sorry

end rounded_product_less_than_original_l67_6761


namespace daisies_given_away_l67_6766

/-- Proves the number of daisies given away based on initial count, petals per daisy, and remaining petals --/
theorem daisies_given_away 
  (initial_daisies : ℕ) 
  (petals_per_daisy : ℕ) 
  (remaining_petals : ℕ) 
  (h1 : initial_daisies = 5)
  (h2 : petals_per_daisy = 8)
  (h3 : remaining_petals = 24) :
  initial_daisies - (remaining_petals / petals_per_daisy) = 2 :=
by
  sorry

#check daisies_given_away

end daisies_given_away_l67_6766


namespace crayons_count_l67_6715

/-- The number of rows of crayons -/
def num_rows : ℕ := 7

/-- The number of crayons in each row -/
def crayons_per_row : ℕ := 30

/-- The total number of crayons -/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem crayons_count : total_crayons = 210 := by
  sorry

end crayons_count_l67_6715


namespace problem_statement_l67_6713

theorem problem_statement (x : ℝ) : 
  x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10 →
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 289/8 := by
  sorry

end problem_statement_l67_6713


namespace quadratic_polynomial_value_l67_6787

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPoly (p : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x, p x = a * x^2 + b * x + c

theorem quadratic_polynomial_value (p : ℤ → ℤ) :
  QuadraticPoly p →
  p 41 = 42 →
  (∃ a b : ℤ, a > 41 ∧ b > 41 ∧ p a = 13 ∧ p b = 73) →
  p 1 = 2842 :=
by sorry

end quadratic_polynomial_value_l67_6787


namespace arithmetic_sequence_max_sum_l67_6765

/-- The sum of the first n terms of an arithmetic sequence with a₁ = 23 and d = -2 -/
def S (n : ℕ+) : ℝ := -n.val^2 + 24 * n.val

/-- The maximum value of S(n) for positive integer n -/
def max_S : ℝ := 144

theorem arithmetic_sequence_max_sum :
  ∃ (n : ℕ+), S n = max_S ∧ ∀ (m : ℕ+), S m ≤ max_S := by
  sorry

end arithmetic_sequence_max_sum_l67_6765


namespace art_dealer_loss_l67_6727

theorem art_dealer_loss (selling_price : ℝ) (selling_price_positive : selling_price > 0) :
  let profit_percentage : ℝ := 0.1
  let loss_percentage : ℝ := 0.1
  let cost_price_1 : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_2 : ℝ := selling_price / (1 - loss_percentage)
  let profit : ℝ := selling_price - cost_price_1
  let loss : ℝ := cost_price_2 - selling_price
  let net_loss : ℝ := loss - profit
  net_loss = 0.02 * selling_price :=
by sorry

end art_dealer_loss_l67_6727


namespace absolute_difference_l67_6789

theorem absolute_difference (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := by
  sorry

end absolute_difference_l67_6789


namespace tickets_left_l67_6750

/-- The number of tickets Dave started with -/
def initial_tickets : ℕ := 98

/-- The number of tickets Dave spent on the stuffed tiger -/
def spent_tickets : ℕ := 43

/-- Theorem stating that Dave had 55 tickets left after spending on the stuffed tiger -/
theorem tickets_left : initial_tickets - spent_tickets = 55 := by
  sorry

end tickets_left_l67_6750


namespace imaginary_part_of_complex_fraction_l67_6762

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 4 * i / (1 + i)
  Complex.im z = 2 := by sorry

end imaginary_part_of_complex_fraction_l67_6762


namespace investment_problem_l67_6724

theorem investment_problem (x y : ℝ) : 
  x * 0.10 - y * 0.08 = 83 →
  y = 650 →
  x + y = 2000 :=
by
  sorry

end investment_problem_l67_6724


namespace square_of_sum_l67_6764

theorem square_of_sum (x : ℝ) (h1 : x^2 - 49 ≥ 0) (h2 : x + 7 ≥ 0) :
  (7 - Real.sqrt (x^2 - 49) + Real.sqrt (x + 7))^2 =
  x^2 + x + 7 - 14 * Real.sqrt (x^2 - 49) - 14 * Real.sqrt (x + 7) + 2 * Real.sqrt (x^2 - 49) * Real.sqrt (x + 7) := by
  sorry

end square_of_sum_l67_6764


namespace star_value_of_a_l67_6782

-- Define the star operation
def star (a b : ℝ) : ℝ := 3 * a - b^2

-- State the theorem
theorem star_value_of_a : ∃ a : ℝ, star a 4 = 14 ∧ a = 10 := by
  sorry

end star_value_of_a_l67_6782


namespace sum_of_f_values_l67_6746

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2/x) + 1

theorem sum_of_f_values : 
  f (-7) + f (-5) + f (-3) + f (-1) + f 3 + f 5 + f 7 + f 9 = 8 := by
  sorry

end sum_of_f_values_l67_6746


namespace min_value_of_expression_l67_6747

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  a / (4 - a) + 1 / (a - 1) ≥ 2 ∧
  (a / (4 - a) + 1 / (a - 1) = 2 ↔ a = 2) :=
by sorry

end min_value_of_expression_l67_6747


namespace percentage_problem_l67_6753

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 1280 = (20 / 100) * 650 + 190 → P = 25 := by
  sorry

end percentage_problem_l67_6753


namespace min_value_theorem_l67_6731

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  x + 4 / (x - 1) ≥ 5 ∧ (x + 4 / (x - 1) = 5 ↔ x = 3) := by
  sorry

end min_value_theorem_l67_6731


namespace stone_distribution_fractions_l67_6769

/-- Number of indistinguishable stones -/
def n : ℕ := 12

/-- Number of distinguishable boxes -/
def k : ℕ := 4

/-- Total number of ways to distribute n stones among k boxes -/
def total_distributions : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Number of ways to distribute stones with even number in each box -/
def even_distributions : ℕ := Nat.choose ((n / 2) + k - 1) (k - 1)

/-- Number of ways to distribute stones with odd number in each box -/
def odd_distributions : ℕ := Nat.choose ((n - k) / 2 + k - 1) (k - 1)

theorem stone_distribution_fractions :
  (even_distributions : ℚ) / total_distributions = 12 / 65 ∧
  (odd_distributions : ℚ) / total_distributions = 1 / 13 := by
  sorry

end stone_distribution_fractions_l67_6769


namespace square_area_equal_perimeter_triangle_l67_6716

theorem square_area_equal_perimeter_triangle (s : ℝ) :
  let triangle_perimeter := 5.5 + 5.5 + 7
  let square_side := triangle_perimeter / 4
  s = square_side → s^2 = 20.25 := by
sorry

end square_area_equal_perimeter_triangle_l67_6716


namespace count_zeros_up_to_3017_l67_6767

/-- A function that checks if a positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- The count of positive integers less than or equal to 3017 that contain the digit 0 -/
def countZeros : ℕ :=
  sorry

/-- Theorem stating that the count of positive integers less than or equal to 3017
    containing the digit 0 is equal to 1011 -/
theorem count_zeros_up_to_3017 : countZeros = 1011 := by
  sorry

end count_zeros_up_to_3017_l67_6767


namespace exists_segment_with_sum_455_l67_6748

/-- Represents a 10x10 table filled with numbers 1 to 100 as described in the problem -/
def Table := Matrix (Fin 10) (Fin 10) Nat

/-- Defines how the table is filled -/
def fillTable : Table :=
  fun i j => i.val * 10 + j.val + 1

/-- Represents a 7-cell segment in the specified form -/
structure Segment where
  center : Fin 10 × Fin 10
  direction : Bool  -- True for vertical, False for horizontal

/-- Calculates the sum of a segment -/
def segmentSum (t : Table) (s : Segment) : Nat :=
  let (i, j) := s.center
  if s.direction then
    t i j + t (i-1) j + t (i+1) j +
    t (i-1) (j-1) + t (i-1) (j+1) +
    t (i+1) (j-1) + t (i+1) (j+1)
  else
    t i j + t i (j-1) + t i (j+1) +
    t (i-1) (j-1) + t (i+1) (j-1) +
    t (i-1) (j+1) + t (i+1) (j+1)

/-- The main theorem to prove -/
theorem exists_segment_with_sum_455 :
  ∃ s : Segment, segmentSum fillTable s = 455 := by
  sorry

end exists_segment_with_sum_455_l67_6748


namespace solve_rent_problem_l67_6797

def rent_problem (n : ℕ) : Prop :=
  let original_average : ℚ := 800
  let increased_rent : ℚ := 800 * (1 + 1/4)
  let new_average : ℚ := 850
  (n * original_average + (increased_rent - 800)) / n = new_average

theorem solve_rent_problem : 
  ∃ (n : ℕ), n > 0 ∧ rent_problem n ∧ n = 4 := by
sorry

end solve_rent_problem_l67_6797


namespace albums_theorem_l67_6794

-- Define the number of albums for each person
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- Define the total number of albums
def total_albums : ℕ := adele_albums + bridget_albums + katrina_albums + miriam_albums

-- Theorem to prove
theorem albums_theorem : total_albums = 585 := by
  sorry

end albums_theorem_l67_6794


namespace abby_and_damon_weight_l67_6791

/-- The combined weight of Abby and Damon given the weights of other pairs -/
theorem abby_and_damon_weight
  (a b c d : ℝ)
  (h1 : a + b = 280)
  (h2 : b + c = 265)
  (h3 : c + d = 290)
  (h4 : b + d = 275) :
  a + d = 305 :=
by sorry

end abby_and_damon_weight_l67_6791


namespace divisor_product_sum_theorem_l67_6799

/-- The type of positive divisors of n -/
def Divisor (n : ℕ) := { d : ℕ // d > 0 ∧ n % d = 0 }

/-- The list of all positive divisors of n in ascending order -/
def divisors (n : ℕ) : List (Divisor n) := sorry

/-- The sum of products of consecutive divisors -/
def D (n : ℕ) : ℕ :=
  let ds := divisors n
  (List.zip ds (List.tail ds)).map (fun (d₁, d₂) => d₁.val * d₂.val) |>.sum

/-- The main theorem -/
theorem divisor_product_sum_theorem (n : ℕ) (h : n > 1) :
  D n < n^2 ∧ (D n ∣ n^2 ↔ Nat.Prime n) := by sorry

end divisor_product_sum_theorem_l67_6799


namespace game_probabilities_l67_6757

/-- Represents the number of balls of each color in the bag -/
def num_balls_per_color : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := 3 * num_balls_per_color

/-- Represents the number of balls drawn in each game -/
def balls_drawn : ℕ := 3

/-- Represents the number of people participating in the game -/
def num_participants : ℕ := 3

/-- Calculates the probability of winning for one person -/
def prob_win : ℚ := 2 / 5

/-- Calculates the probability that exactly 1 person wins out of 3 -/
def prob_one_winner : ℚ := 54 / 125

theorem game_probabilities :
  (prob_win = 2 / 5) ∧
  (prob_one_winner = 54 / 125) := by
  sorry

end game_probabilities_l67_6757


namespace tv_selection_probability_l67_6719

def num_type_a : ℕ := 3
def num_type_b : ℕ := 2
def total_tvs : ℕ := num_type_a + num_type_b
def selection_size : ℕ := 2

theorem tv_selection_probability :
  let total_combinations := Nat.choose total_tvs selection_size
  let favorable_combinations := num_type_a * num_type_b
  (favorable_combinations : ℚ) / total_combinations = 3 / 5 := by sorry

end tv_selection_probability_l67_6719


namespace nathan_writes_25_letters_per_hour_l67_6749

/-- The number of letters Nathan can write in one hour -/
def nathan_letters_per_hour : ℕ := sorry

/-- The number of letters Jacob can write in one hour -/
def jacob_letters_per_hour : ℕ := sorry

/-- Jacob writes twice as fast as Nathan -/
axiom jacob_twice_as_fast : jacob_letters_per_hour = 2 * nathan_letters_per_hour

/-- Together, Jacob and Nathan can write 750 letters in 10 hours -/
axiom combined_output : 10 * (jacob_letters_per_hour + nathan_letters_per_hour) = 750

theorem nathan_writes_25_letters_per_hour : nathan_letters_per_hour = 25 := by
  sorry

end nathan_writes_25_letters_per_hour_l67_6749


namespace gym_time_zero_l67_6714

/-- Represents the exercise plan with yoga and exercise components -/
structure ExercisePlan where
  yoga_time : ℕ
  exercise_time : ℕ
  bike_time : ℕ
  gym_time : ℕ
  yoga_exercise_ratio : yoga_time * 3 = exercise_time * 2
  exercise_components : exercise_time = bike_time + gym_time

/-- 
Given an exercise plan where the bike riding time equals the total exercise time,
prove that the gym workout time is zero.
-/
theorem gym_time_zero (plan : ExercisePlan) 
  (h : plan.bike_time = plan.exercise_time) : plan.gym_time = 0 := by
  sorry

end gym_time_zero_l67_6714


namespace prob_divisible_by_eight_l67_6742

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def die_sides : ℕ := 6

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1 / 2

/-- The probability of rolling a 2 on a single die -/
def prob_two : ℚ := 1 / 6

/-- The probability of rolling a 4 on a single die -/
def prob_four : ℚ := 1 / 6

/-- The probability that the product of the rolls is divisible by 8 -/
theorem prob_divisible_by_eight : 
  (1 : ℚ) - (prob_odd ^ num_dice + 
    (num_dice.choose 1 : ℚ) * prob_two * prob_odd ^ (num_dice - 1) +
    (num_dice.choose 2 : ℚ) * prob_two ^ 2 * prob_odd ^ (num_dice - 2) +
    (num_dice.choose 1 : ℚ) * prob_four * prob_odd ^ (num_dice - 1)) = 65 / 72 := by
  sorry


end prob_divisible_by_eight_l67_6742


namespace binary_101_to_decimal_l67_6712

def binary_to_decimal (b₂ : ℕ) (b₁ : ℕ) (b₀ : ℕ) : ℕ :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_101_to_decimal :
  binary_to_decimal 1 0 1 = 5 := by
  sorry

end binary_101_to_decimal_l67_6712


namespace unique_solution_unique_solution_l67_6796

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, 3, k}
def B (a : ℕ) : Set ℕ := {4, 7, a^4, a^2 + 3*a}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- Theorem statement
theorem unique_solution (a k : ℕ) :
  (∀ x ∈ A k, ∃ y ∈ B a, f x = y) ∧ 
  (∀ y ∈ B a, ∃ x ∈ A k, f x = y) →
  a = 2 ∧ k = 5 := by
  sorry

-- Alternative theorem statement if the above doesn't compile
theorem unique_solution' (a k : ℕ) :
  (∀ x, x ∈ A k → ∃ y ∈ B a, f x = y) ∧ 
  (∀ y, y ∈ B a → ∃ x ∈ A k, f x = y) →
  a = 2 ∧ k = 5 := by
  sorry

end unique_solution_unique_solution_l67_6796


namespace tom_read_six_books_in_june_l67_6775

/-- The number of books Tom read in May -/
def books_may : ℕ := 2

/-- The number of books Tom read in July -/
def books_july : ℕ := 10

/-- The total number of books Tom read -/
def total_books : ℕ := 18

/-- The number of books Tom read in June -/
def books_june : ℕ := total_books - (books_may + books_july)

theorem tom_read_six_books_in_june : books_june = 6 := by
  sorry

end tom_read_six_books_in_june_l67_6775


namespace triangle_properties_l67_6744

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  median_CM_eq : (2 : ℝ) * C.1 - C.2 - 5 = 0
  altitude_BH_eq : B.1 - (2 : ℝ) * B.2 - 5 = 0

/-- The theorem statement -/
theorem triangle_properties (abc : Triangle) 
  (h_A : abc.A = (5, 1)) : 
  abc.C = (4, 3) ∧ 
  (6 : ℝ) * abc.B.1 - 5 * abc.B.2 - 9 = 0 := by
  sorry

end triangle_properties_l67_6744


namespace twentieth_term_of_sequence_l67_6701

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem twentieth_term_of_sequence :
  let a₁ := 8
  let d := -3
  arithmetic_sequence a₁ d 20 = -49 := by sorry

end twentieth_term_of_sequence_l67_6701


namespace complex_solution_l67_6738

/-- Given two complex numbers a and b satisfying the equations
    2a^2 + ab + 2b^2 = 0 and a + 2b = 5, prove that both a and b are non-real. -/
theorem complex_solution (a b : ℂ) 
  (eq1 : 2 * a^2 + a * b + 2 * b^2 = 0)
  (eq2 : a + 2 * b = 5) :
  ¬(a.im = 0 ∧ b.im = 0) := by
  sorry

end complex_solution_l67_6738


namespace tangent_lines_parallel_to_4x_minus_1_l67_6752

/-- The curve function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_lines_parallel_to_4x_minus_1 :
  ∃! (a b : ℝ), 
    (∃ (x : ℝ), f' x = 4 ∧ 
      (∀ y : ℝ, y = 4 * x + a ↔ y - f x = f' x * (y - x))) ∧
    (∃ (x : ℝ), f' x = 4 ∧ 
      (∀ y : ℝ, y = 4 * x + b ↔ y - f x = f' x * (y - x))) ∧
    a ≠ b ∧ 
    ({a, b} : Set ℝ) = {-4, 0} :=
sorry

end tangent_lines_parallel_to_4x_minus_1_l67_6752


namespace safe_flight_probability_l67_6703

/-- Represents a rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.length * prism.width * prism.height

/-- Represents the problem setup -/
def problem_setup : Prop :=
  let outer_prism : RectangularPrism := { length := 5, width := 4, height := 3 }
  let inner_prism : RectangularPrism := { length := 3, width := 2, height := 1 }
  let outer_volume := volume outer_prism
  let inner_volume := volume inner_prism
  (inner_volume / outer_volume) = (1 : ℝ) / 10

/-- The main theorem to prove -/
theorem safe_flight_probability : problem_setup := by
  sorry

end safe_flight_probability_l67_6703


namespace definite_integral_sine_cosine_l67_6739

theorem definite_integral_sine_cosine : 
  ∫ x in (0)..(Real.pi / 2), (4 * Real.sin x + Real.cos x) = 5 := by
  sorry

end definite_integral_sine_cosine_l67_6739


namespace mary_always_wins_l67_6704

/-- Represents a player in the game -/
inductive Player : Type
| john : Player
| mary : Player

/-- Represents a move in the game -/
inductive Move : Type
| plus : Move
| minus : Move

/-- Represents the state of the game -/
structure GameState :=
(moves : List Move)

/-- The list of numbers in the game -/
def numbers : List Int := [-1, -2, -3, -4, -5, -6, -7, -8]

/-- Calculate the final sum based on the moves and numbers -/
def finalSum (state : GameState) : Int :=
  sorry

/-- Check if Mary wins given the final sum -/
def maryWins (sum : Int) : Prop :=
  sum = -4 ∨ sum = -2 ∨ sum = 0 ∨ sum = 2 ∨ sum = 4

/-- Mary's strategy function -/
def maryStrategy (state : GameState) : Move :=
  sorry

/-- Theorem stating that Mary always wins -/
theorem mary_always_wins :
  ∀ (game : List Move),
    game.length ≤ 8 →
    maryWins (finalSum { moves := game ++ [maryStrategy { moves := game }] }) :=
sorry

end mary_always_wins_l67_6704


namespace face_value_of_shares_l67_6779

/-- Proves that the face value of shares is 40, given the dividend rate, return on investment, and purchase price. -/
theorem face_value_of_shares (dividend_rate : ℝ) (roi_rate : ℝ) (purchase_price : ℝ) :
  dividend_rate = 0.125 →
  roi_rate = 0.25 →
  purchase_price = 20 →
  dividend_rate * (purchase_price / roi_rate) = 40 := by
  sorry

end face_value_of_shares_l67_6779


namespace sum_of_abc_l67_6722

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 13 * Real.sqrt 6 := by
  sorry

end sum_of_abc_l67_6722


namespace largest_difference_l67_6778

def P : ℕ := 3 * 1003^1004
def Q : ℕ := 1003^1004
def R : ℕ := 1002 * 1003^1003
def S : ℕ := 3 * 1003^1003
def T : ℕ := 1003^1003
def U : ℕ := 1003^1002 * Nat.factorial 1002

theorem largest_difference (P Q R S T U : ℕ) 
  (hP : P = 3 * 1003^1004)
  (hQ : Q = 1003^1004)
  (hR : R = 1002 * 1003^1003)
  (hS : S = 3 * 1003^1003)
  (hT : T = 1003^1003)
  (hU : U = 1003^1002 * Nat.factorial 1002) :
  P - Q > max (Q - R) (max (R - S) (max (S - T) (T - U))) :=
sorry

end largest_difference_l67_6778


namespace triangle_area_l67_6770

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l67_6770


namespace inequality_holds_iff_first_quadrant_l67_6760

theorem inequality_holds_iff_first_quadrant (θ : Real) :
  (∀ x : Real, x ∈ Set.Icc 0 1 →
    x^2 * Real.cos θ - 3 * x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
  θ ∈ Set.Ioo 0 (Real.pi / 2) := by
  sorry

end inequality_holds_iff_first_quadrant_l67_6760


namespace game_night_group_division_l67_6759

theorem game_night_group_division (n : ℕ) (h : n = 6) :
  Nat.choose n (n / 2) = 20 :=
by sorry

end game_night_group_division_l67_6759


namespace task_assignment_count_l67_6792

/-- The number of ways to assign 4 students to 3 tasks -/
def task_assignments : ℕ := 12

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of tasks -/
def num_tasks : ℕ := 3

/-- The number of students assigned to clean the podium -/
def podium_cleaners : ℕ := 1

/-- The number of students assigned to sweep the floor -/
def floor_sweepers : ℕ := 1

/-- The number of students assigned to mop the floor -/
def floor_moppers : ℕ := 2

theorem task_assignment_count :
  task_assignments = num_students * (num_students - 1) / 2 :=
by sorry

end task_assignment_count_l67_6792


namespace simplify_cube_root_l67_6711

theorem simplify_cube_root (a b c : ℝ) : ∃ x y z w : ℝ, 
  (54 * a^5 * b^9 * c^14)^(1/3) = x * a^y * b^z * c^w ∧ y + z + w = 8 := by
  sorry

end simplify_cube_root_l67_6711


namespace least_common_period_l67_6733

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) + f (x - 3) = f x

-- Define what it means for a function to have a period
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (∃ p : ℝ, p > 0 ∧ HasPeriod f p) →
    (∀ q : ℝ, q > 0 ∧ HasPeriod f q → q ≥ 18) ∧
    HasPeriod f 18 :=
sorry

end least_common_period_l67_6733


namespace average_time_theorem_l67_6707

def relay_race (y z w : ℝ) : Prop :=
  y = 58 ∧ z = 26 ∧ w = 2*z

theorem average_time_theorem (y z w : ℝ) (h : relay_race y z w) :
  (y + z + w) / 3 = (58 + 26 + 2*26) / 3 := by sorry

end average_time_theorem_l67_6707


namespace order_of_four_numbers_l67_6763

theorem order_of_four_numbers (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by sorry

end order_of_four_numbers_l67_6763


namespace perpendicular_line_equation_l67_6768

/-- A line in the xy-plane can be represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are perpendicular if the product of their slopes is -1. -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_line_equation (l : Line) (h1 : l.y_intercept = 1) 
    (h2 : perpendicular l (Line.mk (1/2) 0)) : 
  l.slope = -2 ∧ ∀ x y : ℝ, y = l.slope * x + l.y_intercept ↔ y = -2 * x + 1 := by
  sorry

end perpendicular_line_equation_l67_6768


namespace smallest_largest_product_l67_6780

def digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_three_digit (n : Nat) : Prop := 100 ≤ n ∧ n ≤ 999

def uses_all_digits (a b c : Nat) : Prop :=
  (digits.card = 9) ∧
  (Finset.card (Finset.image (λ d => d % 10) {a, b, c, a / 10, b / 10, c / 10, a / 100, b / 100, c / 100}) = 9)

theorem smallest_largest_product :
  ∀ a b c : Nat,
  is_three_digit a ∧ is_three_digit b ∧ is_three_digit c →
  uses_all_digits a b c →
  (∀ x y z : Nat, is_three_digit x ∧ is_three_digit y ∧ is_three_digit z → uses_all_digits x y z → a * b * c ≤ x * y * z) ∧
  (∀ x y z : Nat, is_three_digit x ∧ is_three_digit y ∧ is_three_digit z → uses_all_digits x y z → x * y * z ≤ 941 * 852 * 763) :=
by sorry

end smallest_largest_product_l67_6780


namespace suv_max_distance_l67_6758

/-- Calculates the maximum distance an SUV can travel given its fuel efficiencies and available fuel -/
theorem suv_max_distance (highway_mpg city_mpg mountain_mpg : ℝ) (fuel : ℝ) : 
  highway_mpg = 12.2 →
  city_mpg = 7.6 →
  mountain_mpg = 9.4 →
  fuel = 22 →
  (highway_mpg + city_mpg + mountain_mpg) * fuel = 642.4 := by
  sorry

end suv_max_distance_l67_6758


namespace inequality_proof_largest_constant_l67_6771

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≤ (Real.sqrt 6 / 2) * Real.sqrt (x + y + z) :=
sorry

theorem largest_constant :
  ∀ k, (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≤ k * Real.sqrt (x + y + z)) →
  k ≤ Real.sqrt 6 / 2 :=
sorry

end inequality_proof_largest_constant_l67_6771


namespace sum_equals_thirty_l67_6726

theorem sum_equals_thirty : 1 + 2 + 3 - 4 + 5 + 6 + 7 - 8 + 9 + 10 + 11 - 12 = 30 := by
  sorry

end sum_equals_thirty_l67_6726


namespace outfits_count_l67_6776

/-- The number of different outfits that can be created given a specific number of shirts, pants, and ties. --/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that with 8 shirts, 5 pants, and 6 ties, the number of possible outfits is 280. --/
theorem outfits_count : number_of_outfits 8 5 6 = 280 := by
  sorry

end outfits_count_l67_6776
