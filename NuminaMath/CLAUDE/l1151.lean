import Mathlib

namespace rational_triplet_problem_l1151_115142

theorem rational_triplet_problem (m n p : ℚ) : 
  m > 0 ∧ n > 0 ∧ p > 0 →
  (∃ (a b c : ℤ), m + 1 / (n * p) = a ∧ n + 1 / (p * m) = b ∧ p + 1 / (m * n) = c) →
  ((m = 1/2 ∧ n = 1/2 ∧ p = 4) ∨ 
   (m = 1/2 ∧ n = 1 ∧ p = 2) ∨ 
   (m = 1 ∧ n = 1 ∧ p = 1) ∨
   (m = 1/2 ∧ n = 4 ∧ p = 1/2) ∨
   (m = 1 ∧ n = 2 ∧ p = 1/2) ∨
   (m = 4 ∧ n = 1/2 ∧ p = 1/2) ∨
   (m = 2 ∧ n = 1/2 ∧ p = 1) ∨
   (m = 2 ∧ n = 1 ∧ p = 1/2) ∨
   (m = 1/2 ∧ n = 2 ∧ p = 1)) :=
by sorry

end rational_triplet_problem_l1151_115142


namespace five_cubic_yards_to_cubic_feet_l1151_115193

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Volume in cubic feet for 5 cubic yards -/
def volume_cubic_feet : ℝ := 135

/-- Theorem stating that 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_to_cubic_feet :
  (5 : ℝ) * yards_to_feet^3 = volume_cubic_feet := by sorry

end five_cubic_yards_to_cubic_feet_l1151_115193


namespace sculpture_height_l1151_115120

theorem sculpture_height (base_height : ℝ) (total_height_feet : ℝ) (h1 : base_height = 10) (h2 : total_height_feet = 3.6666666666666665) : 
  total_height_feet * 12 - base_height = 34 := by
sorry

end sculpture_height_l1151_115120


namespace sandy_payment_l1151_115154

/-- Represents the cost and quantity of a coffee shop item -/
structure Item where
  price : ℚ
  quantity : ℕ

/-- Calculates the total cost of an order -/
def orderTotal (items : List Item) : ℚ :=
  items.foldl (fun acc item => acc + item.price * item.quantity) 0

/-- Proves that Sandy paid $20 given the order details and change received -/
theorem sandy_payment (cappuccino iced_tea cafe_latte espresso : Item)
    (change : ℚ) :
    cappuccino.price = 2 →
    iced_tea.price = 3 →
    cafe_latte.price = 3/2 →
    espresso.price = 1 →
    cappuccino.quantity = 3 →
    iced_tea.quantity = 2 →
    cafe_latte.quantity = 2 →
    espresso.quantity = 2 →
    change = 3 →
    orderTotal [cappuccino, iced_tea, cafe_latte, espresso] + change = 20 := by
  sorry


end sandy_payment_l1151_115154


namespace dennis_teaching_years_l1151_115175

theorem dennis_teaching_years 
  (V A D E N : ℕ) -- Years taught by Virginia, Adrienne, Dennis, Elijah, and Nadine
  (h1 : V + A + D + E + N = 225) -- Total years taught
  (h2 : (V + A + D + E + N) * 5 = (V + A + D + E + N + 150) * 3) -- Total years is 3/5 of age sum
  (h3 : V = A + 9) -- Virginia vs Adrienne
  (h4 : V = D - 15) -- Virginia vs Dennis
  (h5 : E = A - 3) -- Elijah vs Adrienne
  (h6 : E = 2 * N) -- Elijah vs Nadine
  : D = 101 := by
  sorry

end dennis_teaching_years_l1151_115175


namespace remainder_of_12345678_div_10_l1151_115143

theorem remainder_of_12345678_div_10 :
  ∃ q : ℕ, 12345678 = 10 * q + 8 ∧ 8 < 10 := by sorry

end remainder_of_12345678_div_10_l1151_115143


namespace problem_statements_l1151_115116

theorem problem_statements :
  (∀ (p q : Prop), (p ∧ q) → ¬(¬p)) ∧
  (∃ (x : ℝ), x^2 - x - 1 < 0) ↔ ¬(∀ (x : ℝ), x^2 - x - 1 ≥ 0) ∧
  (∃ (a b : ℝ), (a + b > 0) ∧ ¬(a > 5 ∧ b > -5)) ∧
  (∀ (α : ℝ), α < 0 → ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ < x₂ → x₁^α > x₂^α) :=
by sorry

end problem_statements_l1151_115116


namespace eggs_needed_for_scaled_cake_l1151_115192

/-- Represents the recipe for sponge cake -/
structure Recipe where
  eggs : ℝ
  flour : ℝ
  sugar : ℝ

/-- Calculates the total mass of the cake from a recipe -/
def totalMass (r : Recipe) : ℝ := r.eggs + r.flour + r.sugar

/-- The original recipe -/
def originalRecipe : Recipe := { eggs := 300, flour := 120, sugar := 100 }

/-- Theorem: The amount of eggs needed for 2600g of sponge cake is 1500g -/
theorem eggs_needed_for_scaled_cake (desiredMass : ℝ) 
  (h : desiredMass = 2600) : 
  (originalRecipe.eggs / totalMass originalRecipe) * desiredMass = 1500 := by
  sorry

end eggs_needed_for_scaled_cake_l1151_115192


namespace problem1_problem2_problem3_problem4_l1151_115156

-- 1. Prove that 9999×2222+3333×3334 = 33330000
theorem problem1 : 9999 * 2222 + 3333 * 3334 = 33330000 := by sorry

-- 2. Prove that 96%×25+0.75+0.25 = 25
theorem problem2 : (96 / 100) * 25 + 0.75 + 0.25 = 25 := by sorry

-- 3. Prove that 5/8 + 7/10 + 3/8 + 3/10 = 2
theorem problem3 : 5/8 + 7/10 + 3/8 + 3/10 = 2 := by sorry

-- 4. Prove that 3.7 × 6/5 - 2.2 ÷ 5/6 = 1.8
theorem problem4 : 3.7 * (6/5) - 2.2 / (5/6) = 1.8 := by sorry

end problem1_problem2_problem3_problem4_l1151_115156


namespace base6_two_distinct_primes_l1151_115132

/-- Represents a number in base 6 formed by appending fives to 1200 -/
def base6Number (n : ℕ) : ℕ :=
  288 * 6^(10*n + 2) + (6^(10*n + 2) - 1)

/-- Counts the number of distinct prime factors of a natural number -/
noncomputable def countDistinctPrimeFactors (x : ℕ) : ℕ := sorry

/-- Theorem stating that the base 6 number has exactly two distinct prime factors iff n = 0 -/
theorem base6_two_distinct_primes (n : ℕ) : 
  countDistinctPrimeFactors (base6Number n) = 2 ↔ n = 0 := by sorry

end base6_two_distinct_primes_l1151_115132


namespace reflect_c_twice_l1151_115134

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Theorem: Reflecting point C(2,2) over x-axis then y-axis results in C''(-2,-2) -/
theorem reflect_c_twice :
  let c : ℝ × ℝ := (2, 2)
  reflect_y (reflect_x c) = (-2, -2) := by
sorry

end reflect_c_twice_l1151_115134


namespace parabola_intercepts_sum_l1151_115106

theorem parabola_intercepts_sum (d e f : ℝ) : 
  (∀ x, 3 * x^2 - 9 * x + 5 = 3 * 0^2 - 9 * 0 + 5 → d = 3 * 0^2 - 9 * 0 + 5) →
  (3 * e^2 - 9 * e + 5 = 0) →
  (3 * f^2 - 9 * f + 5 = 0) →
  d + e + f = 8 := by
sorry

end parabola_intercepts_sum_l1151_115106


namespace simplify_expression_l1151_115110

theorem simplify_expression (x : ℝ) : 120 * x - 55 * x = 65 * x := by
  sorry

end simplify_expression_l1151_115110


namespace five_solutions_l1151_115123

/-- The system of equations has exactly 5 real solutions -/
theorem five_solutions (x y z w θ : ℝ) : 
  x = 2*z + 2*w + z*w*x →
  y = 2*w + 2*x + w*x*y →
  z = 2*x + 2*y + x*y*z →
  w = 2*y + 2*z + y*z*w →
  w = Real.sin θ ^ 2 →
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), s.card = 5 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔ 
    (a = 2*c + 2*d + c*d*a ∧
     b = 2*d + 2*a + d*a*b ∧
     c = 2*a + 2*b + a*b*c ∧
     d = 2*b + 2*c + b*c*d ∧
     d = Real.sin θ ^ 2) :=
by sorry


end five_solutions_l1151_115123


namespace evaluate_expression_l1151_115139

theorem evaluate_expression : Real.sqrt 5 * 5^(1/2 : ℝ) + 20 / 4 * 3 - 9^(3/2 : ℝ) = -7 := by
  sorry

end evaluate_expression_l1151_115139


namespace complex_equation_solution_l1151_115191

theorem complex_equation_solution (z : ℂ) : Complex.I * (z - 1) = 1 + Complex.I * Complex.I → z = 2 - Complex.I := by
  sorry

end complex_equation_solution_l1151_115191


namespace total_weight_in_kg_l1151_115153

-- Define the weights in grams
def monosodium_glutamate : ℕ := 80
def salt : ℕ := 500
def laundry_detergent : ℕ := 420

-- Define the conversion factor from grams to kilograms
def grams_per_kg : ℕ := 1000

-- Theorem statement
theorem total_weight_in_kg :
  (monosodium_glutamate + salt + laundry_detergent) / grams_per_kg = 1 := by
  sorry

end total_weight_in_kg_l1151_115153


namespace gummy_worm_fraction_l1151_115144

theorem gummy_worm_fraction (initial_count : ℕ) (days : ℕ) (final_count : ℕ) (f : ℚ) :
  initial_count = 64 →
  days = 4 →
  final_count = 4 →
  0 < f →
  f < 1 →
  (1 - f) ^ days * initial_count = final_count →
  f = 1/2 := by
sorry

end gummy_worm_fraction_l1151_115144


namespace paperclips_exceed_250_l1151_115128

def paperclips (n : ℕ) : ℕ := 5 * 2^(n - 1)

theorem paperclips_exceed_250 : 
  ∀ k : ℕ, k < 7 → paperclips k ≤ 250 ∧ paperclips 7 > 250 :=
by sorry

end paperclips_exceed_250_l1151_115128


namespace consecutive_page_numbers_l1151_115171

theorem consecutive_page_numbers (n : ℕ) : 
  n * (n + 1) * (n + 2) = 35280 → n + (n + 1) + (n + 2) = 96 := by
  sorry

end consecutive_page_numbers_l1151_115171


namespace angle_triple_complement_l1151_115170

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end angle_triple_complement_l1151_115170


namespace perfect_square_condition_l1151_115107

theorem perfect_square_condition (x : ℤ) : 
  (∃ y : ℤ, x^2 + 19*x + 95 = y^2) ↔ (x = -14 ∨ x = -5) := by
sorry

end perfect_square_condition_l1151_115107


namespace parabola_intersection_comparison_l1151_115112

theorem parabola_intersection_comparison (m n a b : ℝ) : 
  (∀ x, m * x^2 + x ≥ 0 → x ≤ a) →  -- A(a,0) is the rightmost intersection of y = mx^2 + x with x-axis
  (∀ x, n * x^2 + x ≥ 0 → x ≤ b) →  -- B(b,0) is the rightmost intersection of y = nx^2 + x with x-axis
  m * a^2 + a = 0 →                  -- A(a,0) is on the parabola y = mx^2 + x
  n * b^2 + b = 0 →                  -- B(b,0) is on the parabola y = nx^2 + x
  a > b →                            -- A is to the right of B
  a > 0 →                            -- A is in the positive half of x-axis
  b > 0 →                            -- B is in the positive half of x-axis
  m > n :=                           -- Conclusion: m > n
by sorry

end parabola_intersection_comparison_l1151_115112


namespace planar_edge_pairs_4_2_3_l1151_115149

/-- A rectangular prism with edge dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The number of unordered pairs of edges that determine a plane in a rectangular prism. -/
def planarEdgePairs (prism : RectangularPrism) : ℕ :=
  sorry

/-- Theorem: The number of unordered pairs of edges that determine a plane
    in a rectangular prism with edge dimensions 4, 2, and 3 is equal to 42. -/
theorem planar_edge_pairs_4_2_3 :
  planarEdgePairs { a := 4, b := 2, c := 3 } = 42 := by
  sorry

end planar_edge_pairs_4_2_3_l1151_115149


namespace remainder_problem_l1151_115117

theorem remainder_problem : (56 * 67 * 78) % 15 = 6 := by
  sorry

end remainder_problem_l1151_115117


namespace sqrt_equality_implies_t_value_l1151_115118

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → t = 37/10 := by
  sorry

end sqrt_equality_implies_t_value_l1151_115118


namespace smallest_result_is_16_l1151_115135

def S : Finset Nat := {2, 3, 5, 7, 11, 13}

theorem smallest_result_is_16 :
  ∃ (a b c : Nat), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b) * c = 16 ∧
  ∀ (x y z : Nat), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  (x + y) * z ≥ 16 := by
sorry

end smallest_result_is_16_l1151_115135


namespace max_value_of_a_l1151_115159

theorem max_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 :=
sorry

end max_value_of_a_l1151_115159


namespace max_product_sum_2024_l1151_115198

theorem max_product_sum_2024 : 
  ∃ (x : ℤ), x * (2024 - x) = 1024144 ∧ 
  ∀ (y : ℤ), y * (2024 - y) ≤ 1024144 := by
  sorry

end max_product_sum_2024_l1151_115198


namespace triangle_sinC_l1151_115162

theorem triangle_sinC (A B C : ℝ) (hABC : A + B + C = π) 
  (hAC : 3 = 2 * Real.sqrt 3 * (Real.sin A / Real.sin B))
  (hA : A = 2 * B) : 
  Real.sin C = Real.sqrt 6 / 9 := by
sorry

end triangle_sinC_l1151_115162


namespace m_range_if_f_increasing_l1151_115157

/-- Piecewise function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^2 + 2*m*x - 2 else 1 + Real.log x

/-- Theorem stating that if f is increasing, then m is in [1, 2] -/
theorem m_range_if_f_increasing (m : ℝ) :
  (∀ x y, x < y → f m x < f m y) → m ∈ Set.Icc 1 2 :=
by sorry

end m_range_if_f_increasing_l1151_115157


namespace multiplication_table_odd_fraction_l1151_115180

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let total_products : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 := by
sorry

end multiplication_table_odd_fraction_l1151_115180


namespace range_of_m_l1151_115186

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 2*x^2 - 9*x + m < 0) → 
  m < 9 := by
sorry

end range_of_m_l1151_115186


namespace initial_female_percentage_calculation_l1151_115145

/-- Represents a company's workforce statistics -/
structure Company where
  initial_employees : ℕ
  initial_female_percentage : ℚ
  hired_male_workers : ℕ
  final_employees : ℕ
  final_female_percentage : ℚ

/-- Theorem stating the relationship between initial and final workforce statistics -/
theorem initial_female_percentage_calculation (c : Company) 
  (h1 : c.hired_male_workers = 20)
  (h2 : c.final_employees = 240)
  (h3 : c.final_female_percentage = 55/100)
  (h4 : c.initial_employees + c.hired_male_workers = c.final_employees)
  (h5 : c.initial_employees * c.initial_female_percentage = 
        c.final_employees * c.final_female_percentage) :
  c.initial_female_percentage = 60/100 := by
  sorry

end initial_female_percentage_calculation_l1151_115145


namespace fraction_equals_zero_l1151_115178

theorem fraction_equals_zero (x : ℝ) (h : x + 1 ≠ 0) :
  x = 1 → (x^2 - 1) / (x + 1) = 0 := by
  sorry

end fraction_equals_zero_l1151_115178


namespace jelly_bean_division_l1151_115190

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) 
  (h1 : initial_amount = 36)
  (h2 : eaten_amount = 6)
  (h3 : num_piles = 3)
  (h4 : initial_amount > eaten_amount) :
  (initial_amount - eaten_amount) / num_piles = 10 := by
  sorry

end jelly_bean_division_l1151_115190


namespace gcd_triple_characterization_l1151_115165

theorem gcd_triple_characterization (a b c : ℕ+) :
  Nat.gcd a.val 20 = b.val ∧
  Nat.gcd b.val 15 = c.val ∧
  Nat.gcd a.val c.val = 5 →
  ∃ k : ℕ+, (a = 5 * k ∧ b = 5 ∧ c = 5) ∨
            (a = 5 * k ∧ b = 10 ∧ c = 5) ∨
            (a = 5 * k ∧ b = 20 ∧ c = 5) := by
  sorry


end gcd_triple_characterization_l1151_115165


namespace intersection_implies_m_value_l1151_115181

def M (m : ℤ) : Set ℤ := {m, -3}

def N : Set ℤ := {x : ℤ | 2*x^2 + 7*x + 3 < 0}

theorem intersection_implies_m_value (m : ℤ) :
  (M m ∩ N).Nonempty → m = -2 ∨ m = -1 := by
  sorry

end intersection_implies_m_value_l1151_115181


namespace cricket_team_average_age_l1151_115189

theorem cricket_team_average_age (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) :
  team_size = 11 →
  captain_age = 25 →
  wicket_keeper_age_diff = 5 →
  let total_age := team_size * (captain_age + 7)
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + (captain_age + wicket_keeper_age_diff))
  (remaining_age / remaining_players) + 1 = total_age / team_size →
  total_age / team_size = 32 := by
  sorry

end cricket_team_average_age_l1151_115189


namespace sector_to_cone_l1151_115166

/-- Proves that a 270° sector of a circle with radius 12 forms a cone with base radius 9 and slant height 12 -/
theorem sector_to_cone (sector_angle : Real) (circle_radius : Real) 
  (h1 : sector_angle = 270)
  (h2 : circle_radius = 12) : 
  let base_radius := (sector_angle / 360) * (2 * Real.pi * circle_radius) / (2 * Real.pi)
  let slant_height := circle_radius
  (base_radius = 9 ∧ slant_height = 12) := by
  sorry

end sector_to_cone_l1151_115166


namespace sphere_division_l1151_115148

theorem sphere_division (R : ℝ) : 
  (∃ (n : ℕ), n = 216 ∧ (4 / 3 * Real.pi * R^3 = n * (4 / 3 * Real.pi * 1^3))) ↔ R = 6 :=
sorry

end sphere_division_l1151_115148


namespace square_area_ratio_l1151_115182

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 / b^2 = 16 := by
  sorry

end square_area_ratio_l1151_115182


namespace complete_factorization_w4_minus_81_l1151_115150

theorem complete_factorization_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by
  sorry

end complete_factorization_w4_minus_81_l1151_115150


namespace circle_a_range_l1151_115173

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0

-- Define the condition for the equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y a

-- Theorem stating the range of a for which the equation represents a circle
theorem circle_a_range :
  ∀ a : ℝ, is_circle a ↔ -2 < a ∧ a < 2/3 :=
sorry

end circle_a_range_l1151_115173


namespace largest_solution_quadratic_l1151_115151

theorem largest_solution_quadratic (x : ℝ) : 
  (9 * x^2 - 51 * x + 70 = 0) → x ≤ 70/9 :=
by sorry

end largest_solution_quadratic_l1151_115151


namespace greatest_q_minus_r_l1151_115115

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  975 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 975 = 23 * q' + r' → (q - r : ℤ) ≥ (q' - r' : ℤ) ∧
  (q - r : ℤ) = 33 :=
sorry

end greatest_q_minus_r_l1151_115115


namespace connect_four_games_total_l1151_115121

/-- Given that Kaleb's ratio of won to lost games is 3:2 and he won 18 games,
    prove that the total number of games played is 30. -/
theorem connect_four_games_total (won lost total : ℕ) : 
  won = 18 → 
  3 * lost = 2 * won → 
  total = won + lost → 
  total = 30 := by
sorry

end connect_four_games_total_l1151_115121


namespace l_shape_perimeter_l1151_115158

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the L-shape configuration -/
structure LShape where
  vertical : Rectangle
  horizontal : Rectangle
  overlap : ℝ

/-- Calculates the perimeter of the L-shape -/
def LShape.perimeter (l : LShape) : ℝ :=
  l.vertical.perimeter + l.horizontal.perimeter - 2 * l.overlap

theorem l_shape_perimeter :
  let l : LShape := {
    vertical := { width := 3, height := 6 },
    horizontal := { width := 4, height := 2 },
    overlap := 1
  }
  l.perimeter = 28 := by sorry

end l_shape_perimeter_l1151_115158


namespace absolute_difference_of_opposite_signs_l1151_115114

theorem absolute_difference_of_opposite_signs (m n : ℤ) : 
  (abs m = 5) → (abs n = 2) → (m * n < 0) → abs (m - n) = 7 := by
  sorry

end absolute_difference_of_opposite_signs_l1151_115114


namespace min_n_for_equation_property_l1151_115108

theorem min_n_for_equation_property : ∃ (n : ℕ), n = 835 ∧ 
  (∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 999) →
    ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + 2*b + 3*c = d) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T.card = m ∧ (∀ x ∈ T, x ≥ 1 ∧ x ≤ 999) ∧
    ¬(∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + 2*b + 3*c = d)) :=
by sorry

end min_n_for_equation_property_l1151_115108


namespace vector_decomposition_l1151_115130

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![2, -1, 11]
def p : Fin 3 → ℝ := ![1, 1, 0]
def q : Fin 3 → ℝ := ![0, 1, -2]
def r : Fin 3 → ℝ := ![1, 0, 3]

/-- Theorem stating that x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (-3 : ℝ) • p + 2 • q + 5 • r := by
  sorry

end vector_decomposition_l1151_115130


namespace sin_45_degrees_l1151_115125

theorem sin_45_degrees :
  Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end sin_45_degrees_l1151_115125


namespace or_necessary_not_sufficient_for_and_l1151_115197

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q) :=
by sorry

end or_necessary_not_sufficient_for_and_l1151_115197


namespace max_value_of_f_l1151_115136

-- Define the function
def f (x : ℝ) : ℝ := x * abs x - 2 * x + 1

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 16 ∧ ∀ x : ℝ, |x + 1| ≤ 6 → f x ≤ M :=
sorry

end max_value_of_f_l1151_115136


namespace fish_tournament_ratio_l1151_115179

def fish_tournament (jacob_initial : ℕ) (alex_lost : ℕ) (jacob_needed : ℕ) : Prop :=
  ∃ (alex_initial : ℕ) (n : ℕ),
    alex_initial = n * jacob_initial ∧
    jacob_initial + jacob_needed = (alex_initial - alex_lost) + 1 ∧
    alex_initial / jacob_initial = 7

theorem fish_tournament_ratio :
  fish_tournament 8 23 26 := by
  sorry

end fish_tournament_ratio_l1151_115179


namespace no_solution_exists_l1151_115101

theorem no_solution_exists (x y : ℕ) (h : x > 1) : (x^7 - 1) / (x - 1) ≠ y^5 + 1 := by
  sorry

end no_solution_exists_l1151_115101


namespace expression_evaluation_l1151_115195

theorem expression_evaluation :
  let a : ℤ := 1
  let b : ℤ := 10
  let c : ℤ := 100
  let d : ℤ := 1000
  (a + b + c - d) + (a + b - c + d) + (a - b + c + d) + (-a + b + c + d) = 2222 :=
by sorry

end expression_evaluation_l1151_115195


namespace cloth_selling_price_l1151_115119

/-- Calculates the total selling price of cloth given its length, cost price per meter, and profit per meter. -/
def total_selling_price (length : ℝ) (cost_price_per_meter : ℝ) (profit_per_meter : ℝ) : ℝ :=
  length * (cost_price_per_meter + profit_per_meter)

/-- The total selling price of 78 meters of cloth with a cost price of Rs. 58.02564102564102 per meter
    and a profit of Rs. 29 per meter is approximately Rs. 6788.00. -/
theorem cloth_selling_price :
  let length : ℝ := 78
  let cost_price_per_meter : ℝ := 58.02564102564102
  let profit_per_meter : ℝ := 29
  abs (total_selling_price length cost_price_per_meter profit_per_meter - 6788) < 0.01 := by
sorry

end cloth_selling_price_l1151_115119


namespace angle_problem_l1151_115131

theorem angle_problem (α β : Real) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h_distance : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = Real.sqrt 10 / 5)
  (h_tan : Real.tan (α/2) = 1/2) :
  (Real.cos (α - β) = 4/5) ∧
  (Real.cos α = 3/5) ∧
  (Real.cos β = 24/25) := by
sorry

end angle_problem_l1151_115131


namespace geometric_sequence_sum_l1151_115169

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_1_2 : a 1 + a 2 = 30
  sum_3_4 : a 3 + a 4 = 60

/-- The theorem stating that a_7 + a_8 = 240 for the given geometric sequence -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 7 + seq.a 8 = 240 := by
  sorry

end geometric_sequence_sum_l1151_115169


namespace sqrt_144_div_6_l1151_115163

theorem sqrt_144_div_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end sqrt_144_div_6_l1151_115163


namespace tangent_product_special_angles_l1151_115133

theorem tangent_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 40 * π / 180
  let C : Real := 5 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) * (1 + Real.tan C) = 3 + Real.sqrt 3 := by
  sorry

end tangent_product_special_angles_l1151_115133


namespace factors_of_180_l1151_115105

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_180 : number_of_factors 180 = 18 := by
  sorry

end factors_of_180_l1151_115105


namespace consecutive_lucky_years_exist_l1151_115113

def is_lucky_year (n : ℕ) : Prop :=
  let a := n / 100
  let b := n % 100
  n % (a + b) = 0

theorem consecutive_lucky_years_exist : ∃ n : ℕ, is_lucky_year n ∧ is_lucky_year (n + 1) :=
sorry

end consecutive_lucky_years_exist_l1151_115113


namespace square_difference_l1151_115127

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) : 
  (x - y)^2 = 17 := by
  sorry

end square_difference_l1151_115127


namespace smallest_valid_number_l1151_115126

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    n = 1000 + 100 * a + b ∧
    n = (10 * a + b) ^ 2 ∧
    0 ≤ a ∧ a ≤ 99 ∧
    0 ≤ b ∧ b ≤ 99

theorem smallest_valid_number :
  is_valid_number 2025 ∧ ∀ n, is_valid_number n → n ≥ 2025 :=
sorry

end smallest_valid_number_l1151_115126


namespace ladder_construction_possible_l1151_115140

/-- Represents the ladder construction problem --/
def ladder_problem (total_wood rung_length rung_spacing side_support_length climbing_height : ℝ) : Prop :=
  let num_rungs : ℝ := climbing_height / rung_spacing + 1
  let wood_for_rungs : ℝ := num_rungs * rung_length
  let wood_for_supports : ℝ := 2 * side_support_length
  let total_wood_needed : ℝ := wood_for_rungs + wood_for_supports
  let leftover_wood : ℝ := total_wood - total_wood_needed
  total_wood_needed ≤ total_wood ∧ leftover_wood = 36.5

/-- Theorem stating that the ladder can be built with the given conditions --/
theorem ladder_construction_possible : 
  ladder_problem 300 1.5 0.5 56 50 := by
  sorry

#check ladder_construction_possible

end ladder_construction_possible_l1151_115140


namespace ab_gt_ac_not_sufficient_nor_necessary_for_b_gt_c_l1151_115100

theorem ab_gt_ac_not_sufficient_nor_necessary_for_b_gt_c :
  ¬(∀ a b c : ℝ, a * b > a * c → b > c) ∧
  ¬(∀ a b c : ℝ, b > c → a * b > a * c) := by
  sorry

end ab_gt_ac_not_sufficient_nor_necessary_for_b_gt_c_l1151_115100


namespace sum_of_squares_l1151_115104

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 51)
  (h2 : x * x * y + x * y * y = 560) :
  x * x + y * y = 186 := by
sorry

end sum_of_squares_l1151_115104


namespace count_decompositions_l1151_115102

/-- The number of ways to write 4020 in the specified form -/
def M : ℕ := 40000

/-- A function that represents the decomposition of 4020 -/
def decomposition (b₃ b₂ b₁ b₀ : ℕ) : ℕ :=
  b₃ * 1000 + b₂ * 100 + b₁ * 10 + b₀

/-- The theorem stating that M is the correct count -/
theorem count_decompositions :
  M = (Finset.filter (fun (b : ℕ × ℕ × ℕ × ℕ) => 
    let (b₃, b₂, b₁, b₀) := b
    decomposition b₃ b₂ b₁ b₀ = 4020 ∧ 
    b₃ ≤ 99 ∧ b₂ ≤ 99 ∧ b₁ ≤ 99 ∧ b₀ ≤ 99)
    (Finset.product (Finset.range 100) 
      (Finset.product (Finset.range 100) 
        (Finset.product (Finset.range 100) (Finset.range 100))))).card :=
by
  sorry

end count_decompositions_l1151_115102


namespace horner_method_polynomial_evaluation_l1151_115161

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def polynomial (x : ℝ) : ℝ :=
  3 * x^4 - x^2 + 2 * x + 1

theorem horner_method_polynomial_evaluation :
  let coeffs := [3, 0, -1, 2, 1]
  let x := 2
  let v₃ := (horner_method (coeffs.take 4) x)
  v₃ = 22 :=
by sorry

end horner_method_polynomial_evaluation_l1151_115161


namespace valid_parameterization_l1151_115103

/-- A vector parameterization of a line --/
structure VectorParam where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if a vector parameterization is valid for the line y = 3x + 4 --/
def is_valid_param (p : VectorParam) : Prop :=
  p.b = 3 * p.a + 4 ∧ p.d = 3 * p.c

theorem valid_parameterization (p : VectorParam) :
  is_valid_param p ↔
    (∀ t : ℝ, (p.a + t * p.c, p.b + t * p.d) ∈ {(x, y) : ℝ × ℝ | y = 3 * x + 4}) :=
by sorry

end valid_parameterization_l1151_115103


namespace sticker_distribution_l1151_115183

/-- The number of ways to distribute n indistinguishable objects among k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

/-- Theorem stating that there are 935 ways to distribute 10 stickers among 5 sheets -/
theorem sticker_distribution : distribute num_stickers num_sheets = 935 := by sorry

end sticker_distribution_l1151_115183


namespace missing_mark_calculation_l1151_115138

def calculate_missing_mark (english math physics chemistry average : ℕ) : ℕ :=
  5 * average - (english + math + physics + chemistry)

theorem missing_mark_calculation (english math physics chemistry average biology : ℕ)
  (h1 : english = 76)
  (h2 : math = 65)
  (h3 : physics = 82)
  (h4 : chemistry = 67)
  (h5 : average = 73)
  (h6 : biology = calculate_missing_mark english math physics chemistry average) :
  biology = 75 := by
  sorry

end missing_mark_calculation_l1151_115138


namespace carl_index_cards_cost_l1151_115122

/-- Calculates the total cost of index cards for Carl's classes -/
def total_cost (cards_per_student : ℕ) (periods : ℕ) (students_per_class : ℕ) (cards_per_pack : ℕ) (cost_per_pack : ℕ) : ℕ :=
  let total_students := periods * students_per_class
  let total_cards := total_students * cards_per_student
  let packs_needed := (total_cards + cards_per_pack - 1) / cards_per_pack  -- Ceiling division
  packs_needed * cost_per_pack

/-- Proves that the total cost of index cards for Carl's classes is $108 -/
theorem carl_index_cards_cost : 
  total_cost 10 6 30 50 3 = 108 := by
  sorry

end carl_index_cards_cost_l1151_115122


namespace square_sum_given_condition_l1151_115111

theorem square_sum_given_condition (x y : ℝ) :
  (2*x + 1)^2 + |y - 1| = 0 → x^2 + y^2 = 5/4 := by
  sorry

end square_sum_given_condition_l1151_115111


namespace complex_equation_solution_l1151_115174

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * (1 + 2 * Complex.I) = (a + b * Complex.I) * (1 + Complex.I) → 
  a = 3/2 ∧ b = 1/2 := by
  sorry

end complex_equation_solution_l1151_115174


namespace sum_of_coefficients_l1151_115196

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by
  sorry

end sum_of_coefficients_l1151_115196


namespace paulas_friends_l1151_115177

/-- Given the initial number of candies, additional candies bought, and candies per friend,
    prove that the number of friends is equal to the total number of candies divided by the number of candies per friend. -/
theorem paulas_friends (initial_candies additional_candies candies_per_friend : ℕ) 
  (h1 : initial_candies = 20)
  (h2 : additional_candies = 4)
  (h3 : candies_per_friend = 4)
  : (initial_candies + additional_candies) / candies_per_friend = 6 :=
by
  sorry

end paulas_friends_l1151_115177


namespace fence_perimeter_is_200_l1151_115194

/-- A rectangular fence with evenly spaced posts -/
structure RectangularFence where
  num_posts : ℕ
  post_width : ℝ
  post_spacing : ℝ
  length_width_ratio : ℝ

/-- Calculate the outer perimeter of a rectangular fence -/
def outer_perimeter (fence : RectangularFence) : ℝ :=
  sorry

/-- Theorem: The outer perimeter of the specified fence is 200 feet -/
theorem fence_perimeter_is_200 :
  let fence : RectangularFence :=
    { num_posts := 36
    , post_width := 0.5  -- 6 inches = 0.5 feet
    , post_spacing := 4
    , length_width_ratio := 2 }
  outer_perimeter fence = 200 :=
by sorry

end fence_perimeter_is_200_l1151_115194


namespace train_length_l1151_115176

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 16 → speed * time * (1000 / 3600) = 280 :=
by
  sorry

end train_length_l1151_115176


namespace quadratic_equation_condition_l1151_115172

theorem quadratic_equation_condition (m : ℝ) : (|m| = 2 ∧ m + 2 ≠ 0) ↔ m = 2 := by
  sorry

end quadratic_equation_condition_l1151_115172


namespace similar_triangles_perimeter_possibilities_l1151_115184

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def similar_triangles (t1 t2 : Triangle) : Prop := sorry

/-- A triangle is defined by its three side lengths. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- The perimeter of a triangle is the sum of its side lengths. -/
def perimeter (t : Triangle) : ℝ := t.side1 + t.side2 + t.side3

theorem similar_triangles_perimeter_possibilities :
  ∀ (t1 t2 : Triangle),
    similar_triangles t1 t2 →
    t1.side1 = 4 ∧ t1.side2 = 6 ∧ t1.side3 = 8 →
    (t2.side1 = 2 ∨ t2.side2 = 2 ∨ t2.side3 = 2) →
    (perimeter t2 = 4.5 ∨ perimeter t2 = 6 ∨ perimeter t2 = 9) :=
by sorry

end similar_triangles_perimeter_possibilities_l1151_115184


namespace det_matrix_l1151_115141

def matrix (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![y + 2, 2*y, 2*y;
     2*y, y + 2, 2*y;
     2*y, 2*y, y + 2]

theorem det_matrix (y : ℝ) :
  Matrix.det (matrix y) = 5*y^3 - 10*y^2 + 12*y + 8 := by
  sorry

end det_matrix_l1151_115141


namespace part_one_part_two_l1151_115185

-- Define the sets A, B, and U
def A (a : ℝ) : Set ℝ := {x | x - 3 ≤ x ∧ x ≤ 2*a + 1}
def B : Set ℝ := {x | x^2 + 2*x - 15 ≤ 0}
def U : Set ℝ := Set.univ

-- Part I: Prove the intersection of complement of A and B when a = 1
theorem part_one : (Set.compl (A 1) ∩ B) = {x : ℝ | -5 ≤ x ∧ x < -2} := by sorry

-- Part II: Prove the condition for A to be a subset of B
theorem part_two : ∀ a : ℝ, A a ⊆ B ↔ (a < -4 ∨ (-2 ≤ a ∧ a ≤ 1)) := by sorry

end part_one_part_two_l1151_115185


namespace power_729_minus_reciprocal_l1151_115167

theorem power_729_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * 2) :
  x^729 - 1/(x^729) = Complex.I * 2 := by
  sorry

end power_729_minus_reciprocal_l1151_115167


namespace expand_expression_l1151_115188

theorem expand_expression (x y : ℝ) : 5 * (4 * x^2 + 3 * x * y - 4) = 20 * x^2 + 15 * x * y - 20 := by
  sorry

end expand_expression_l1151_115188


namespace arccos_one_equals_zero_l1151_115187

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_equals_zero_l1151_115187


namespace pauls_paint_cans_l1151_115147

theorem pauls_paint_cans 
  (initial_rooms : ℕ) 
  (lost_cans : ℕ) 
  (remaining_rooms : ℕ) 
  (h1 : initial_rooms = 50)
  (h2 : lost_cans = 5)
  (h3 : remaining_rooms = 38) :
  (initial_rooms : ℚ) * lost_cans / (initial_rooms - remaining_rooms) = 21 :=
by sorry

end pauls_paint_cans_l1151_115147


namespace pure_imaginary_ratio_l1151_115146

/-- If p and q are nonzero real numbers and (3 - 4i)(p + qi) is pure imaginary, then p/q = -4/3 -/
theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end pure_imaginary_ratio_l1151_115146


namespace parabola_coefficient_l1151_115109

/-- Given a parabola y = ax^2 + bx + c with vertex at (p, p) and y-intercept at (0, -3p),
    where p ≠ 0, the coefficient b is equal to 8/p. -/
theorem parabola_coefficient (a b c p : ℝ) : 
  p ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p) → 
  a * 0^2 + b * 0 + c = -3 * p → 
  b = 8 / p := by
  sorry

end parabola_coefficient_l1151_115109


namespace sum_of_solutions_quadratic_l1151_115155

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 7*x - 12) → (∃ y : ℝ, y^2 = 7*y - 12 ∧ x + y = 7) :=
by sorry

end sum_of_solutions_quadratic_l1151_115155


namespace speed_gain_per_week_baseball_training_speed_gain_l1151_115168

/-- Calculates the speed gained per week given initial speed, training details, and final speed increase. -/
theorem speed_gain_per_week 
  (initial_speed : ℝ) 
  (training_sessions : ℕ) 
  (weeks_per_session : ℕ) 
  (speed_increase_percent : ℝ) : ℝ :=
  let final_speed := initial_speed * (1 + speed_increase_percent / 100)
  let total_speed_gain := final_speed - initial_speed
  let total_weeks := training_sessions * weeks_per_session
  total_speed_gain / total_weeks

/-- Proves that the speed gained per week is 1 mph under the given conditions. -/
theorem baseball_training_speed_gain :
  speed_gain_per_week 80 4 4 20 = 1 := by
  sorry

end speed_gain_per_week_baseball_training_speed_gain_l1151_115168


namespace fixed_intersection_point_l1151_115164

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an angle with two sides -/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- Predicate to check if two circles are non-overlapping -/
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

/-- Predicate to check if a point is on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Predicate to check if the angle touches both circles -/
def touches_circles (a : Angle) (c1 c2 : Circle) : Prop :=
  ∃ p1 p2 : ℝ × ℝ,
    a.side1 p1 ∧ on_circle p1 c1 ∧
    a.side2 p2 ∧ on_circle p2 c2 ∧
    p1 ≠ a.vertex ∧ p2 ≠ a.vertex

/-- The main theorem -/
theorem fixed_intersection_point
  (c1 c2 : Circle)
  (h_non_overlapping : non_overlapping c1 c2) :
  ∃ p : ℝ × ℝ,
    ∀ a : Angle,
      touches_circles a c1 c2 →
      ∃ t : ℝ,
        p.1 = a.vertex.1 + t * (p.1 - a.vertex.1) ∧
        p.2 = a.vertex.2 + t * (p.2 - a.vertex.2) :=
  sorry

end fixed_intersection_point_l1151_115164


namespace quadratic_equation_integer_solutions_l1151_115137

theorem quadratic_equation_integer_solutions :
  ∀ (x n : ℤ), x^2 + 3*x + 9 - 9*n^2 = 0 → (x = 0 ∨ x = -3) := by
  sorry

end quadratic_equation_integer_solutions_l1151_115137


namespace complex_fraction_equality_l1151_115199

theorem complex_fraction_equality : (1 - 2*I) / (2 + I) = -I := by
  sorry

end complex_fraction_equality_l1151_115199


namespace cookies_to_mike_is_23_l1151_115124

/-- The number of cookies Uncle Jude gave to Mike -/
def cookies_to_mike (total cookies_to_tim cookies_in_fridge : ℕ) : ℕ :=
  total - (cookies_to_tim + 2 * cookies_to_tim + cookies_in_fridge)

/-- Theorem: Uncle Jude gave 23 cookies to Mike -/
theorem cookies_to_mike_is_23 :
  cookies_to_mike 256 15 188 = 23 := by
  sorry

end cookies_to_mike_is_23_l1151_115124


namespace angle_ABC_measure_l1151_115160

/- Given a point B with three angles around it -/
def point_B (angle_ABC angle_ABD angle_CBD : ℝ) : Prop :=
  /- ∠CBD is a right angle -/
  angle_CBD = 90 ∧
  /- The sum of angles around point B is 200° -/
  angle_ABC + angle_ABD + angle_CBD = 200 ∧
  /- The measure of ∠ABD is 70° -/
  angle_ABD = 70

/- Theorem statement -/
theorem angle_ABC_measure :
  ∀ (angle_ABC angle_ABD angle_CBD : ℝ),
  point_B angle_ABC angle_ABD angle_CBD →
  angle_ABC = 40 := by
sorry

end angle_ABC_measure_l1151_115160


namespace benny_comic_books_l1151_115152

theorem benny_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 17) → initial = 22 := by
  sorry

end benny_comic_books_l1151_115152


namespace chair_arrangement_l1151_115129

theorem chair_arrangement (total_chairs : Nat) (h1 : total_chairs = 49) :
  (∃! (rows columns : Nat), rows ≥ 2 ∧ columns ≥ 2 ∧ rows * columns = total_chairs) :=
by sorry

end chair_arrangement_l1151_115129
