import Mathlib

namespace composition_value_l2065_206548

/-- Given two functions f and g, prove that g(f(3)) = 1902 -/
theorem composition_value (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = x^3 - 2) 
  (hg : ∀ x, g x = 3*x^2 + x + 2) : 
  g (f 3) = 1902 := by
  sorry

end composition_value_l2065_206548


namespace x_n_perfect_square_iff_b_10_l2065_206536

def x_n (b n : ℕ) : ℕ :=
  let ones := (b^(2*n) - b^(n+1)) / (b - 1)
  let twos := 2 * (b^n - 1) / (b - 1)
  ones + twos + 5

theorem x_n_perfect_square_iff_b_10 (b : ℕ) (h : b > 5) :
  (∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, x_n b n = k^2) ↔ b = 10 :=
sorry

end x_n_perfect_square_iff_b_10_l2065_206536


namespace valid_systematic_sample_l2065_206597

/-- Represents a systematic sample of student numbers -/
def SystematicSample (n : ℕ) (k : ℕ) (sample : Finset ℕ) : Prop :=
  ∃ (start : ℕ) (step : ℕ), 
    sample = Finset.image (fun i => start + i * step) (Finset.range k) ∧
    start ≤ n ∧
    ∀ i ∈ Finset.range k, start + i * step ≤ n

/-- The given sample is a valid systematic sample -/
theorem valid_systematic_sample :
  SystematicSample 50 5 {5, 15, 25, 35, 45} :=
by sorry

end valid_systematic_sample_l2065_206597


namespace direction_vector_y_component_l2065_206596

/-- Given a line determined by two points in 2D space, prove that if the direction vector
    has a specific x-component, then its y-component has a specific value. -/
theorem direction_vector_y_component 
  (p1 p2 : ℝ × ℝ) 
  (h1 : p1 = (-1, -1)) 
  (h2 : p2 = (3, 4)) 
  (direction : ℝ × ℝ) 
  (h_x_component : direction.1 = 3) : 
  direction.2 = 15/4 := by
sorry

end direction_vector_y_component_l2065_206596


namespace odd_function_product_nonpositive_l2065_206569

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : odd_function f) :
  ∀ x : ℝ, f x * f (-x) ≤ 0 := by
  sorry

end odd_function_product_nonpositive_l2065_206569


namespace last_digit_of_3_to_2010_l2065_206557

theorem last_digit_of_3_to_2010 (h : ∀ n : ℕ, 
  (3^n % 10) = (3^(n % 4) % 10)) : 
  3^2010 % 10 = 9 := by
sorry

end last_digit_of_3_to_2010_l2065_206557


namespace petals_per_rose_correct_petals_per_rose_l2065_206588

theorem petals_per_rose (petals_per_ounce : ℕ) (roses_per_bush : ℕ) (bushes_harvested : ℕ) 
  (bottles_produced : ℕ) (ounces_per_bottle : ℕ) : ℕ :=
  let total_ounces := bottles_produced * ounces_per_bottle
  let total_petals := total_ounces * petals_per_ounce
  let petals_per_bush := total_petals / bushes_harvested
  petals_per_bush / roses_per_bush

theorem correct_petals_per_rose :
  petals_per_rose 320 12 800 20 12 = 8 := by
  sorry

end petals_per_rose_correct_petals_per_rose_l2065_206588


namespace ratio_w_y_l2065_206558

-- Define the variables
variable (w x y z : ℚ)

-- Define the given ratios
def ratio_w_x : w / x = 5 / 4 := by sorry
def ratio_y_z : y / z = 4 / 3 := by sorry
def ratio_z_x : z / x = 1 / 8 := by sorry

-- Theorem to prove
theorem ratio_w_y (hw : w / x = 5 / 4) (hy : y / z = 4 / 3) (hz : z / x = 1 / 8) :
  w / y = 15 / 2 := by sorry

end ratio_w_y_l2065_206558


namespace remainder_7835_mod_11_l2065_206564

theorem remainder_7835_mod_11 : 7835 % 11 = (7 + 8 + 3 + 5) % 11 := by
  sorry

end remainder_7835_mod_11_l2065_206564


namespace line_segment_can_have_specific_length_l2065_206503

/-- A line segment is a geometric object with a measurable, finite length. -/
structure LineSegment where
  length : ℝ
  length_positive : length > 0

/-- Theorem: A line segment can have a specific, finite length (e.g., 0.7 meters). -/
theorem line_segment_can_have_specific_length : ∃ (s : LineSegment), s.length = 0.7 :=
sorry

end line_segment_can_have_specific_length_l2065_206503


namespace largest_m_for_quadratic_inequality_l2065_206538

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem largest_m_for_quadratic_inequality 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h1 : ∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x))
  (h2 : ∀ x : ℝ, f a b c x ≥ x)
  (h3 : ∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2)
  (h4 : ∃ x : ℝ, ∀ y : ℝ, f a b c x ≤ f a b c y)
  (h5 : ∃ x : ℝ, f a b c x = 0) :
  (∃ m : ℝ, m > 1 ∧ 
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) ∧
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x)) ∧
  (∀ m : ℝ, (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) → m ≤ 9) :=
by sorry

end largest_m_for_quadratic_inequality_l2065_206538


namespace nine_team_league_games_l2065_206529

/-- The number of games played in a league where each team plays every other team once -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 9 teams, where each team plays every other team once, 
    the total number of games played is 36 -/
theorem nine_team_league_games :
  num_games 9 = 36 := by
  sorry

end nine_team_league_games_l2065_206529


namespace power_function_decreasing_condition_l2065_206530

/-- A function f is a power function if it's of the form f(x) = x^a for some real a -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x > 0, f x = x^a

/-- A function f is decreasing on (0, +∞) if for all x, y in (0, +∞), x < y implies f(x) > f(y) -/
def is_decreasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x > f y

/-- The main theorem -/
theorem power_function_decreasing_condition (m : ℝ) : 
  (is_power_function (fun x => (m^2 - m - 1) * x^(m^2 - 2*m - 1)) ∧ 
   is_decreasing_on_positive_reals (fun x => (m^2 - m - 1) * x^(m^2 - 2*m - 1))) → 
  m = 2 := by
  sorry

end power_function_decreasing_condition_l2065_206530


namespace smallest_w_l2065_206589

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_w : 
  ∃ (w : ℕ), w > 0 ∧ 
  is_factor (2^6) (1916 * w) ∧
  is_factor (3^4) (1916 * w) ∧
  is_factor (5^3) (1916 * w) ∧
  is_factor (7^3) (1916 * w) ∧
  is_factor (11^3) (1916 * w) ∧
  ∀ (x : ℕ), x > 0 ∧ 
    is_factor (2^6) (1916 * x) ∧
    is_factor (3^4) (1916 * x) ∧
    is_factor (5^3) (1916 * x) ∧
    is_factor (7^3) (1916 * x) ∧
    is_factor (11^3) (1916 * x) →
    w ≤ x ∧
  w = 74145392000 :=
sorry

end smallest_w_l2065_206589


namespace dane_daughters_flowers_l2065_206549

def flowers_per_basket (initial_flowers_per_daughter : ℕ) (daughters : ℕ) (new_flowers : ℕ) (dead_flowers : ℕ) (baskets : ℕ) : ℕ :=
  ((initial_flowers_per_daughter * daughters + new_flowers) - dead_flowers) / baskets

theorem dane_daughters_flowers :
  flowers_per_basket 5 2 20 10 5 = 4 := by
  sorry

end dane_daughters_flowers_l2065_206549


namespace sqrt_expression_equals_one_l2065_206532

theorem sqrt_expression_equals_one :
  (Real.sqrt 24 - Real.sqrt 216) / Real.sqrt 6 + 5 = 1 := by
  sorry

end sqrt_expression_equals_one_l2065_206532


namespace jeans_final_price_is_correct_l2065_206516

def socks_price : ℝ := 5
def tshirt_price : ℝ := socks_price + 10
def jeans_price : ℝ := 2 * tshirt_price
def jeans_discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08

def jeans_final_price : ℝ :=
  let discounted_price := jeans_price * (1 - jeans_discount_rate)
  discounted_price * (1 + sales_tax_rate)

theorem jeans_final_price_is_correct :
  jeans_final_price = 27.54 := by sorry

end jeans_final_price_is_correct_l2065_206516


namespace lucky_years_2010_to_2014_l2065_206542

/-- A year is lucky if there exists a date in that year where the product of the month and day
    equals the last two digits of the year. -/
def is_lucky_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

/-- 2013 is not a lucky year, while 2010, 2011, 2012, and 2014 are lucky years. -/
theorem lucky_years_2010_to_2014 :
  is_lucky_year 2010 ∧ is_lucky_year 2011 ∧ is_lucky_year 2012 ∧
  ¬is_lucky_year 2013 ∧ is_lucky_year 2014 := by
  sorry

#check lucky_years_2010_to_2014

end lucky_years_2010_to_2014_l2065_206542


namespace jamies_father_weight_loss_l2065_206522

/-- Jamie's father's weight loss problem -/
theorem jamies_father_weight_loss 
  (calories_burned_per_day : ℕ)
  (calories_eaten_per_day : ℕ)
  (calories_per_pound : ℕ)
  (days_to_lose_weight : ℕ)
  (h1 : calories_burned_per_day = 2500)
  (h2 : calories_eaten_per_day = 2000)
  (h3 : calories_per_pound = 3500)
  (h4 : days_to_lose_weight = 35) :
  (days_to_lose_weight * (calories_burned_per_day - calories_eaten_per_day)) / calories_per_pound = 5 := by
  sorry


end jamies_father_weight_loss_l2065_206522


namespace two_axisymmetric_additions_l2065_206576

/-- Represents a position on a 4x4 grid --/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents a configuration of shaded squares on a 4x4 grid --/
def Configuration := List Position

/-- Checks if a configuration is axisymmetric --/
def isAxisymmetric (config : Configuration) : Bool :=
  sorry

/-- Counts the number of ways to add one square to make the configuration axisymmetric --/
def countAxisymmetricAdditions (initialConfig : Configuration) : Nat :=
  sorry

/-- The initial configuration with 3 shaded squares --/
def initialConfig : Configuration :=
  sorry

theorem two_axisymmetric_additions :
  countAxisymmetricAdditions initialConfig = 2 :=
sorry

end two_axisymmetric_additions_l2065_206576


namespace parabola_coefficient_b_l2065_206579

/-- 
Given a parabola y = ax^2 + bx + c with vertex (p, -p) and y-intercept (0, p), 
where p ≠ 0, the value of b is -4.
-/
theorem parabola_coefficient_b (a b c p : ℝ) : 
  p ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 - p) →
  (a * 0^2 + b * 0 + c = p) →
  b = -4 := by
  sorry

end parabola_coefficient_b_l2065_206579


namespace triangle_rational_area_l2065_206509

/-- Triangle with rational side lengths and angle bisectors has rational area -/
theorem triangle_rational_area (a b c fa fb fc : ℚ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  fa > 0 ∧ fb > 0 ∧ fc > 0 →  -- positive angle bisector lengths
  ∃ (area : ℚ), area > 0 ∧ area^2 = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c) / 16 :=
sorry

end triangle_rational_area_l2065_206509


namespace inequality_solution_set_l2065_206508

-- Define the inequality
def inequality (x : ℝ) : Prop := |x^2 - 5*x + 6| < x^2 - 4

-- Define the solution set
def solution_set : Set ℝ := {x | x > 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end inequality_solution_set_l2065_206508


namespace repair_cost_is_5000_l2065_206572

/-- Calculates the repair cost for a machine given its purchase price, transportation cost, selling price, and profit percentage. -/
def repair_cost (purchase_price transportation_cost selling_price profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + transportation_cost + (selling_price * 100 / (100 + profit_percentage) - purchase_price - transportation_cost)
  selling_price * 100 / (100 + profit_percentage) - purchase_price - transportation_cost

/-- Theorem stating that for the given conditions, the repair cost is 5000. -/
theorem repair_cost_is_5000 :
  repair_cost 10000 1000 24000 50 = 5000 := by
  sorry

end repair_cost_is_5000_l2065_206572


namespace intersection_point_AB_CD_l2065_206550

def A : ℝ × ℝ × ℝ := (8, -9, 5)
def B : ℝ × ℝ × ℝ := (18, -19, 15)
def C : ℝ × ℝ × ℝ := (2, 5, -8)
def D : ℝ × ℝ × ℝ := (4, -3, 12)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

theorem intersection_point_AB_CD :
  line_intersection A B C D = (16, -19, 13) := by sorry

end intersection_point_AB_CD_l2065_206550


namespace difference_of_squares_l2065_206521

theorem difference_of_squares (y : ℝ) : y^2 - 4 = (y + 2) * (y - 2) := by
  sorry

end difference_of_squares_l2065_206521


namespace books_about_sports_l2065_206545

theorem books_about_sports (total_books school_books : ℕ) 
  (h1 : total_books = 58) 
  (h2 : school_books = 19) : 
  total_books - school_books = 39 :=
sorry

end books_about_sports_l2065_206545


namespace fraction_meaningful_l2065_206582

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x - 5)) ↔ x ≠ 5 := by
sorry

end fraction_meaningful_l2065_206582


namespace parameterized_to_ordinary_equation_l2065_206524

theorem parameterized_to_ordinary_equation :
  ∀ (x y t : ℝ),
  (x = Real.sqrt t ∧ y = 2 * Real.sqrt (1 - t)) →
  (x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2) :=
by sorry

end parameterized_to_ordinary_equation_l2065_206524


namespace vaishalis_hats_l2065_206598

/-- The number of hats with three stripes each that Vaishali has -/
def hats_with_three_stripes : ℕ := sorry

/-- The number of hats with four stripes each that Vaishali has -/
def hats_with_four_stripes : ℕ := 3

/-- The number of hats with no stripes that Vaishali has -/
def hats_with_no_stripes : ℕ := 6

/-- The number of hats with five stripes each that Vaishali has -/
def hats_with_five_stripes : ℕ := 2

/-- The total number of stripes on all of Vaishali's hats -/
def total_stripes : ℕ := 34

/-- Theorem stating that the number of hats with three stripes is 4 -/
theorem vaishalis_hats : hats_with_three_stripes = 4 := by
  sorry

end vaishalis_hats_l2065_206598


namespace incorrect_permutations_hello_l2065_206552

def word_length : ℕ := 5
def repeated_letter_count : ℕ := 2

theorem incorrect_permutations_hello :
  (word_length.factorial / repeated_letter_count.factorial) - 1 = 119 := by
  sorry

end incorrect_permutations_hello_l2065_206552


namespace symmetry_about_y_axis_l2065_206520

/-- Given two lines in the xy-plane, this function checks if they are symmetric with respect to the y-axis -/
def symmetric_about_y_axis (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 (-x) y

/-- The original line: 3x - 4y + 5 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The symmetric line: 3x + 4y + 22 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 3 * x + 4 * y + 22 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to the y-axis -/
theorem symmetry_about_y_axis : symmetric_about_y_axis original_line symmetric_line := by
  sorry

end symmetry_about_y_axis_l2065_206520


namespace candy_probability_l2065_206534

def total_candies : ℕ := 24
def red_candies : ℕ := 12
def blue_candies : ℕ := 12
def terry_picks : ℕ := 2
def mary_picks : ℕ := 3

def same_color_probability : ℚ := 66 / 1771

theorem candy_probability :
  red_candies = blue_candies ∧
  red_candies + blue_candies = total_candies ∧
  terry_picks + mary_picks < total_candies →
  same_color_probability = (2 * (Nat.choose red_candies terry_picks * Nat.choose (red_candies - terry_picks) mary_picks)) / 
                           (Nat.choose total_candies terry_picks * Nat.choose (total_candies - terry_picks) mary_picks) :=
by sorry

end candy_probability_l2065_206534


namespace distance_A_to_B_l2065_206505

/-- Prove that the distance from A to B is 510 km given the travel conditions -/
theorem distance_A_to_B : 
  ∀ (d_AB : ℝ) (d_AC : ℝ) (t_E t_F : ℝ) (speed_ratio : ℝ),
  d_AC = 300 →
  t_E = 3 →
  t_F = 4 →
  speed_ratio = 2.2666666666666666 →
  (d_AB / t_E) / (d_AC / t_F) = speed_ratio →
  d_AB = 510 := by
sorry

end distance_A_to_B_l2065_206505


namespace xyz_product_l2065_206553

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 2 * y = -8)
  (eq2 : y * z + 2 * z = -8)
  (eq3 : z * x + 2 * x = -8) : 
  x * y * z = 32 := by
sorry

end xyz_product_l2065_206553


namespace sample_size_is_ten_l2065_206511

/-- Represents a collection of products -/
structure ProductCollection where
  total : Nat
  selected : Nat
  random_selection : selected ≤ total

/-- Definition of sample size for a product collection -/
def sample_size (pc : ProductCollection) : Nat := pc.selected

/-- Theorem: For a product collection with 80 total products and 10 randomly selected,
    the sample size is 10 -/
theorem sample_size_is_ten (pc : ProductCollection) 
  (h1 : pc.total = 80) 
  (h2 : pc.selected = 10) : 
  sample_size pc = 10 := by
  sorry


end sample_size_is_ten_l2065_206511


namespace strap_problem_l2065_206568

theorem strap_problem (shorter longer : ℝ) 
  (h1 : shorter + longer = 64)
  (h2 : longer = shorter + 48) :
  longer / shorter = 7 := by
  sorry

end strap_problem_l2065_206568


namespace chicken_eggs_per_chicken_l2065_206591

theorem chicken_eggs_per_chicken 
  (num_chickens : ℕ) 
  (num_cartons : ℕ) 
  (eggs_per_carton : ℕ) 
  (h1 : num_chickens = 20)
  (h2 : num_cartons = 10)
  (h3 : eggs_per_carton = 12) :
  (num_cartons * eggs_per_carton) / num_chickens = 6 :=
by sorry

end chicken_eggs_per_chicken_l2065_206591


namespace square_odd_implies_odd_l2065_206537

theorem square_odd_implies_odd (n : ℕ) : Odd (n^2) → Odd n := by
  sorry

end square_odd_implies_odd_l2065_206537


namespace no_solutions_abs_x_eq_3_abs_x_plus_2_l2065_206599

theorem no_solutions_abs_x_eq_3_abs_x_plus_2 :
  ∀ x : ℝ, ¬(|x| = 3 * (|x| + 2)) :=
by
  sorry

end no_solutions_abs_x_eq_3_abs_x_plus_2_l2065_206599


namespace debby_dvd_sale_l2065_206559

theorem debby_dvd_sale (original : ℕ) (left : ℕ) (sold : ℕ) : 
  original = 13 → left = 7 → sold = original - left → sold = 6 := by sorry

end debby_dvd_sale_l2065_206559


namespace union_of_A_and_B_l2065_206504

def A : Set ℤ := {1, 3}

def B : Set ℤ := {x | 0 < Real.log (x + 1) ∧ Real.log (x + 1) < 1/2}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end union_of_A_and_B_l2065_206504


namespace triangle_side_length_l2065_206544

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 3 → b = 5 → C = 2 * π / 3 → c = 7 := by
  sorry

end triangle_side_length_l2065_206544


namespace volume_ratio_in_divided_tetrahedron_l2065_206551

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Represents the ratio of distances on an edge -/
def ratio (P Q R : Point3D) : ℝ := sorry

/-- Theorem: Volume ratio in a divided tetrahedron -/
theorem volume_ratio_in_divided_tetrahedron (ABCD : Tetrahedron) 
  (P : Point3D) (Q : Point3D) (R : Point3D) (S : Point3D)
  (hP : ratio P ABCD.A ABCD.B = 1)
  (hQ : ratio Q ABCD.B ABCD.D = 1/2)
  (hR : ratio R ABCD.C ABCD.D = 1/2)
  (hS : ratio S ABCD.A ABCD.C = 1)
  (V1 V2 : ℝ)
  (hV : V1 < V2)
  (hV1V2 : V1 + V2 = volume ABCD)
  : V1 / V2 = 13 / 23 := by sorry

end volume_ratio_in_divided_tetrahedron_l2065_206551


namespace arithmetic_seq_sum_l2065_206574

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem stating that for an arithmetic sequence with S_17 = 170, a_7 + a_8 + a_12 = 30 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 17 = 170) :
  seq.a 7 + seq.a 8 + seq.a 12 = 30 := by
  sorry

end arithmetic_seq_sum_l2065_206574


namespace twenty_point_circle_special_chords_l2065_206593

/-- A circle with equally spaced points on its circumference -/
structure PointedCircle where
  n : ℕ  -- number of points
  (n_pos : n > 0)

/-- Counts chords in a PointedCircle satisfying certain length conditions -/
def count_special_chords (c : PointedCircle) : ℕ :=
  sorry

/-- Theorem statement -/
theorem twenty_point_circle_special_chords :
  ∃ (c : PointedCircle), c.n = 20 ∧ count_special_chords c = 120 :=
sorry

end twenty_point_circle_special_chords_l2065_206593


namespace eliminate_y_by_addition_l2065_206526

/-- Given a system of two linear equations in two variables x and y,
    prove that adding the first equation to twice the second equation
    eliminates the y variable. -/
theorem eliminate_y_by_addition (a b c d e f : ℝ) :
  let eq1 := (a * x + b * y = e)
  let eq2 := (c * x + d * y = f)
  (b = -2 * d) →
  ∃ k, (a * x + b * y) + 2 * (c * x + d * y) = k * x + e + 2 * f :=
by sorry

end eliminate_y_by_addition_l2065_206526


namespace roots_distribution_l2065_206584

/-- The polynomial p(z) = z^6 + 6z + 10 -/
def p (z : ℂ) : ℂ := z^6 + 6*z + 10

/-- The number of roots of p(z) in the first quadrant -/
def roots_first_quadrant : ℕ := 1

/-- The number of roots of p(z) in the second quadrant -/
def roots_second_quadrant : ℕ := 2

/-- The number of roots of p(z) in the third quadrant -/
def roots_third_quadrant : ℕ := 2

/-- The number of roots of p(z) in the fourth quadrant -/
def roots_fourth_quadrant : ℕ := 1

theorem roots_distribution :
  (∃ (z : ℂ), z.re > 0 ∧ z.im > 0 ∧ p z = 0) ∧
  (∃ (z1 z2 : ℂ), z1 ≠ z2 ∧ z1.re < 0 ∧ z1.im > 0 ∧ z2.re < 0 ∧ z2.im > 0 ∧ p z1 = 0 ∧ p z2 = 0) ∧
  (∃ (z1 z2 : ℂ), z1 ≠ z2 ∧ z1.re < 0 ∧ z1.im < 0 ∧ z2.re < 0 ∧ z2.im < 0 ∧ p z1 = 0 ∧ p z2 = 0) ∧
  (∃ (z : ℂ), z.re > 0 ∧ z.im < 0 ∧ p z = 0) :=
sorry

end roots_distribution_l2065_206584


namespace max_sequence_length_l2065_206528

/-- A sequence satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) (n m : ℕ) : Prop :=
  (∀ k, k < n → a k ≤ m) ∧
  (∀ k, 1 < k ∧ k < n - 1 → a (k - 1) ≠ a (k + 1)) ∧
  (∀ i₁ i₂ i₃ i₄, i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ < n →
    ¬(a i₁ = a i₃ ∧ a i₁ ≠ a i₂ ∧ a i₂ = a i₄))

/-- The maximum length of a valid sequence -/
def MaxSequenceLength (m : ℕ) : ℕ :=
  4 * m - 2

/-- Theorem: The maximum length of a valid sequence is 4m - 2 -/
theorem max_sequence_length (m : ℕ) (h : m > 0) :
  (∃ a n, n = MaxSequenceLength m ∧ ValidSequence a n m) ∧
  (∀ a n, ValidSequence a n m → n ≤ MaxSequenceLength m) :=
sorry

end max_sequence_length_l2065_206528


namespace joan_seashells_l2065_206592

/-- Calculates the number of seashells Joan has after giving some away -/
def remaining_seashells (found : ℕ) (given_away : ℕ) : ℕ :=
  found - given_away

/-- Proves that Joan has 16 seashells after finding 79 and giving away 63 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end joan_seashells_l2065_206592


namespace quadrilateral_inequality_l2065_206555

theorem quadrilateral_inequality (A B P : ℝ) (θ₁ θ₂ : ℝ) 
  (hA : A > 0) (hB : B > 0) (hP : P > 0) 
  (hP_bound : P ≤ A + B)
  (h_cos : A * Real.cos θ₁ + B * Real.cos θ₂ = P) :
  A * Real.sin θ₁ + B * Real.sin θ₂ ≤ Real.sqrt ((A + B - P) * (A + B + P)) := by
  sorry

end quadrilateral_inequality_l2065_206555


namespace roots_location_l2065_206578

theorem roots_location (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ (x₁ x₂ : ℝ), 
    (a < x₁ ∧ x₁ < b) ∧ 
    (b < x₂ ∧ x₂ < c) ∧ 
    (∀ x, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end roots_location_l2065_206578


namespace arithmetic_geometric_sequences_l2065_206556

/-- Arithmetic sequence {a_n} -/
def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 2

/-- Geometric sequence {b_n} -/
def geometric_sequence (n : ℕ) : ℝ := 2^(n - 1)

/-- Sum of first n terms of geometric sequence -/
def geometric_sum (n : ℕ) : ℝ := 2^n - 1

theorem arithmetic_geometric_sequences :
  (∀ n : ℕ, arithmetic_sequence n = 2 * n - 2) ∧
  arithmetic_sequence 2 = 2 ∧
  arithmetic_sequence 5 = 8 ∧
  (∀ n : ℕ, geometric_sequence n > 0) ∧
  geometric_sequence 1 = 1 ∧
  geometric_sequence 2 + geometric_sequence 3 = arithmetic_sequence 4 ∧
  (∀ n : ℕ, geometric_sum n = 2^n - 1) :=
sorry

end arithmetic_geometric_sequences_l2065_206556


namespace jane_rounds_played_l2065_206562

-- Define the parameters of the game
def points_per_round : ℕ := 10
def final_points : ℕ := 60
def lost_points : ℕ := 20

-- Define the theorem
theorem jane_rounds_played :
  (final_points + lost_points) / points_per_round = 8 :=
by sorry

end jane_rounds_played_l2065_206562


namespace total_lives_calculation_l2065_206540

theorem total_lives_calculation (initial_players additional_players lives_per_player : ℕ) :
  initial_players = 4 →
  additional_players = 5 →
  lives_per_player = 3 →
  (initial_players + additional_players) * lives_per_player = 27 :=
by sorry

end total_lives_calculation_l2065_206540


namespace triangle_longest_side_l2065_206577

theorem triangle_longest_side (x : ℕ) (h1 : x > 0) 
  (h2 : 5 * x + 6 * x + 7 * x = 720) 
  (h3 : 5 * x + 6 * x > 7 * x) 
  (h4 : 5 * x + 7 * x > 6 * x) 
  (h5 : 6 * x + 7 * x > 5 * x) :
  7 * x = 280 := by
  sorry

#check triangle_longest_side

end triangle_longest_side_l2065_206577


namespace sum_of_squared_differences_equals_three_l2065_206575

theorem sum_of_squared_differences_equals_three (a b c : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : 
  (b - c)^2 / ((a - b) * (a - c)) + 
  (c - a)^2 / ((b - c) * (b - a)) + 
  (a - b)^2 / ((c - a) * (c - b)) = 3 := by
  sorry

#check sum_of_squared_differences_equals_three

end sum_of_squared_differences_equals_three_l2065_206575


namespace ladder_problem_l2065_206535

theorem ladder_problem (ladder_length base_distance height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : base_distance = 5)
  (h3 : ladder_length ^ 2 = base_distance ^ 2 + height ^ 2) :
  height = 12 := by
  sorry

end ladder_problem_l2065_206535


namespace inequality_proof_l2065_206561

theorem inequality_proof (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 1/2) :
  (k + a/(b+c)) * (k + b/(c+a)) * (k + c/(a+b)) ≥ (k + 1/2)^3 := by
  sorry

end inequality_proof_l2065_206561


namespace correlation_coefficient_measures_linear_relationship_l2065_206518

/-- The correlation coefficient is a measure related to the relationship between two variables. -/
def correlation_coefficient : Type := sorry

/-- The strength of the linear relationship between two variables. -/
def linear_relationship_strength : Type := sorry

/-- The correlation coefficient measures the strength of the linear relationship between two variables. -/
theorem correlation_coefficient_measures_linear_relationship :
  correlation_coefficient = linear_relationship_strength := by sorry

end correlation_coefficient_measures_linear_relationship_l2065_206518


namespace four_numbers_average_l2065_206585

theorem four_numbers_average (a b c d : ℕ) : 
  a < b ∧ b < c ∧ c < d →  -- Four different positive integers
  a = 3 →                  -- Smallest number is 3
  (a + b + c + d) / 4 = 6 →  -- Average is 6
  d - a = 9 →              -- Difference between largest and smallest is maximized
  (b + c) / 2 = (9 : ℚ) / 2 := by  -- Average of middle two numbers is 4.5
sorry

end four_numbers_average_l2065_206585


namespace store_shirts_count_l2065_206523

theorem store_shirts_count (shirts_sold : ℕ) (shirts_left : ℕ) :
  shirts_sold = 21 →
  shirts_left = 28 →
  shirts_sold + shirts_left = 49 :=
by sorry

end store_shirts_count_l2065_206523


namespace third_group_first_student_l2065_206507

/-- Systematic sampling function that returns the number of the first student in a given group -/
def systematic_sample (total_students : ℕ) (sample_size : ℕ) (group : ℕ) : ℕ :=
  let interval := total_students / sample_size
  (group - 1) * interval

theorem third_group_first_student :
  systematic_sample 800 40 3 = 40 := by
  sorry

end third_group_first_student_l2065_206507


namespace distinct_subsets_removal_l2065_206563

theorem distinct_subsets_removal (n : ℕ) (X : Finset ℕ) (A : Fin n → Finset ℕ) 
  (h1 : n ≥ 2) 
  (h2 : X.card = n) 
  (h3 : ∀ i : Fin n, A i ⊆ X) 
  (h4 : ∀ i j : Fin n, i ≠ j → A i ≠ A j) :
  ∃ x ∈ X, ∀ i j : Fin n, i ≠ j → A i \ {x} ≠ A j \ {x} := by
  sorry

end distinct_subsets_removal_l2065_206563


namespace inequality_holds_l2065_206517

theorem inequality_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end inequality_holds_l2065_206517


namespace other_ticket_price_l2065_206512

/-- Represents the ticket sales scenario for the Red Rose Theatre --/
def theatre_sales (other_price : ℝ) : Prop :=
  let total_tickets : ℕ := 380
  let cheap_tickets : ℕ := 205
  let cheap_price : ℝ := 4.50
  let total_revenue : ℝ := 1972.50
  (cheap_tickets : ℝ) * cheap_price + (total_tickets - cheap_tickets : ℝ) * other_price = total_revenue

/-- Theorem stating that the price of the other tickets is $6.00 --/
theorem other_ticket_price : ∃ (price : ℝ), theatre_sales price ∧ price = 6 := by
  sorry

end other_ticket_price_l2065_206512


namespace min_distance_to_line_l2065_206566

/-- Given a right triangle with sides a, b, and hypotenuse c, and a point M(m, n) on the line ax+by+3c=0,
    the minimum value of m^2+n^2 is 9. -/
theorem min_distance_to_line (a b c : ℝ) (m n : ℝ → ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (∀ t, a * (m t) + b * (n t) + 3 * c = 0) →
  (∃ t₀, ∀ t, (m t)^2 + (n t)^2 ≥ (m t₀)^2 + (n t₀)^2) →
  ∃ t₀, (m t₀)^2 + (n t₀)^2 = 9 :=
by sorry

end min_distance_to_line_l2065_206566


namespace product_equals_negative_six_l2065_206543

/-- Given eight real numbers satisfying certain conditions, prove that their product equals -6 -/
theorem product_equals_negative_six
  (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ)
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end product_equals_negative_six_l2065_206543


namespace range_of_a_l2065_206539

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f satisfying the given conditions -/
noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x < 0 then 9*x + a^2/x + 7 else 
       if x > 0 then 9*x + a^2/x - 7 else 0

theorem range_of_a (a : ℝ) : 
  (IsOddFunction (f a)) → 
  (∀ x ≥ 0, f a x ≥ a + 1) →
  a ≤ -8/7 := by
sorry

end range_of_a_l2065_206539


namespace candidates_count_l2065_206594

theorem candidates_count (x : ℝ) : 
  (x > 0) →  -- number of candidates is positive
  (0.07 * x = 0.06 * x + 80) →  -- State B had 80 more selected candidates
  (x = 8000) := by
sorry

end candidates_count_l2065_206594


namespace necessary_not_sufficient_l2065_206581

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ x y : ℝ, x > y + 1 → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(x > y + 1)) :=
sorry

end necessary_not_sufficient_l2065_206581


namespace donut_holes_problem_l2065_206541

/-- Given the number of mini-cupcakes, students, and desserts per student,
    calculate the number of donut holes needed. -/
def donut_holes_needed (mini_cupcakes : ℕ) (students : ℕ) (desserts_per_student : ℕ) : ℕ :=
  students * desserts_per_student - mini_cupcakes

/-- Theorem stating that given 14 mini-cupcakes, 13 students, and 2 desserts per student,
    the number of donut holes needed is 12. -/
theorem donut_holes_problem :
  donut_holes_needed 14 13 2 = 12 := by
  sorry


end donut_holes_problem_l2065_206541


namespace solution_is_three_l2065_206514

/-- A linear function passing through (-2, 0) with y-intercept 3 -/
structure LinearFunction where
  k : ℝ
  k_nonzero : k ≠ 0
  passes_through : k * (-2) + 3 = 0

/-- The solution to k(x-5)+3=0 is x=3 -/
theorem solution_is_three (f : LinearFunction) : 
  ∃ x : ℝ, f.k * (x - 5) + 3 = 0 ∧ x = 3 := by
  sorry

end solution_is_three_l2065_206514


namespace fraction_undefined_values_l2065_206513

def undefined_values (b : ℝ) : Prop :=
  b^2 - 9 = 0

theorem fraction_undefined_values :
  {b : ℝ | undefined_values b} = {-3, 3} := by
  sorry

end fraction_undefined_values_l2065_206513


namespace officer_selection_count_l2065_206519

/-- The number of ways to choose officers from a club -/
def choose_officers (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (n - k + 1).factorial / (n - k).factorial

/-- Theorem: Choosing 5 officers from 15 members results in 360,360 possibilities -/
theorem officer_selection_count :
  choose_officers 15 5 = 360360 := by
  sorry

end officer_selection_count_l2065_206519


namespace log_equality_implies_golden_ratio_l2065_206501

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 9 = Real.log q / Real.log 12) ∧
  (Real.log p / Real.log 9 = Real.log (p + q) / Real.log 16) →
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end log_equality_implies_golden_ratio_l2065_206501


namespace sum_of_three_numbers_l2065_206527

theorem sum_of_three_numbers : 0.8 + (1 / 2 : ℚ) + 0.9 = 2.2 := by
  sorry

end sum_of_three_numbers_l2065_206527


namespace two_digit_number_problem_l2065_206506

theorem two_digit_number_problem (A B : ℕ) : 
  (A ≥ 1 ∧ A ≤ 9) →  -- A is a digit from 1 to 9 (tens digit)
  (B ≥ 0 ∧ B ≤ 9) →  -- B is a digit from 0 to 9 (ones digit)
  (10 * A + B) - 21 = 14 →  -- The equation AB - 21 = 14
  B = 5 := by  -- We want to prove B = 5
sorry

end two_digit_number_problem_l2065_206506


namespace tank_capacity_l2065_206546

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 9 →
  final_fraction = 7/8 →
  (initial_fraction * C + added_amount = final_fraction * C) →
  C = 72 :=
by sorry

#check tank_capacity

end tank_capacity_l2065_206546


namespace max_value_of_expression_l2065_206554

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 1 := by
  sorry

end max_value_of_expression_l2065_206554


namespace fifth_term_is_negative_three_l2065_206525

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 5th term of the sequence is -3 -/
theorem fifth_term_is_negative_three
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_third : a 3 = -5)
  (h_ninth : a 9 = 1) :
  a 5 = -3 := by
sorry

end fifth_term_is_negative_three_l2065_206525


namespace maggies_earnings_l2065_206515

/-- Calculates the total earnings for Maggie's magazine subscription sales --/
theorem maggies_earnings 
  (family_commission : ℕ) 
  (neighbor_commission : ℕ)
  (bonus_threshold : ℕ)
  (bonus_base : ℕ)
  (bonus_per_extra : ℕ)
  (family_subscriptions : ℕ)
  (neighbor_subscriptions : ℕ)
  (h1 : family_commission = 7)
  (h2 : neighbor_commission = 6)
  (h3 : bonus_threshold = 10)
  (h4 : bonus_base = 10)
  (h5 : bonus_per_extra = 1)
  (h6 : family_subscriptions = 9)
  (h7 : neighbor_subscriptions = 6) :
  family_commission * family_subscriptions +
  neighbor_commission * neighbor_subscriptions +
  bonus_base +
  (if family_subscriptions + neighbor_subscriptions > bonus_threshold
   then (family_subscriptions + neighbor_subscriptions - bonus_threshold) * bonus_per_extra
   else 0) = 114 := by
  sorry


end maggies_earnings_l2065_206515


namespace A_subset_B_l2065_206590

def A : Set ℕ := {x | ∃ a : ℕ, a > 0 ∧ x = a^2 + 1}
def B : Set ℕ := {y | ∃ b : ℕ, b > 0 ∧ y = b^2 - 4*b + 5}

theorem A_subset_B : A ⊆ B := by
  sorry

end A_subset_B_l2065_206590


namespace sum_of_cubes_zero_l2065_206587

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -7) : a^3 + b^3 = 0 := by
  sorry

end sum_of_cubes_zero_l2065_206587


namespace problem_solution_l2065_206500

def f (x a : ℝ) : ℝ := |2*x - 1| + |x + a|

theorem problem_solution :
  (∀ x : ℝ, f x 1 ≥ 3 ↔ x ≥ 1 ∨ x ≤ -1) ∧
  ((∃ x : ℝ, f x a ≤ |a - 1|) → a ≤ 1/4) :=
by sorry

end problem_solution_l2065_206500


namespace function_through_point_l2065_206510

/-- Proves that if the function y = k/x passes through the point (3, -1), then k = -3 -/
theorem function_through_point (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f 3 = -1) → k = -3 := by
  sorry

end function_through_point_l2065_206510


namespace plane_parallel_transitivity_l2065_206547

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem plane_parallel_transitivity (α β γ : Plane) :
  parallel γ α → parallel γ β → parallel α β := by sorry

end plane_parallel_transitivity_l2065_206547


namespace frank_final_position_l2065_206583

def dance_sequence (start : Int) : Int :=
  let step1 := start - 5
  let step2 := step1 + 10
  let step3 := step2 - 2
  let step4 := step3 + (2 * 2)
  step4

theorem frank_final_position :
  dance_sequence 0 = 7 := by
  sorry

end frank_final_position_l2065_206583


namespace ellipse_and_hyperbola_equations_l2065_206570

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of a line (asymptote) -/
structure Line where
  m : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = m * x

def foci : (Point × Point) := ⟨⟨-5, 0⟩, ⟨5, 0⟩⟩
def intersectionPoint : Point := ⟨4, 3⟩

/-- Theorem stating the equations of the ellipse and hyperbola -/
theorem ellipse_and_hyperbola_equations 
  (e : Ellipse) 
  (h : Hyperbola) 
  (l : Line) 
  (hfoci : e.a^2 - e.b^2 = h.a^2 + h.b^2 ∧ e.a^2 - e.b^2 = 25) 
  (hpoint_on_ellipse : e.equation intersectionPoint.x intersectionPoint.y) 
  (hpoint_on_line : l.equation intersectionPoint.x intersectionPoint.y) 
  (hline_is_asymptote : l.m = h.b / h.a) :
  e.a^2 = 40 ∧ e.b^2 = 15 ∧ h.a^2 = 16 ∧ h.b^2 = 9 := by
  sorry


end ellipse_and_hyperbola_equations_l2065_206570


namespace solve_a_b_l2065_206533

def U : Set ℝ := Set.univ

def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + 12*b = 0}

def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b = 0}

theorem solve_a_b :
  ∀ a b : ℝ, 
    (2 ∈ (U \ A a b) ∩ (B a b)) → 
    (4 ∈ (A a b) ∩ (U \ B a b)) → 
    a = 8/7 ∧ b = -12/7 := by
  sorry

end solve_a_b_l2065_206533


namespace david_twice_rosy_age_l2065_206595

/-- The number of years it will take for David to be twice as old as Rosy -/
def years_until_twice_age : ℕ :=
  sorry

/-- David's current age -/
def david_age : ℕ :=
  sorry

/-- Rosy's current age -/
def rosy_age : ℕ :=
  12

theorem david_twice_rosy_age :
  (david_age = rosy_age + 18) →
  (david_age + years_until_twice_age = 2 * (rosy_age + years_until_twice_age)) →
  years_until_twice_age = 6 :=
by sorry

end david_twice_rosy_age_l2065_206595


namespace circle_sectors_and_square_area_l2065_206573

/-- Given a circle with radius 6 and two perpendicular diameters, 
    prove that the sum of the areas of two 120° sectors and 
    the square formed by connecting the diameter endpoints 
    is equal to 24π + 144. -/
theorem circle_sectors_and_square_area :
  let r : ℝ := 6
  let sector_angle : ℝ := 120
  let sector_area := (sector_angle / 360) * π * r^2
  let square_side := 2 * r
  let square_area := square_side^2
  2 * sector_area + square_area = 24 * π + 144 := by
sorry

end circle_sectors_and_square_area_l2065_206573


namespace sqrt_sum_implies_product_l2065_206502

theorem sqrt_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8 →
  (10 + x) * (30 - x) = 144 := by
sorry

end sqrt_sum_implies_product_l2065_206502


namespace third_even_integer_l2065_206586

/-- Given four consecutive even integers where the sum of the second and fourth is 156,
    prove that the third integer is 78. -/
theorem third_even_integer (n : ℤ) : 
  (n + 2) + (n + 6) = 156 → n + 4 = 78 := by
  sorry

end third_even_integer_l2065_206586


namespace williams_land_percentage_l2065_206565

/-- Given a village with farm tax and Mr. William's tax payment, calculate the percentage of
    Mr. William's taxable land over the total taxable land of the village. -/
theorem williams_land_percentage 
  (total_tax : ℝ) 
  (williams_tax : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : williams_tax = 480) :
  williams_tax / total_tax = 0.125 := by
  sorry

#check williams_land_percentage

end williams_land_percentage_l2065_206565


namespace fruit_basket_count_l2065_206560

/-- The number of ways to choose k items from n identical items -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := (n + k - 1).choose k

/-- The number of possible fruit baskets -/
def fruit_baskets (apples oranges : ℕ) : ℕ :=
  (apples) * (choose_with_repetition (oranges + 1) 1)

theorem fruit_basket_count :
  fruit_baskets 7 12 = 91 :=
by sorry

end fruit_basket_count_l2065_206560


namespace relay_arrangements_verify_arrangements_l2065_206580

def total_athletes : ℕ := 8
def relay_positions : ℕ := 4

def arrangements_condition1 : ℕ := 60
def arrangements_condition2 : ℕ := 480
def arrangements_condition3 : ℕ := 180

/-- Theorem stating the number of arrangements for each condition -/
theorem relay_arrangements :
  (arrangements_condition1 = 60) ∧
  (arrangements_condition2 = 480) ∧
  (arrangements_condition3 = 180) := by
  sorry

/-- Function to calculate the number of arrangements for condition 1 -/
def calc_arrangements_condition1 : ℕ :=
  2 * 1 * 6 * 5

/-- Function to calculate the number of arrangements for condition 2 -/
def calc_arrangements_condition2 : ℕ :=
  2 * 2 * 6 * 5 * 4

/-- Function to calculate the number of arrangements for condition 3 -/
def calc_arrangements_condition3 : ℕ :=
  2 * 1 * (6 * 5 / (2 * 1)) * 3 * 2 * 1

/-- Theorem proving that the calculated arrangements match the given ones -/
theorem verify_arrangements :
  (calc_arrangements_condition1 = arrangements_condition1) ∧
  (calc_arrangements_condition2 = arrangements_condition2) ∧
  (calc_arrangements_condition3 = arrangements_condition3) := by
  sorry

end relay_arrangements_verify_arrangements_l2065_206580


namespace local_face_value_difference_l2065_206567

/-- The numeral we're working with -/
def numeral : ℕ := 65793

/-- The digit we're focusing on -/
def digit : ℕ := 7

/-- The place value of the digit in the numeral (hundreds) -/
def place_value : ℕ := 100

/-- The local value of the digit in the numeral -/
def local_value : ℕ := digit * place_value

/-- The face value of the digit -/
def face_value : ℕ := digit

/-- Theorem stating the difference between local value and face value -/
theorem local_face_value_difference :
  local_value - face_value = 693 := by sorry

end local_face_value_difference_l2065_206567


namespace oliver_used_30_tickets_l2065_206531

/-- The number of tickets Oliver used at the town carnival -/
def olivers_tickets (ferris_wheel_rides bumper_car_rides tickets_per_ride : ℕ) : ℕ :=
  (ferris_wheel_rides + bumper_car_rides) * tickets_per_ride

/-- Theorem: Oliver used 30 tickets at the town carnival -/
theorem oliver_used_30_tickets :
  olivers_tickets 7 3 3 = 30 := by
  sorry

end oliver_used_30_tickets_l2065_206531


namespace parallel_vectors_x_value_l2065_206571

theorem parallel_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → x = 2 := by
  sorry

end parallel_vectors_x_value_l2065_206571
