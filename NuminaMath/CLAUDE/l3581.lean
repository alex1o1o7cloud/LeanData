import Mathlib

namespace existence_of_n_l3581_358164

theorem existence_of_n (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hcd : c * d = 1) :
  ∃ n : ℕ, (a * b ≤ n^2) ∧ (n^2 ≤ (a + c) * (b + d)) := by
  sorry

end existence_of_n_l3581_358164


namespace consistent_coloring_pattern_l3581_358150

-- Define a hexagonal board
structure HexBoard where
  size : ℕ
  traversal : List ℕ

-- Define the coloring function
def color (n : ℕ) : String :=
  if n % 3 = 0 then "Black"
  else if n % 3 = 1 then "Red"
  else "White"

-- Define a property that checks if two boards have the same coloring pattern
def sameColoringPattern (board1 board2 : HexBoard) : Prop :=
  board1.traversal.map color = board2.traversal.map color

-- Theorem statement
theorem consistent_coloring_pattern 
  (board1 board2 : HexBoard) 
  (h : board1.traversal.length = board2.traversal.length) : 
  sameColoringPattern board1 board2 := by
  sorry

end consistent_coloring_pattern_l3581_358150


namespace parity_and_squares_equivalence_l3581_358178

theorem parity_and_squares_equivalence (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a % 2 = b % 2) ↔ (∃ (c d : ℕ), 0 < c ∧ 0 < d ∧ a^2 + b^2 + c^2 + 1 = d^2) := by
  sorry

end parity_and_squares_equivalence_l3581_358178


namespace lemonade_scaling_l3581_358137

/-- Represents a lemonade recipe -/
structure LemonadeRecipe where
  lemons : ℚ
  sugar : ℚ
  gallons : ℚ

/-- The original recipe -/
def originalRecipe : LemonadeRecipe :=
  { lemons := 30
  , sugar := 5
  , gallons := 40 }

/-- Calculate the amount of an ingredient needed for a given number of gallons -/
def calculateIngredient (original : LemonadeRecipe) (ingredient : ℚ) (targetGallons : ℚ) : ℚ :=
  (ingredient / original.gallons) * targetGallons

/-- The theorem to prove -/
theorem lemonade_scaling (recipe : LemonadeRecipe) (targetGallons : ℚ) :
  let scaledLemons := calculateIngredient recipe recipe.lemons targetGallons
  let scaledSugar := calculateIngredient recipe recipe.sugar targetGallons
  recipe.gallons = 40 ∧ recipe.lemons = 30 ∧ recipe.sugar = 5 ∧ targetGallons = 10 →
  scaledLemons = 7.5 ∧ scaledSugar = 1.25 := by
  sorry

end lemonade_scaling_l3581_358137


namespace abs_inequality_solution_set_l3581_358193

theorem abs_inequality_solution_set (x : ℝ) :
  |x - 2| < |2*x + 1| ↔ x < -3 ∨ x > 1/3 := by sorry

end abs_inequality_solution_set_l3581_358193


namespace correct_calculation_l3581_358152

theorem correct_calculation (x : ℝ) : 2 * x * x^2 = 2 * x^3 := by
  sorry

end correct_calculation_l3581_358152


namespace candy_sharing_l3581_358114

theorem candy_sharing (hugh tommy melany : ℕ) (h1 : hugh = 8) (h2 : tommy = 6) (h3 : melany = 7) :
  (hugh + tommy + melany) / 3 = 7 := by
  sorry

end candy_sharing_l3581_358114


namespace wholesale_price_is_90_l3581_358108

def retail_price : ℝ := 120

def discount_rate : ℝ := 0.1

def profit_rate : ℝ := 0.2

def selling_price (retail : ℝ) (discount : ℝ) : ℝ :=
  retail * (1 - discount)

def profit (wholesale : ℝ) (rate : ℝ) : ℝ :=
  wholesale * rate

theorem wholesale_price_is_90 :
  ∃ (wholesale : ℝ),
    selling_price retail_price discount_rate = wholesale + profit wholesale profit_rate ∧
    wholesale = 90 := by sorry

end wholesale_price_is_90_l3581_358108


namespace senior_teachers_in_sample_l3581_358115

theorem senior_teachers_in_sample
  (total_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (sample_intermediate : ℕ)
  (h_total : total_teachers = 300)
  (h_intermediate : intermediate_teachers = 192)
  (h_sample_intermediate : sample_intermediate = 64)
  (h_ratio : ∃ k : ℕ, k > 0 ∧ total_teachers - intermediate_teachers = 9 * k ∧ 5 * k = 4 * k + k) :
  ∃ sample_size : ℕ,
    sample_size * intermediate_teachers = sample_intermediate * total_teachers ∧
    ∃ sample_senior : ℕ,
      9 * sample_senior = 5 * (sample_size - sample_intermediate) ∧
      sample_senior = 20 :=
sorry

end senior_teachers_in_sample_l3581_358115


namespace hyperbola_min_eccentricity_asymptote_l3581_358110

/-- The asymptotic equation of a hyperbola with minimum eccentricity -/
theorem hyperbola_min_eccentricity_asymptote (m : ℝ) (h : m > 0) :
  let e := Real.sqrt (m + 4 / m + 1)
  let hyperbola := fun (x y : ℝ) => x^2 / m - y^2 / (m^2 + 4) = 1
  let asymptote := fun (x : ℝ) => (2 * x, -2 * x)
  (∀ m' > 0, e ≤ Real.sqrt (m' + 4 / m' + 1)) →
  (∃ t : ℝ, hyperbola (asymptote t).1 (asymptote t).2) :=
by sorry


end hyperbola_min_eccentricity_asymptote_l3581_358110


namespace triangle_dissection_theorem_l3581_358143

/-- A triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents a dissection of a triangle -/
def Dissection (t : Triangle) := List (List (ℝ × ℝ))

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if one triangle is a reflection of another -/
def is_reflection (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a dissection can transform one triangle to another using only translations -/
def can_transform_by_translation (d : Dissection t1) (t1 t2 : Triangle) : Prop := sorry

theorem triangle_dissection_theorem (t1 t2 : Triangle) :
  are_congruent t1 t2 → is_reflection t1 t2 →
  ∃ (d : Dissection t1), can_transform_by_translation d t1 t2 ∧ d.length ≤ 4 := by
  sorry

end triangle_dissection_theorem_l3581_358143


namespace only_sixteen_seventeen_not_divide_l3581_358103

/-- A number satisfying the conditions of the problem -/
def special_number (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 30, k + 2 ∣ n ∨ k + 3 ∣ n

/-- The theorem stating that 16 and 17 are the only consecutive numbers
    that don't divide the special number -/
theorem only_sixteen_seventeen_not_divide (n : ℕ) (h : special_number n) :
    ∃! (a : ℕ), a ∈ Finset.range 30 ∧ ¬(a + 2 ∣ n) ∧ ¬(a + 3 ∣ n) ∧ a = 14 := by
  sorry

#check only_sixteen_seventeen_not_divide

end only_sixteen_seventeen_not_divide_l3581_358103


namespace harris_feeds_one_carrot_per_day_l3581_358196

/-- Represents the number of carrots Harris feeds his dog per day -/
def carrots_per_day : ℚ :=
  let carrots_per_bag : ℕ := 5
  let cost_per_bag : ℚ := 2
  let annual_spend : ℚ := 146
  let days_per_year : ℕ := 365
  (annual_spend / days_per_year.cast) / cost_per_bag * carrots_per_bag

/-- Proves that Harris feeds his dog 1 carrot per day -/
theorem harris_feeds_one_carrot_per_day : 
  carrots_per_day = 1 := by sorry

end harris_feeds_one_carrot_per_day_l3581_358196


namespace lcm_from_product_and_hcf_l3581_358117

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h_product : a * b = 82500)
  (h_hcf : Nat.gcd a b = 55) :
  Nat.lcm a b = 1500 := by
  sorry

end lcm_from_product_and_hcf_l3581_358117


namespace divisibility_implication_l3581_358177

theorem divisibility_implication (u v : ℤ) : 
  (9 ∣ u^2 + u*v + v^2) → (3 ∣ u) ∧ (3 ∣ v) := by
  sorry

end divisibility_implication_l3581_358177


namespace unique_magnitude_quadratic_l3581_358180

/-- For the quadratic equation z^2 - 10z + 50 = 0, there is only one possible value for |z| -/
theorem unique_magnitude_quadratic : 
  ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m :=
by sorry

end unique_magnitude_quadratic_l3581_358180


namespace constant_term_of_polynomial_product_l3581_358156

theorem constant_term_of_polynomial_product :
  let p : Polynomial ℤ := X^3 + 2*X + 7
  let q : Polynomial ℤ := 2*X^4 + 3*X^2 + 10
  (p * q).coeff 0 = 70 := by sorry

end constant_term_of_polynomial_product_l3581_358156


namespace rahims_average_book_price_l3581_358169

/-- Calculates the average price per book given two separate book purchases -/
def average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating that the average price per book for Rahim's purchases is 20 -/
theorem rahims_average_book_price :
  average_price_per_book 50 1000 40 800 = 20 := by
  sorry

end rahims_average_book_price_l3581_358169


namespace ages_sum_l3581_358174

theorem ages_sum (a b c : ℕ) 
  (h1 : a = 20 + b + c) 
  (h2 : a^2 = 2000 + (b + c)^2) : 
  a + b + c = 80 := by
sorry

end ages_sum_l3581_358174


namespace polynomial_inequality_conditions_l3581_358153

/-- A polynomial function of degree 3 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The main theorem stating the conditions on a, b, and c -/
theorem polynomial_inequality_conditions
  (a b c : ℝ)
  (h : ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → f a b c (x + y) ≥ f a b c x + f a b c y) :
  a ≥ (3/2) * (9*c)^(1/3) ∧ c ≤ 0 ∧ b ∈ Set.univ :=
sorry

end polynomial_inequality_conditions_l3581_358153


namespace negation_of_universal_positive_square_plus_two_l3581_358120

theorem negation_of_universal_positive_square_plus_two :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 ≤ 0) := by sorry

end negation_of_universal_positive_square_plus_two_l3581_358120


namespace inequality_proof_l3581_358104

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  x / (1 + x^2) + y / (1 + y^2) + z / (1 + z^2) ≤ 3 * Real.sqrt 3 / 4 := by
  sorry

end inequality_proof_l3581_358104


namespace average_screen_time_l3581_358182

/-- Calculates the average screen time per player in minutes given the screen times for 5 players in seconds -/
theorem average_screen_time (point_guard shooting_guard small_forward power_forward center : ℕ) 
  (h1 : point_guard = 130)
  (h2 : shooting_guard = 145)
  (h3 : small_forward = 85)
  (h4 : power_forward = 60)
  (h5 : center = 180) :
  (point_guard + shooting_guard + small_forward + power_forward + center) / (5 * 60) = 2 := by
  sorry

end average_screen_time_l3581_358182


namespace total_snake_owners_l3581_358102

theorem total_snake_owners (total : Nat) (only_dogs : Nat) (only_cats : Nat) (only_birds : Nat) (only_snakes : Nat)
  (cats_and_dogs : Nat) (birds_and_dogs : Nat) (birds_and_cats : Nat) (snakes_and_dogs : Nat) (snakes_and_cats : Nat)
  (snakes_and_birds : Nat) (cats_dogs_snakes : Nat) (cats_dogs_birds : Nat) (cats_birds_snakes : Nat)
  (dogs_birds_snakes : Nat) (all_four : Nat)
  (h1 : total = 150)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_birds = 10)
  (h5 : only_snakes = 7)
  (h6 : cats_and_dogs = 15)
  (h7 : birds_and_dogs = 12)
  (h8 : birds_and_cats = 8)
  (h9 : snakes_and_dogs = 3)
  (h10 : snakes_and_cats = 4)
  (h11 : snakes_and_birds = 2)
  (h12 : cats_dogs_snakes = 5)
  (h13 : cats_dogs_birds = 4)
  (h14 : cats_birds_snakes = 6)
  (h15 : dogs_birds_snakes = 9)
  (h16 : all_four = 10) :
  only_snakes + snakes_and_dogs + snakes_and_cats + snakes_and_birds + cats_dogs_snakes + cats_birds_snakes + dogs_birds_snakes + all_four = 46 := by
  sorry

end total_snake_owners_l3581_358102


namespace rotation_sum_l3581_358168

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Represents a rotation transformation -/
structure Rotation where
  angle : ℝ
  center : Point

/-- Checks if a rotation transforms one triangle to another -/
def rotates (r : Rotation) (t1 t2 : Triangle) : Prop := sorry

theorem rotation_sum (t1 t2 : Triangle) (r : Rotation) :
  t1.p1 = Point.mk 2 2 ∧
  t1.p2 = Point.mk 2 14 ∧
  t1.p3 = Point.mk 18 2 ∧
  t2.p1 = Point.mk 32 26 ∧
  t2.p2 = Point.mk 44 26 ∧
  t2.p3 = Point.mk 32 10 ∧
  rotates r t1 t2 ∧
  0 < r.angle ∧ r.angle < 180 →
  r.angle + r.center.x + r.center.y = 124 := by
  sorry

end rotation_sum_l3581_358168


namespace circle_transformation_l3581_358149

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point vertically by a given amount -/
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ := (p.1, p.2 + dy)

/-- The initial coordinates of the center of circle S -/
def initial_point : ℝ × ℝ := (3, -4)

/-- The vertical translation amount -/
def translation_amount : ℝ := 5

theorem circle_transformation :
  translate_y (reflect_x initial_point) translation_amount = (3, 9) := by
  sorry

end circle_transformation_l3581_358149


namespace complex_modulus_sqrt_5_l3581_358194

theorem complex_modulus_sqrt_5 (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : a + 2 * i = 1 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
sorry

end complex_modulus_sqrt_5_l3581_358194


namespace triangle_trigonometry_l3581_358142

theorem triangle_trigonometry (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  2 * a * Real.sin B = Real.sqrt 3 * b →
  Real.cos C = 5 / 13 →
  Real.sin A = Real.sqrt 3 / 2 ∧
  Real.cos B = (12 * Real.sqrt 3 - 5) / 26 := by
sorry

end triangle_trigonometry_l3581_358142


namespace hugo_roll_four_given_win_l3581_358107

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 4
def roll_four_prob : ℚ := 1 / die_sides

-- Define the probability of Hugo winning given his first roll was 4
def hugo_win_given_four : ℚ := 145 / 1296

-- Theorem statement
theorem hugo_roll_four_given_win (
  num_players : ℕ) (die_sides : ℕ) (hugo_win_prob : ℚ) (roll_four_prob : ℚ) (hugo_win_given_four : ℚ) :
  num_players = 5 →
  die_sides = 6 →
  hugo_win_prob = 1 / num_players →
  roll_four_prob = 1 / die_sides →
  hugo_win_given_four = 145 / 1296 →
  (roll_four_prob * hugo_win_given_four) / hugo_win_prob = 145 / 1552 :=
by sorry

end hugo_roll_four_given_win_l3581_358107


namespace uniform_rv_expected_value_l3581_358135

/-- A random variable uniformly distributed in the interval (a, b) -/
def UniformRV (a b : ℝ) : Type := ℝ

/-- The expected value of a random variable -/
def ExpectedValue (X : Type) : ℝ := sorry

/-- Theorem: The expected value of a uniformly distributed random variable -/
theorem uniform_rv_expected_value (a b : ℝ) (h : a < b) :
  ExpectedValue (UniformRV a b) = (a + b) / 2 := by sorry

end uniform_rv_expected_value_l3581_358135


namespace surface_area_specific_parallelepiped_l3581_358158

/-- The surface area of a rectangular parallelepiped with given face areas -/
def surface_area_parallelepiped (a b c : ℝ) : ℝ :=
  2 * (a + b + c)

/-- Theorem: The surface area of a rectangular parallelepiped with face areas 4, 3, and 6 is 26 -/
theorem surface_area_specific_parallelepiped :
  surface_area_parallelepiped 4 3 6 = 26 := by
  sorry

#check surface_area_specific_parallelepiped

end surface_area_specific_parallelepiped_l3581_358158


namespace walts_earnings_l3581_358161

/-- Proves that Walt's total earnings from his part-time job were $9000 -/
theorem walts_earnings (interest_rate_1 interest_rate_2 : ℝ) 
  (investment_2 total_interest : ℝ) :
  interest_rate_1 = 0.09 →
  interest_rate_2 = 0.08 →
  investment_2 = 4000 →
  total_interest = 770 →
  ∃ (investment_1 : ℝ),
    interest_rate_1 * investment_1 + interest_rate_2 * investment_2 = total_interest ∧
    investment_1 + investment_2 = 9000 :=
by
  sorry

#check walts_earnings

end walts_earnings_l3581_358161


namespace angle_C_is_pi_over_3_side_c_is_sqrt_6_l3581_358171

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a + t.b = Real.sqrt 3 * t.c ∧
  2 * (Real.sin t.C)^2 = 3 * Real.sin t.A * Real.sin t.B

-- Define the area condition
def hasAreaSqrt3 (t : Triangle) : Prop :=
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3

-- Theorem 1
theorem angle_C_is_pi_over_3 (t : Triangle) 
  (h : satisfiesConditions t) : t.C = π/3 :=
sorry

-- Theorem 2
theorem side_c_is_sqrt_6 (t : Triangle) 
  (h1 : satisfiesConditions t) 
  (h2 : hasAreaSqrt3 t) : t.c = Real.sqrt 6 :=
sorry

end angle_C_is_pi_over_3_side_c_is_sqrt_6_l3581_358171


namespace class_one_is_correct_l3581_358197

/-- Represents the correct way to refer to a numbered class -/
inductive ClassReference
  | CardinalNumber (n : Nat)
  | OrdinalNumber (n : Nat)

/-- Checks if a class reference is correct -/
def is_correct_reference (ref : ClassReference) : Prop :=
  match ref with
  | ClassReference.CardinalNumber n => true
  | ClassReference.OrdinalNumber n => false

/-- The statement that "Class One" is the correct way to refer to the first class -/
theorem class_one_is_correct :
  is_correct_reference (ClassReference.CardinalNumber 1) = true :=
sorry


end class_one_is_correct_l3581_358197


namespace regular_polygon_interior_angle_sum_l3581_358163

/-- Proves that for a regular polygon with an exterior angle of 36°, the sum of its interior angles is 1440°. -/
theorem regular_polygon_interior_angle_sum (n : ℕ) (ext_angle : ℝ) : 
  ext_angle = 36 → 
  n * ext_angle = 360 →
  (n - 2) * 180 = 1440 := by
  sorry

end regular_polygon_interior_angle_sum_l3581_358163


namespace chemistry_marks_proof_l3581_358145

def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks ∧
    chemistry_marks = 67 :=
by sorry

end chemistry_marks_proof_l3581_358145


namespace valid_drawing_probability_l3581_358148

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The number of red balls in the box -/
def red_balls : ℕ := 1

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls + red_balls

/-- The number of valid drawing sequences -/
def valid_sequences : ℕ := 2

/-- The probability of drawing the balls in a valid sequence -/
def probability : ℚ := valid_sequences / (Nat.factorial total_balls / (Nat.factorial white_balls * Nat.factorial black_balls * Nat.factorial red_balls))

theorem valid_drawing_probability : probability = 1 / 231 := by
  sorry

end valid_drawing_probability_l3581_358148


namespace sum_of_perfect_squares_l3581_358181

theorem sum_of_perfect_squares (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m ^ 2) ∧ x + y = 2 * x + 2 * (x.sqrt) + 1 :=
sorry

end sum_of_perfect_squares_l3581_358181


namespace office_officers_count_l3581_358170

/-- Represents the number of officers in an office. -/
def num_officers : ℕ := 15

/-- Represents the number of non-officers in the office. -/
def num_non_officers : ℕ := 525

/-- Represents the average salary of all employees in rupees per month. -/
def avg_salary_all : ℕ := 120

/-- Represents the average salary of officers in rupees per month. -/
def avg_salary_officers : ℕ := 470

/-- Represents the average salary of non-officers in rupees per month. -/
def avg_salary_non_officers : ℕ := 110

/-- Theorem stating that the number of officers is 15, given the conditions. -/
theorem office_officers_count :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / (num_officers + num_non_officers) = avg_salary_all ∧
  num_officers = 15 :=
sorry

end office_officers_count_l3581_358170


namespace smallest_y_absolute_equation_l3581_358141

theorem smallest_y_absolute_equation : 
  let y₁ := -46 / 5
  let y₂ := 64 / 5
  (∀ y : ℚ, |5 * y - 9| = 55 → y ≥ y₁) ∧ 
  |5 * y₁ - 9| = 55 ∧ 
  |5 * y₂ - 9| = 55 ∧
  y₁ < y₂ :=
by sorry

end smallest_y_absolute_equation_l3581_358141


namespace game_theorems_l3581_358138

/-- Game with three possible point values and their probabilities --/
structure Game where
  p : ℝ
  prob_5 : ℝ := 2 * p
  prob_10 : ℝ := p
  prob_20 : ℝ := 1 - 3 * p
  h_p_pos : 0 < p
  h_p_bound : p < 1/3
  h_prob_sum : prob_5 + prob_10 + prob_20 = 1

/-- A round consists of three games --/
def Round := Fin 3 → Game

/-- The probability of total points not exceeding 25 in one round --/
def prob_not_exceed_25 (r : Round) : ℝ := sorry

/-- The expected value of total points in one round --/
def expected_value (r : Round) : ℝ := sorry

theorem game_theorems (r : Round) (h_same_p : ∀ i j : Fin 3, (r i).p = (r j).p) :
  (∃ (p : ℝ), prob_not_exceed_25 r = 26 * p^3) ∧
  (∃ (p : ℝ), p = 1/9 → expected_value r = 140/3) :=
sorry

end game_theorems_l3581_358138


namespace weight_of_replaced_person_l3581_358188

theorem weight_of_replaced_person 
  (n : ℕ) 
  (original_total : ℝ) 
  (new_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : n = 10)
  (h2 : new_weight = 75)
  (h3 : average_increase = 3)
  : 
  (original_total + new_weight - (original_total / n + average_increase * n)) = 45 :=
by sorry

end weight_of_replaced_person_l3581_358188


namespace spinner_final_direction_l3581_358125

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a spinner with its current direction --/
structure Spinner :=
  (direction : Direction)

/-- Calculates the new direction after a given number of quarter turns clockwise --/
def new_direction_after_quarter_turns (initial : Direction) (quarter_turns : Int) : Direction :=
  sorry

/-- Converts revolutions to quarter turns --/
def revolutions_to_quarter_turns (revolutions : Rat) : Int :=
  sorry

theorem spinner_final_direction :
  let initial_spinner := Spinner.mk Direction.South
  let clockwise_turns := revolutions_to_quarter_turns (7/2)
  let counterclockwise_turns := revolutions_to_quarter_turns (9/4)
  let net_turns := clockwise_turns - counterclockwise_turns
  let final_direction := new_direction_after_quarter_turns initial_spinner.direction net_turns
  final_direction = Direction.West := by sorry

end spinner_final_direction_l3581_358125


namespace seventh_alignment_time_l3581_358157

/-- Represents a standard clock with 12 divisions -/
structure Clock :=
  (divisions : Nat)
  (minute_hand_speed : Nat)
  (hour_hand_speed : Nat)

/-- Represents a time in hours and minutes -/
structure Time :=
  (hours : Nat)
  (minutes : Nat)

/-- Calculates the time until the nth alignment of clock hands -/
def time_until_nth_alignment (c : Clock) (start : Time) (n : Nat) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem seventh_alignment_time (c : Clock) (start : Time) :
  c.divisions = 12 →
  c.minute_hand_speed = 12 →
  c.hour_hand_speed = 1 →
  start.hours = 16 →
  start.minutes = 45 →
  time_until_nth_alignment c start 7 = 435 :=
sorry

end seventh_alignment_time_l3581_358157


namespace black_piece_position_l3581_358151

-- Define the structure of a piece
structure Piece :=
  (cubes : Fin 4 → Unit)
  (shape : String)

-- Define the rectangular prism
structure RectangularPrism :=
  (pieces : Fin 4 → Piece)
  (visible : Fin 4 → Bool)
  (bottom_layer : Fin 2 → Piece)

-- Define the positions
inductive Position
  | A | B | C | D

-- Define the properties of the black piece
def is_black_piece (p : Piece) : Prop :=
  p.shape = "T" ∧ 
  (∃ (i : Fin 4), i.val = 3 → p.cubes i = ())

-- Theorem statement
theorem black_piece_position (prism : RectangularPrism) 
  (h1 : ∃ (i : Fin 4), ¬prism.visible i)
  (h2 : ∃ (i : Fin 2), is_black_piece (prism.bottom_layer i))
  (h3 : ∃ (i : Fin 2), prism.bottom_layer i = prism.pieces 3) :
  ∃ (p : Piece), is_black_piece p ∧ p = prism.pieces 2 :=
sorry

end black_piece_position_l3581_358151


namespace quadratic_properties_l3581_358185

/-- A quadratic function passing through given points -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) :
  quadratic_function a b c (-2) = 6 →
  quadratic_function a b c 0 = -4 →
  quadratic_function a b c 1 = -6 →
  quadratic_function a b c 3 = -4 →
  (a > 0 ∧ ∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > (3/2 : ℝ) → quadratic_function a b c x₁ > quadratic_function a b c x₂) :=
by sorry

end quadratic_properties_l3581_358185


namespace redWhiteJellyBeansCount_l3581_358159

/-- Represents the number of jelly beans of each color in one bag -/
structure JellyBeanBag where
  red : ℕ
  black : ℕ
  green : ℕ
  purple : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of red and white jelly beans in the fishbowl -/
def totalRedWhiteInFishbowl (bag : JellyBeanBag) (bagsToFill : ℕ) : ℕ :=
  (bag.red + bag.white) * bagsToFill

/-- Theorem: The total number of red and white jelly beans in the fishbowl is 126 -/
theorem redWhiteJellyBeansCount : 
  let bag : JellyBeanBag := {
    red := 24,
    black := 13,
    green := 36,
    purple := 28,
    yellow := 32,
    white := 18
  }
  let bagsToFill : ℕ := 3
  totalRedWhiteInFishbowl bag bagsToFill = 126 := by
  sorry


end redWhiteJellyBeansCount_l3581_358159


namespace distance_A_to_yoz_l3581_358123

/-- The distance from a point to the yoz plane is the absolute value of its x-coordinate. -/
def distance_to_yoz (p : ℝ × ℝ × ℝ) : ℝ :=
  |p.1|

/-- Point A with coordinates (-3, 1, -4) -/
def A : ℝ × ℝ × ℝ := (-3, 1, -4)

/-- Theorem: The distance from point A to the yoz plane is 3 -/
theorem distance_A_to_yoz : distance_to_yoz A = 3 := by
  sorry

end distance_A_to_yoz_l3581_358123


namespace simplified_expansion_terms_l3581_358155

/-- The number of terms in the simplified expansion of (x+y+z)^2008 + (x-y-z)^2008 -/
def num_terms : ℕ :=
  (Finset.range 1005).card + (Finset.range 1006).card

theorem simplified_expansion_terms :
  num_terms = 505815 :=
sorry

end simplified_expansion_terms_l3581_358155


namespace inverse_variation_problem_l3581_358140

-- Define the inverse variation relationship
def inverse_variation (y z : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y^4 * z^(1/4) = k

-- State the theorem
theorem inverse_variation_problem (y z : ℝ) :
  inverse_variation y z →
  (3 : ℝ)^4 * 16^(1/4) = 6^4 * z^(1/4) →
  z = 1 / 4096 :=
by sorry

end inverse_variation_problem_l3581_358140


namespace divisibility_theorem_l3581_358144

/-- The set of natural numbers m for which 3^m - 1 is divisible by 2^m -/
def S : Set ℕ := {m : ℕ | ∃ k : ℕ, 3^m - 1 = k * 2^m}

/-- The set of natural numbers m for which 31^m - 1 is divisible by 2^m -/
def T : Set ℕ := {m : ℕ | ∃ k : ℕ, 31^m - 1 = k * 2^m}

theorem divisibility_theorem :
  S = {1, 2, 4} ∧ T = {1, 2, 4, 6, 8} := by sorry

end divisibility_theorem_l3581_358144


namespace alvarez_diesel_consumption_l3581_358116

/-- Given that Mr. Alvarez spends $36 on diesel fuel each week and the cost of diesel fuel is $3 per gallon,
    prove that he uses 24 gallons of diesel fuel in two weeks. -/
theorem alvarez_diesel_consumption
  (weekly_expenditure : ℝ)
  (cost_per_gallon : ℝ)
  (h1 : weekly_expenditure = 36)
  (h2 : cost_per_gallon = 3)
  : (weekly_expenditure / cost_per_gallon) * 2 = 24 := by
  sorry

end alvarez_diesel_consumption_l3581_358116


namespace factorization_equality_l3581_358130

theorem factorization_equality (x y : ℝ) :
  (5 * x - 4 * y) * (x + 2 * y) = 5 * x^2 + 6 * x * y - 8 * y^2 := by sorry

end factorization_equality_l3581_358130


namespace lehmer_mean_properties_l3581_358113

noncomputable def A (a b : ℝ) : ℝ := (a + b) / 2

noncomputable def G (a b : ℝ) : ℝ := Real.sqrt (a * b)

noncomputable def L (p a b : ℝ) : ℝ := (a^p + b^p) / (a^(p-1) + b^(p-1))

theorem lehmer_mean_properties (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  L 0.5 a b ≤ A a b ∧
  L 0 a b ≥ G a b ∧
  L 2 a b ≥ L 1 a b ∧
  ∃ n, L (n + 1) a b > L n a b :=
sorry

end lehmer_mean_properties_l3581_358113


namespace geometric_sequence_condition_l3581_358176

/-- For a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ
  h₁ : a₁ > 0

/-- The third term of a geometric sequence -/
def GeometricSequence.a₃ (g : GeometricSequence) : ℝ := g.a₁ * g.q^2

theorem geometric_sequence_condition (g : GeometricSequence) :
  (g.q > 1 → g.a₁ < g.a₃) ∧ 
  ¬(g.a₁ < g.a₃ → g.q > 1) :=
sorry

end geometric_sequence_condition_l3581_358176


namespace condition_necessary_not_sufficient_l3581_358105

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end condition_necessary_not_sufficient_l3581_358105


namespace quadratic_sum_l3581_358136

/-- A quadratic function g(x) = px^2 + qx + r passing through (0, 3) and (2, 3) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := λ x ↦ p * x^2 + q * x + r

theorem quadratic_sum (p q r : ℝ) :
  QuadraticFunction p q r 0 = 3 ∧ QuadraticFunction p q r 2 = 3 →
  p + 2*q + r = 3 := by
sorry

end quadratic_sum_l3581_358136


namespace ellipse_a_value_l3581_358122

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type :=
  (h_pos : 0 < b ∧ b < a)

/-- The foci of an ellipse -/
def foci (e : Ellipse a b) : ℝ × ℝ := sorry

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse a b) : Type :=
  (x y : ℝ)
  (on_ellipse : x^2 / a^2 + y^2 / b^2 = 1)

/-- The area of a triangle formed by a point on the ellipse and the foci -/
def triangle_area (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- The tangent of the angle PF₁F₂ -/
def tan_angle_PF1F2 (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- The tangent of the angle PF₂F₁ -/
def tan_angle_PF2F1 (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- Theorem: If there exists a point P on the ellipse satisfying the given conditions, 
    then the semi-major axis a equals √15/2 -/
theorem ellipse_a_value (a b : ℝ) (e : Ellipse a b) :
  (∃ p : PointOnEllipse e, 
    triangle_area e p = 1 ∧ 
    tan_angle_PF1F2 e p = 1/2 ∧ 
    tan_angle_PF2F1 e p = -2) →
  a = Real.sqrt 15 / 2 := by sorry

end ellipse_a_value_l3581_358122


namespace ellipse_equation_l3581_358139

/-- The standard equation of an ellipse with given foci and a point on the ellipse -/
theorem ellipse_equation (P A B : ℝ × ℝ) (h_P : P = (5/2, -3/2)) (h_A : A = (-2, 0)) (h_B : B = (2, 0)) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 ↔
    (x - A.1)^2 + (y - A.2)^2 + (x - B.1)^2 + (y - B.2)^2 = 4 * a^2 ∧
    (x - P.1)^2 + (y - P.2)^2 = ((x - A.1)^2 + (y - A.2)^2)^(1/2) * ((x - B.1)^2 + (y - B.2)^2)^(1/2)) ∧
  a^2 = 10 ∧ b^2 = 6 :=
by sorry

end ellipse_equation_l3581_358139


namespace union_of_A_and_complement_of_B_l3581_358109

open Set

theorem union_of_A_and_complement_of_B (A B : Set ℝ) : 
  A = {x : ℝ | x^2 - 4*x - 12 < 0} →
  B = {x : ℝ | x < 2} →
  A ∪ (univ \ B) = {x : ℝ | x > -2} := by
sorry

end union_of_A_and_complement_of_B_l3581_358109


namespace largest_product_of_three_l3581_358134

def S : Finset Int := {-5, -4, -1, 6, 7}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z → 
   x * y * z ≤ a * b * c) →
  a * b * c = 140 :=
sorry

end largest_product_of_three_l3581_358134


namespace horner_method_f_3_f_3_equals_328_l3581_358199

/-- Horner's method representation of a polynomial -/
def horner_rep (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

/-- The polynomial f(x) = x^5 + 2x^3 + 3x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_method_f_3 :
  f 3 = horner_rep [1, 0, 2, 3, 1, 1] 3 := by
  sorry

theorem f_3_equals_328 : f 3 = 328 := by
  sorry

end horner_method_f_3_f_3_equals_328_l3581_358199


namespace abcd_inequality_l3581_358195

theorem abcd_inequality (a b c d : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hd : 0 < d ∧ d < 1) 
  (h_prod : a * b * c * d = (1 - a) * (1 - b) * (1 - c) * (1 - d)) : 
  (a + b + c + d) - (a + c) * (b + d) ≥ 1 := by
  sorry

end abcd_inequality_l3581_358195


namespace set_M_value_l3581_358146

def M : Set ℤ := {a | ∃ (n : ℕ+), 6 / (5 - a) = n}

theorem set_M_value : M = {-1, 2, 3, 4} := by
  sorry

end set_M_value_l3581_358146


namespace binomial_expansion_coefficient_l3581_358192

theorem binomial_expansion_coefficient (n : ℕ) : 
  (Nat.choose n 2) * 9 = 54 → n = 4 := by sorry

end binomial_expansion_coefficient_l3581_358192


namespace attendants_using_pen_pen_users_count_l3581_358162

theorem attendants_using_pen (total_pencil : ℕ) (only_one_tool : ℕ) (both_tools : ℕ) : ℕ :=
  let pencil_only := total_pencil - both_tools
  let pen_only := only_one_tool - pencil_only
  pen_only + both_tools

theorem pen_users_count : attendants_using_pen 25 20 10 = 15 := by
  sorry

end attendants_using_pen_pen_users_count_l3581_358162


namespace f_1998_is_zero_l3581_358154

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_1998_is_zero
  (f : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_period : ∀ x, f (x + 3) = -f x) :
  f 1998 = 0 := by
  sorry

end f_1998_is_zero_l3581_358154


namespace complex_equation_solutions_l3581_358179

theorem complex_equation_solutions :
  ∃ (s : Finset ℂ), (∀ z ∈ s, (z^4 + 1) / (z^2 - z - 2) = 0) ∧ s.card = 4 :=
by
  -- We define the numerator and denominator polynomials
  let num := fun (z : ℂ) ↦ z^4 + 1
  let den := fun (z : ℂ) ↦ z^2 - z - 2

  -- We assume the factorizations given in the problem
  have h_num : ∀ z, num z = (z^2 + Real.sqrt 2 * z + 1) * (z^2 - Real.sqrt 2 * z + 1) := by sorry
  have h_den : ∀ z, den z = (z - 2) * (z + 1) := by sorry

  -- The proof would go here
  sorry

end complex_equation_solutions_l3581_358179


namespace hippopotamus_crayons_l3581_358124

/-- The number of crayons eaten by a hippopotamus --/
def crayonsEaten (initial final : ℕ) : ℕ := initial - final

/-- Theorem: The number of crayons eaten by the hippopotamus is the difference between 
    the initial and final number of crayons --/
theorem hippopotamus_crayons (initial final : ℕ) (h : initial ≥ final) :
  crayonsEaten initial final = initial - final := by
  sorry

/-- Given Jane's initial and final crayon counts, calculate how many were eaten --/
def janesCrayons : ℕ := 
  let initial := 87
  let final := 80
  crayonsEaten initial final

#eval janesCrayons  -- Should output 7

end hippopotamus_crayons_l3581_358124


namespace complex_number_problem_l3581_358190

theorem complex_number_problem (z : ℂ) : 
  Complex.abs z = 1 ∧ 
  (∃ (y : ℝ), (3 + 4*I) * z = y * I) → 
  z = Complex.mk (-4/5) (-3/5) ∨ 
  z = Complex.mk (4/5) (3/5) := by
  sorry

end complex_number_problem_l3581_358190


namespace sqrt_sum_greater_than_sqrt_of_sum_l3581_358183

theorem sqrt_sum_greater_than_sqrt_of_sum : Real.sqrt 2 + Real.sqrt 3 > Real.sqrt (2 + 3) := by
  sorry

end sqrt_sum_greater_than_sqrt_of_sum_l3581_358183


namespace least_clock_equivalent_hour_l3581_358119

theorem least_clock_equivalent_hour : ∃ (h : ℕ), 
  h > 3 ∧ 
  (∀ k : ℕ, k > 3 ∧ k < h → ¬(12 ∣ (k^2 - k))) ∧ 
  (12 ∣ (h^2 - h)) :=
by sorry

end least_clock_equivalent_hour_l3581_358119


namespace smaller_part_is_4000_l3581_358184

/-- Represents an investment split into two parts -/
structure Investment where
  total : ℝ
  greater_part : ℝ
  smaller_part : ℝ
  greater_rate : ℝ
  smaller_rate : ℝ

/-- Conditions for the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.total = 10000 ∧
  i.greater_part + i.smaller_part = i.total ∧
  i.greater_rate = 0.06 ∧
  i.smaller_rate = 0.05 ∧
  i.greater_rate * i.greater_part = i.smaller_rate * i.smaller_part + 160

/-- Theorem stating that under the given conditions, the smaller part of the investment is 4000 -/
theorem smaller_part_is_4000 (i : Investment) 
  (h : investment_conditions i) : i.smaller_part = 4000 := by
  sorry

end smaller_part_is_4000_l3581_358184


namespace complex_distance_sum_l3581_358189

/-- Given a complex number z satisfying |z - 3 - 2i| = 7, 
    prove that |z - 2 + i|^2 + |z - 11 - 5i|^2 = 554 -/
theorem complex_distance_sum (z : ℂ) (h : Complex.abs (z - (3 + 2*I)) = 7) : 
  (Complex.abs (z - (2 - I)))^2 + (Complex.abs (z - (11 + 5*I)))^2 = 554 := by
  sorry

end complex_distance_sum_l3581_358189


namespace largest_fraction_l3581_358175

theorem largest_fraction :
  let a := (8 + 5) / 3
  let b := 8 / (3 + 5)
  let c := (3 + 5) / 8
  let d := (8 + 3) / 5
  let e := 3 / (8 + 5)
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) :=
by sorry

end largest_fraction_l3581_358175


namespace polynomial_product_equality_l3581_358165

theorem polynomial_product_equality (x : ℝ) : 
  (1 + x^3) * (1 - 2*x + x^4) = 1 - 2*x + x^3 - x^4 + x^7 := by
  sorry

end polynomial_product_equality_l3581_358165


namespace eliminate_denominators_l3581_358111

theorem eliminate_denominators (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) :
  (3 / (2 * x) = 1 / (x - 1)) ↔ (3 * x - 3 = 2 * x) :=
sorry

end eliminate_denominators_l3581_358111


namespace negation_equivalence_l3581_358126

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end negation_equivalence_l3581_358126


namespace compute_expression_l3581_358133

theorem compute_expression : 20 * ((150 / 5) - (40 / 8) + (16 / 32) + 3) = 570 := by
  sorry

end compute_expression_l3581_358133


namespace S_is_line_l3581_358112

-- Define the set of points satisfying the equation
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 5 * Real.sqrt ((p.1 - 1)^2 + (p.2 - 2)^2) = |3 * p.1 + 4 * p.2 - 11|}

-- Theorem stating that S is a line
theorem S_is_line : ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ S = {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0} :=
sorry

end S_is_line_l3581_358112


namespace rectangle_length_given_perimeter_and_breadth_l3581_358106

/-- The perimeter of a rectangle given its length and breadth -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangular garden with perimeter 500 m and breadth 100 m, the length is 150 m -/
theorem rectangle_length_given_perimeter_and_breadth :
  ∀ length : ℝ, rectanglePerimeter length 100 = 500 → length = 150 := by
sorry

end rectangle_length_given_perimeter_and_breadth_l3581_358106


namespace rook_placement_exists_l3581_358160

/-- Represents a chessboard with rook placements -/
structure Chessboard (n : ℕ) :=
  (placements : Fin n → Fin n)
  (colors : Fin n → Fin n → Fin (n^2 / 2))

/-- Predicate to check if rook placements are valid -/
def valid_placements (n : ℕ) (board : Chessboard n) : Prop :=
  (∀ i j : Fin n, i ≠ j → board.placements i ≠ board.placements j) ∧
  (∀ i j : Fin n, i ≠ j → board.colors i (board.placements i) ≠ board.colors j (board.placements j))

/-- Predicate to check if the coloring is valid -/
def valid_coloring (n : ℕ) (board : Chessboard n) : Prop :=
  ∀ c : Fin (n^2 / 2), ∃! (i j k l : Fin n), 
    (i, j) ≠ (k, l) ∧ board.colors i j = c ∧ board.colors k l = c

/-- Main theorem -/
theorem rook_placement_exists (n : ℕ) (h_even : Even n) (h_gt_2 : n > 2) :
  ∃ (board : Chessboard n), valid_placements n board ∧ valid_coloring n board :=
sorry

end rook_placement_exists_l3581_358160


namespace fixed_point_theorem_dot_product_range_l3581_358129

-- Define the curves and line
def curve_C (x y : ℝ) : Prop := y^2 = 4*x
def curve_M (x y : ℝ) : Prop := (x-1)^2 + y^2 = 4 ∧ x ≥ 1
def line_l (m n x y : ℝ) : Prop := x = m*y + n

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

-- Part I
theorem fixed_point_theorem (m n x1 y1 x2 y2 : ℝ) :
  curve_C x1 y1 ∧ curve_C x2 y2 ∧
  line_l m n x1 y1 ∧ line_l m n x2 y2 ∧
  dot_product x1 y1 x2 y2 = -4 →
  n = 2 :=
sorry

-- Part II
theorem dot_product_range (m n x1 y1 x2 y2 : ℝ) :
  curve_C x1 y1 ∧ curve_C x2 y2 ∧
  line_l m n x1 y1 ∧ line_l m n x2 y2 ∧
  curve_M 1 0 ∧
  (∀ x y, curve_M x y → ¬(line_l m n x y ∧ (x, y) ≠ (1, 0))) →
  dot_product (x1-1) y1 (x2-1) y2 ≤ -8 :=
sorry

end fixed_point_theorem_dot_product_range_l3581_358129


namespace two_digit_number_problem_l3581_358167

theorem two_digit_number_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧
  ∃ k : ℤ, 3 * n - 4 = 10 * k ∧
  60 < 4 * n - 15 ∧ 4 * n - 15 < 100 ∧
  n = 28 :=
sorry

end two_digit_number_problem_l3581_358167


namespace sum_of_real_and_imaginary_parts_of_one_plus_i_squared_l3581_358166

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_real_and_imaginary_parts_of_one_plus_i_squared (a b : ℝ) : 
  (1 + i)^2 = a + b * i → a + b = 2 := by sorry

end sum_of_real_and_imaginary_parts_of_one_plus_i_squared_l3581_358166


namespace range_of_a_l3581_358128

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2 ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by sorry

end range_of_a_l3581_358128


namespace jones_clothes_count_l3581_358101

/-- Represents the ratio of shirts to pants -/
def shirt_to_pants_ratio : ℕ := 6

/-- Represents the number of pants Mr. Jones owns -/
def pants_count : ℕ := 40

/-- Calculates the total number of clothes Mr. Jones owns -/
def total_clothes : ℕ := shirt_to_pants_ratio * pants_count + pants_count

/-- Proves that the total number of clothes Mr. Jones owns is 280 -/
theorem jones_clothes_count : total_clothes = 280 := by
  sorry

end jones_clothes_count_l3581_358101


namespace intersection_midpoint_l3581_358121

/-- Given a straight line x - y = 2 intersecting a parabola y² = 4x at points A and B,
    the midpoint M of line segment AB has coordinates (4, 2). -/
theorem intersection_midpoint (A B M : ℝ × ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    A = (x₁, y₁) ∧ B = (x₂, y₂) ∧
    x₁ - y₁ = 2 ∧ x₂ - y₂ = 2 ∧
    y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂ ∧
    M = ((x₁ + x₂)/2, (y₁ + y₂)/2)) →
  M = (4, 2) := by
sorry


end intersection_midpoint_l3581_358121


namespace cot_thirty_degrees_l3581_358198

theorem cot_thirty_degrees : 
  let θ : Real := 30 * π / 180 -- Convert 30 degrees to radians
  let cot (x : Real) := 1 / Real.tan x -- Definition of cotangent
  (Real.tan θ = 1 / Real.sqrt 3) → -- Given condition
  (cot θ = Real.sqrt 3) := by
sorry

end cot_thirty_degrees_l3581_358198


namespace second_group_women_l3581_358172

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of women in the second group -/
def x : ℕ := sorry

/-- The work rate of 3 men and 8 women equals the work rate of 6 men and x women -/
axiom work_rate_equality : 3 * man_rate + 8 * woman_rate = 6 * man_rate + x * woman_rate

/-- The work rate of 4 men and 5 women is 0.9285714285714286 times the work rate of 3 men and 8 women -/
axiom work_rate_fraction : 4 * man_rate + 5 * woman_rate = 0.9285714285714286 * (3 * man_rate + 8 * woman_rate)

/-- The number of women in the second group is 14 -/
theorem second_group_women : x = 14 := by sorry

end second_group_women_l3581_358172


namespace spider_count_l3581_358147

theorem spider_count (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 40) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 5 := by
  sorry

end spider_count_l3581_358147


namespace diagonal_length_l3581_358173

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = FG = 12
  (ex - fx)^2 + (ey - fy)^2 = 12^2 ∧
  (fx - gx)^2 + (fy - gy)^2 = 12^2 ∧
  -- GH = HE = 20
  (gx - hx)^2 + (gy - hy)^2 = 20^2 ∧
  (hx - ex)^2 + (hy - ey)^2 = 20^2 ∧
  -- Angle GHE = 90°
  (gx - hx) * (ex - hx) + (gy - hy) * (ey - hy) = 0

-- Theorem statement
theorem diagonal_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (gx, gy) := q.G
  (ex - gx)^2 + (ey - gy)^2 = 2 * 20^2 := by
  sorry

end diagonal_length_l3581_358173


namespace rhombus_perimeter_l3581_358191

/-- The perimeter of a rhombus given the lengths of its diagonals -/
theorem rhombus_perimeter (d1 d2 θ : ℝ) (h1 : d1 > 0) (h2 : d2 > 0) (h3 : 0 < θ ∧ θ < π) :
  ∃ (P : ℝ), P = 2 * Real.sqrt (d1^2 + d2^2) ∧ P > 0 := by
  sorry

end rhombus_perimeter_l3581_358191


namespace correct_employee_count_l3581_358186

/-- The number of employees in Kim's office -/
def num_employees : ℕ := 9

/-- The total time Kim spends on her morning routine in minutes -/
def total_time : ℕ := 50

/-- The time Kim spends making coffee in minutes -/
def coffee_time : ℕ := 5

/-- The time Kim spends per employee for status update in minutes -/
def status_update_time : ℕ := 2

/-- The time Kim spends per employee for payroll update in minutes -/
def payroll_update_time : ℕ := 3

/-- Theorem stating that the number of employees is correct given the conditions -/
theorem correct_employee_count :
  num_employees * (status_update_time + payroll_update_time) + coffee_time = total_time :=
by sorry

end correct_employee_count_l3581_358186


namespace train_speed_calculation_l3581_358127

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 25) :
  (train_length + bridge_length) / crossing_time = 16 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l3581_358127


namespace rebus_solution_l3581_358132

theorem rebus_solution : ∃! (K I S : Nat), 
  K < 10 ∧ I < 10 ∧ S < 10 ∧
  K ≠ I ∧ K ≠ S ∧ I ≠ S ∧
  100 * K + 10 * I + S + 100 * K + 10 * S + I = 100 * I + 10 * S + K := by
  sorry

end rebus_solution_l3581_358132


namespace investment_problem_l3581_358131

/-- Proves that the initial investment is $698 given the conditions of Peter and David's investments -/
theorem investment_problem (peter_amount : ℝ) (david_amount : ℝ) (peter_years : ℝ) (david_years : ℝ)
  (h_peter : peter_amount = 815)
  (h_david : david_amount = 854)
  (h_peter_years : peter_years = 3)
  (h_david_years : david_years = 4)
  (h_same_principal : ∃ (P : ℝ), P > 0 ∧ 
    ∃ (r : ℝ), r > 0 ∧ 
      peter_amount = P * (1 + r * peter_years) ∧
      david_amount = P * (1 + r * david_years)) :
  ∃ (P : ℝ), P = 698 ∧ 
    ∃ (r : ℝ), r > 0 ∧ 
      peter_amount = P * (1 + r * peter_years) ∧
      david_amount = P * (1 + r * david_years) :=
sorry


end investment_problem_l3581_358131


namespace three_card_picks_count_l3581_358187

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- The number of ways to pick three different cards from a standard deck where order matters -/
def threeCardPicks (d : Deck) : ℕ :=
  52 * 51 * 50

/-- Theorem stating that the number of ways to pick three different cards from a standard 
    52-card deck, where order matters, is equal to 132600 -/
theorem three_card_picks_count (d : Deck) : threeCardPicks d = 132600 := by
  sorry

end three_card_picks_count_l3581_358187


namespace swap_positions_l3581_358100

/-- Represents the color of a checker -/
inductive Color
| Black
| White

/-- Represents a move in the game -/
structure Move where
  color : Color
  count : Nat

/-- Represents the state of the game -/
structure GameState where
  n : Nat
  blackPositions : List Nat
  whitePositions : List Nat

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move.color with
  | Color.Black => move.count ≤ state.n ∧ move.count > 0
  | Color.White => move.count ≤ state.n ∧ move.count > 0

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Generates the sequence of moves for the game -/
def generateMoves (n : Nat) : List Move :=
  sorry

/-- Checks if the final state has swapped positions -/
def isSwappedState (initialState : GameState) (finalState : GameState) : Prop :=
  sorry

/-- Theorem stating that the generated moves will swap the positions -/
theorem swap_positions (n : Nat) :
  let initialState : GameState := { n := n, blackPositions := List.range n, whitePositions := List.range n |>.map (λ x => 2*n - x) }
  let moves := generateMoves n
  let finalState := moves.foldl applyMove initialState
  (∀ move ∈ moves, isValidMove initialState move) ∧
  isSwappedState initialState finalState :=
sorry

end swap_positions_l3581_358100


namespace fourth_power_difference_not_prime_l3581_358118

theorem fourth_power_difference_not_prime (p q : ℕ) (hp : Prime p) (hq : Prime q) (hne : p ≠ q) :
  ¬ Prime (p^4 - q^4) := by
  sorry

end fourth_power_difference_not_prime_l3581_358118
