import Mathlib

namespace ellipse_eccentricity_from_max_ratio_l3113_311393

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity_from_max_ratio (e : Ellipse) :
  (∀ p : ℝ × ℝ, ∃ q : ℝ × ℝ, distance p e.F₁ / distance p e.F₂ ≤ distance q e.F₁ / distance q e.F₂) →
  (∃ p : ℝ × ℝ, distance p e.F₁ / distance p e.F₂ = 3) →
  eccentricity e = 1/2 := by
  sorry

end ellipse_eccentricity_from_max_ratio_l3113_311393


namespace no_real_solution_l3113_311399

theorem no_real_solution :
  ¬∃ (x : ℝ), 4 * (3 * x)^2 + (3 * x) + 3 = 2 * (9 * x^2 + (3 * x) + 1) := by
  sorry

end no_real_solution_l3113_311399


namespace systemC_is_linear_l3113_311319

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationInTwoVars (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- A system of two equations in two variables -/
structure SystemOfTwoEquations :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)

/-- Definition of the specific system of equations given in Option C -/
def systemC : SystemOfTwoEquations :=
  { eq1 := λ x y => x - 3,
    eq2 := λ x y => 2 * x - y - 7 }

/-- Theorem stating that the given system is a system of linear equations in two variables -/
theorem systemC_is_linear : 
  IsLinearEquationInTwoVars systemC.eq1 ∧ IsLinearEquationInTwoVars systemC.eq2 :=
sorry

end systemC_is_linear_l3113_311319


namespace inequality_solutions_l3113_311331

theorem inequality_solutions :
  (∀ x : ℝ, 2*x - 1 > x - 3 ↔ x > -2) ∧
  (∀ x : ℝ, x - 3*(x - 2) ≥ 4 ∧ (x - 1)/5 < (x + 1)/2 ↔ -7/3 < x ∧ x ≤ 1) :=
by sorry

end inequality_solutions_l3113_311331


namespace two_part_journey_average_speed_l3113_311309

/-- Calculates the average speed for a two-part journey -/
theorem two_part_journey_average_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (second_part_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : first_part_distance = 12)
  (h3 : first_part_speed = 24)
  (h4 : second_part_speed = 48)
  : (total_distance / (first_part_distance / first_part_speed +
     (total_distance - first_part_distance) / second_part_speed)) = 40 := by
  sorry

#check two_part_journey_average_speed

end two_part_journey_average_speed_l3113_311309


namespace no_separable_representation_l3113_311387

theorem no_separable_representation : ¬∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), 1 + x^2016 * y^2016 = f x * g y := by
  sorry

end no_separable_representation_l3113_311387


namespace sarah_DC_probability_l3113_311383

def probability_DC : ℚ := 2/5

theorem sarah_DC_probability :
  let p : ℚ → ℚ := λ x => 1/3 + 1/6 * x
  ∃! x : ℚ, x = p x ∧ x = probability_DC :=
by sorry

end sarah_DC_probability_l3113_311383


namespace regular_ngon_minimal_l3113_311308

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an n-gon
structure NGon where
  n : ℕ
  vertices : List (ℝ × ℝ)

-- Function to check if an n-gon is inscribed in a circle
def isInscribed (ngon : NGon) (circle : Circle) : Prop :=
  ngon.vertices.length = ngon.n ∧
  ∀ v ∈ ngon.vertices, (v.1 - circle.center.1)^2 + (v.2 - circle.center.2)^2 = circle.radius^2

-- Function to check if an n-gon is regular
def isRegular (ngon : NGon) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ v ∈ ngon.vertices, (v.1 - center.1)^2 + (v.2 - center.2)^2 = radius^2

-- Function to calculate the area of an n-gon
noncomputable def area (ngon : NGon) : ℝ := sorry

-- Function to calculate the perimeter of an n-gon
noncomputable def perimeter (ngon : NGon) : ℝ := sorry

-- Theorem statement
theorem regular_ngon_minimal (circle : Circle) (n : ℕ) :
  ∀ (ngon : NGon), 
    ngon.n = n → 
    isInscribed ngon circle → 
    ∃ (regular_ngon : NGon), 
      regular_ngon.n = n ∧ 
      isInscribed regular_ngon circle ∧ 
      isRegular regular_ngon ∧ 
      area regular_ngon ≤ area ngon ∧ 
      perimeter regular_ngon ≤ perimeter ngon :=
by sorry

end regular_ngon_minimal_l3113_311308


namespace problem_equivalence_l3113_311358

theorem problem_equivalence (y x : ℝ) (h : x ≠ -1) :
  (y + 2) * (y - 2) + (y - 1) * (y + 3) = 2 * y^2 + 2 * y - 7 ∧
  (1 + 2 / (x + 1)) / ((x^2 + 6*x + 9) / (x + 1)) = 1 / (x + 3) := by
  sorry

end problem_equivalence_l3113_311358


namespace cat_count_correct_l3113_311300

/-- The number of cats that can meow -/
def meow : ℕ := 70

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can fetch -/
def fetch : ℕ := 30

/-- The number of cats that can roll -/
def roll : ℕ := 50

/-- The number of cats that can meow and jump -/
def meow_jump : ℕ := 25

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 15

/-- The number of cats that can fetch and roll -/
def fetch_roll : ℕ := 20

/-- The number of cats that can meow and roll -/
def meow_roll : ℕ := 28

/-- The number of cats that can meow, jump, and fetch -/
def meow_jump_fetch : ℕ := 5

/-- The number of cats that can jump, fetch, and roll -/
def jump_fetch_roll : ℕ := 10

/-- The number of cats that can fetch, roll, and meow -/
def fetch_roll_meow : ℕ := 12

/-- The number of cats that can do all four tricks -/
def all_four : ℕ := 8

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 12

/-- The total number of cats in the studio -/
def total_cats : ℕ := 129

theorem cat_count_correct : 
  total_cats = meow + jump + fetch + roll - meow_jump - jump_fetch - fetch_roll - meow_roll + 
               meow_jump_fetch + jump_fetch_roll + fetch_roll_meow - 2 * all_four + no_tricks := by
  sorry

end cat_count_correct_l3113_311300


namespace locus_of_angle_bisector_intersection_l3113_311352

-- Define the points and constants
variable (a : ℝ) -- distance between A and B
variable (x₀ y₀ : ℝ) -- coordinates of point C
variable (x y : ℝ) -- coordinates of point P

-- State the theorem
theorem locus_of_angle_bisector_intersection 
  (h1 : a > 0) -- A and B are distinct points
  (h2 : x₀^2 + y₀^2 = 1) -- C is on the unit circle centered at A
  (h3 : x = (a * x₀) / (1 + a)) -- x-coordinate of P
  (h4 : y = (a * y₀) / (1 + a)) -- y-coordinate of P
  : x^2 + y^2 = (a^2) / ((1 + a)^2) := by
  sorry

end locus_of_angle_bisector_intersection_l3113_311352


namespace rhombus_perimeter_l3113_311372

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 10 feet is 52 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end rhombus_perimeter_l3113_311372


namespace ascending_order_for_negative_x_l3113_311351

theorem ascending_order_for_negative_x (x : ℝ) (h : -1 < x ∧ x < 0) : 
  5 * x < 0.5 * x ∧ 0.5 * x < 5 - x := by
  sorry

end ascending_order_for_negative_x_l3113_311351


namespace multiple_of_reciprocal_l3113_311343

theorem multiple_of_reciprocal (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 3) (h3 : x + 17 = k * (1 / x)) : k = 60 := by
  sorry

end multiple_of_reciprocal_l3113_311343


namespace dinner_cakes_count_l3113_311337

def lunch_cakes : ℕ := 6
def dinner_difference : ℕ := 3

theorem dinner_cakes_count : lunch_cakes + dinner_difference = 9 := by
  sorry

end dinner_cakes_count_l3113_311337


namespace encyclopedia_total_pages_l3113_311380

/-- Represents a chapter in the encyclopedia -/
structure Chapter where
  main_pages : ℕ
  sub_chapters : ℕ
  sub_chapter_pages : ℕ

/-- The encyclopedia with 12 chapters -/
def encyclopedia : Vector Chapter 12 :=
  Vector.ofFn fun i =>
    match i with
    | 0 => ⟨450, 3, 90⟩
    | 1 => ⟨650, 5, 68⟩
    | 2 => ⟨712, 4, 75⟩
    | 3 => ⟨820, 6, 120⟩
    | 4 => ⟨530, 2, 110⟩
    | 5 => ⟨900, 7, 95⟩
    | 6 => ⟨680, 4, 80⟩
    | 7 => ⟨555, 3, 180⟩
    | 8 => ⟨990, 5, 53⟩
    | 9 => ⟨825, 6, 150⟩
    | 10 => ⟨410, 2, 200⟩
    | 11 => ⟨1014, 7, 69⟩

/-- Total pages in a chapter -/
def total_pages_in_chapter (c : Chapter) : ℕ :=
  c.main_pages + c.sub_chapters * c.sub_chapter_pages

/-- Total pages in the encyclopedia -/
def total_pages : ℕ :=
  (encyclopedia.toList.map total_pages_in_chapter).sum

theorem encyclopedia_total_pages :
  total_pages = 13659 := by sorry

end encyclopedia_total_pages_l3113_311380


namespace range_of_f_l3113_311312

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -2} :=
by
  sorry

end range_of_f_l3113_311312


namespace original_number_proof_l3113_311328

theorem original_number_proof (y : ℝ) (h : 1 - 1/y = 1/5) : y = 5/4 := by
  sorry

end original_number_proof_l3113_311328


namespace triangle_inequality_l3113_311395

theorem triangle_inequality (a b c S : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S > 0)
  (h_triangle : S = Real.sqrt (((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 16)) :
  4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ a^2 + b^2 + c^2 ∧
  a^2 + b^2 + c^2 ≤ 4 * Real.sqrt 3 * S + 3 * ((a - b)^2 + (b - c)^2 + (c - a)^2) := by
  sorry

end triangle_inequality_l3113_311395


namespace binary_1011011_equals_base7_160_l3113_311376

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its representation in base 7. -/
def nat_to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: nat_to_base7 (n / 7)

/-- The binary representation of 1011011. -/
def binary_1011011 : List Bool :=
  [true, false, true, true, false, true, true]

/-- The base 7 representation of 160. -/
def base7_160 : List ℕ :=
  [0, 6, 1]

theorem binary_1011011_equals_base7_160 :
  nat_to_base7 (binary_to_nat binary_1011011) = base7_160 := by
  sorry

#eval binary_to_nat binary_1011011
#eval nat_to_base7 (binary_to_nat binary_1011011)

end binary_1011011_equals_base7_160_l3113_311376


namespace binomial_probability_one_l3113_311327

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability of a binomial random variable taking a specific value -/
def probability (X : BinomialRV) (k : ℕ) : ℝ :=
  (X.n.choose k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

theorem binomial_probability_one (η : BinomialRV) 
  (h_p : η.p = 0.6) 
  (h_expectation : expectation η = 3) :
  probability η 1 = 3 * 0.4^4 := by
  sorry

end binomial_probability_one_l3113_311327


namespace total_distance_jogged_l3113_311353

/-- The total distance jogged by Kyle and Sarah -/
def total_distance (inner_track_length outer_track_length : ℝ)
  (kyle_inner_laps kyle_outer_laps : ℝ)
  (sarah_inner_laps sarah_outer_laps : ℝ) : ℝ :=
  (kyle_inner_laps * inner_track_length + kyle_outer_laps * outer_track_length) +
  (sarah_inner_laps * inner_track_length + sarah_outer_laps * outer_track_length)

/-- Theorem stating the total distance jogged by Kyle and Sarah -/
theorem total_distance_jogged :
  total_distance 250 400 1.12 1.78 2.73 1.36 = 2218.5 := by
  sorry

end total_distance_jogged_l3113_311353


namespace citrus_grove_total_orchards_l3113_311398

/-- Represents the number of orchards for each fruit type and the total -/
structure CitrusGrove where
  lemons : ℕ
  oranges : ℕ
  grapefruits : ℕ
  limes : ℕ
  total : ℕ

/-- Theorem stating the total number of orchards in the citrus grove -/
theorem citrus_grove_total_orchards (cg : CitrusGrove) : cg.total = 16 :=
  by
  have h1 : cg.lemons = 8 := by sorry
  have h2 : cg.oranges = cg.lemons / 2 := by sorry
  have h3 : cg.grapefruits = 2 := by sorry
  have h4 : cg.limes + cg.grapefruits = cg.total - (cg.lemons + cg.oranges) := by sorry
  sorry

#check citrus_grove_total_orchards

end citrus_grove_total_orchards_l3113_311398


namespace arithmetic_evaluation_l3113_311339

theorem arithmetic_evaluation : 8 + 15 / 3 * 2 - 5 + 4 = 17 := by
  sorry

end arithmetic_evaluation_l3113_311339


namespace hexagon_perimeter_l3113_311364

/-- The perimeter of a hexagon with side lengths in arithmetic sequence -/
theorem hexagon_perimeter (a b c d e f : ℕ) (h1 : b = a + 2) (h2 : c = b + 2) (h3 : d = c + 2) 
  (h4 : e = d + 2) (h5 : f = e + 2) (h6 : a = 10) : a + b + c + d + e + f = 90 :=
by sorry

end hexagon_perimeter_l3113_311364


namespace expression_evaluation_l3113_311360

theorem expression_evaluation (x y : ℚ) (hx : x = -1/2) (hy : y = 2022) :
  ((2*x - y)^2 - (2*x + y)*(2*x - y)) / (2*y) = 2023 := by
  sorry

end expression_evaluation_l3113_311360


namespace quadratic_inequality_solution_set_l3113_311303

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 3*x - 4 < 0} = Set.Ioo (-4 : ℝ) 1 := by sorry

end quadratic_inequality_solution_set_l3113_311303


namespace greatest_x_value_l3113_311329

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 220000) :
  x ≤ 5 ∧ (2.134 : ℝ) * (10 : ℝ) ^ (5 : ℝ) < 220000 :=
sorry

end greatest_x_value_l3113_311329


namespace complex_division_simplification_l3113_311333

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_division_simplification :
  (2 + i) / i = 1 - 2*i :=
by sorry

end complex_division_simplification_l3113_311333


namespace polynomial_factorization_l3113_311306

theorem polynomial_factorization :
  (∀ x : ℝ, x^2 + 14*x + 49 = (x + 7)^2) ∧
  (∀ m n : ℝ, (m - 1) + n^2*(1 - m) = (m - 1)*(1 - n)*(1 + n)) := by
  sorry

end polynomial_factorization_l3113_311306


namespace coinciding_rest_days_l3113_311349

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 7

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 5

/-- Total number of days -/
def total_days : ℕ := 1200

/-- Number of rest days Al has in one cycle -/
def al_rest_days : ℕ := 2

/-- Number of rest days Barb has in one cycle -/
def barb_rest_days : ℕ := 1

/-- The theorem to prove -/
theorem coinciding_rest_days : 
  ∃ (n : ℕ), n = (total_days / (Nat.lcm al_cycle barb_cycle)) * 
    (al_rest_days * barb_rest_days) ∧ n = 34 := by
  sorry

end coinciding_rest_days_l3113_311349


namespace fraction_equation_solution_l3113_311315

theorem fraction_equation_solution :
  ∃ (x : ℚ), (x + 7) / (x - 4) = (x - 3) / (x + 6) ∧ x = -3/2 := by
  sorry

end fraction_equation_solution_l3113_311315


namespace reservoir_capacity_l3113_311363

theorem reservoir_capacity (x : ℝ) 
  (h1 : x > 0) -- Ensure the capacity is positive
  (h2 : (1/4) * x + 100 = (3/8) * x) -- Condition from initial state to final state
  : x = 800 := by
  sorry

end reservoir_capacity_l3113_311363


namespace pyramid_volume_in_cone_l3113_311347

/-- The volume of a pyramid inscribed in a cone, where the pyramid's base is an isosceles triangle -/
theorem pyramid_volume_in_cone (V : ℝ) (α : ℝ) :
  let cone_volume := V
  let base_angle := α
  let pyramid_volume := (2 * V / Real.pi) * Real.sin α * (Real.cos (α / 2))^2
  0 < V → 0 < α → α < π →
  pyramid_volume = (2 * cone_volume / Real.pi) * Real.sin base_angle * (Real.cos (base_angle / 2))^2 :=
by sorry

end pyramid_volume_in_cone_l3113_311347


namespace min_value_exponential_sum_l3113_311390

theorem min_value_exponential_sum (x y : ℝ) (h : x + y = 5) :
  3^x + 3^y ≥ 18 * Real.sqrt 3 := by
  sorry

end min_value_exponential_sum_l3113_311390


namespace cubic_polynomial_root_l3113_311342

theorem cubic_polynomial_root (d e : ℚ) :
  (3 - Real.sqrt 5 : ℂ) ^ 3 + d * (3 - Real.sqrt 5 : ℂ) + e = 0 →
  (-6 : ℂ) ^ 3 + d * (-6 : ℂ) + e = 0 := by
  sorry

end cubic_polynomial_root_l3113_311342


namespace nearest_integer_to_x_plus_2y_l3113_311370

theorem nearest_integer_to_x_plus_2y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + 2 * y = 6) (h2 : |x| * y + x^3 = 2) :
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |x + 2 * y - ↑n| ≤ |x + 2 * y - ↑m| :=
sorry

end nearest_integer_to_x_plus_2y_l3113_311370


namespace at_least_one_hits_target_l3113_311311

theorem at_least_one_hits_target (p_both : ℝ) (h : p_both = 0.6) :
  1 - (1 - p_both) * (1 - p_both) = 0.84 := by
  sorry

end at_least_one_hits_target_l3113_311311


namespace arithmetic_sequence_sum_l3113_311350

/-- The sum of an arithmetic sequence with first term a₁ = k^2 - k + 1 and common difference d = 2 for k^2 terms -/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ : ℤ := k^2 - k + 1
  let d : ℤ := 2
  let n : ℕ := k^2
  let Sₙ : ℤ := n * (2 * a₁ + (n - 1) * d) / 2
  Sₙ = 2 * k^4 - k^3 := by
sorry

end arithmetic_sequence_sum_l3113_311350


namespace g_of_2_eq_1_l3113_311396

/-- The function that keeps only the last k digits of a number --/
def lastKDigits (n : ℕ) (k : ℕ) : ℕ :=
  n % (10^k)

/-- The sequence of numbers on the board for a given k --/
def boardSequence (k : ℕ) : List ℕ :=
  sorry

/-- The function g(k) as defined in the problem --/
def g (k : ℕ) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem g_of_2_eq_1 : g 2 = 1 := by
  sorry

end g_of_2_eq_1_l3113_311396


namespace theodore_tax_rate_l3113_311397

/-- Calculates the tax rate for Theodore's statue business --/
theorem theodore_tax_rate :
  let stone_statues : ℕ := 10
  let wooden_statues : ℕ := 20
  let stone_price : ℚ := 20
  let wooden_price : ℚ := 5
  let after_tax_earnings : ℚ := 270
  let before_tax_earnings := stone_statues * stone_price + wooden_statues * wooden_price
  let tax_rate := (before_tax_earnings - after_tax_earnings) / before_tax_earnings
  tax_rate = 1/10 := by sorry

end theodore_tax_rate_l3113_311397


namespace simplify_expression_l3113_311338

theorem simplify_expression (x : ℝ) : x - 2*(1+x) + 3*(1-x) - 4*(1+2*x) = -12*x - 3 := by
  sorry

end simplify_expression_l3113_311338


namespace no_cracked_seashells_l3113_311301

theorem no_cracked_seashells (tom_shells fred_shells total_shells : ℕ) 
  (h1 : tom_shells = 15)
  (h2 : fred_shells = 43)
  (h3 : total_shells = 58)
  (h4 : tom_shells + fred_shells = total_shells) :
  total_shells - (tom_shells + fred_shells) = 0 :=
by sorry

end no_cracked_seashells_l3113_311301


namespace power_function_not_through_origin_l3113_311389

-- Define the power function
def f (m : ℕ+) (x : ℝ) : ℝ := x^(m.val - 2)

-- Theorem statement
theorem power_function_not_through_origin (m : ℕ+) :
  (∀ x ≠ 0, f m x ≠ 0) → m = 1 ∨ m = 2 := by
  sorry

end power_function_not_through_origin_l3113_311389


namespace complex_modulus_implies_real_value_l3113_311307

theorem complex_modulus_implies_real_value (a : ℝ) : 
  Complex.abs ((a + 2 * Complex.I) * (1 + Complex.I)) = 4 → a = 2 ∨ a = -2 := by
  sorry

end complex_modulus_implies_real_value_l3113_311307


namespace distance_between_locations_l3113_311384

/-- Represents a car with its speed -/
structure Car where
  speed : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  carA : Car
  carB : Car
  meetingTime : ℝ
  additionalTime : ℝ
  finalDistanceA : ℝ
  finalDistanceB : ℝ

/-- The theorem stating the distance between locations A and B -/
theorem distance_between_locations (setup : ProblemSetup)
  (h1 : setup.meetingTime = 5)
  (h2 : setup.additionalTime = 3)
  (h3 : setup.finalDistanceA = 130)
  (h4 : setup.finalDistanceB = 160) :
  setup.carA.speed * setup.meetingTime + setup.carB.speed * setup.meetingTime = 290 :=
by sorry

end distance_between_locations_l3113_311384


namespace unique_solution_quadratic_inequality_l3113_311346

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 + a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
sorry

end unique_solution_quadratic_inequality_l3113_311346


namespace dot_product_of_vectors_l3113_311361

theorem dot_product_of_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (1, -3)) 
  (h2 : a - b = (3, 7)) : 
  a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end dot_product_of_vectors_l3113_311361


namespace meadow_trees_l3113_311392

/-- Represents the number of trees around the meadow. -/
def num_trees : ℕ := sorry

/-- Represents Serezha's count of a specific tree. -/
def serezha_count1 : ℕ := 20

/-- Represents Misha's count of the same tree as serezha_count1. -/
def misha_count1 : ℕ := 7

/-- Represents Serezha's count of another specific tree. -/
def serezha_count2 : ℕ := 7

/-- Represents Misha's count of the same tree as serezha_count2. -/
def misha_count2 : ℕ := 94

/-- The theorem stating that the number of trees around the meadow is 100. -/
theorem meadow_trees : num_trees = 100 := by sorry

end meadow_trees_l3113_311392


namespace min_value_of_squares_l3113_311381

theorem min_value_of_squares (a b t c : ℝ) (h : a + b = t) :
  ∃ m : ℝ, m = (t^2 + c^2 - 2*t*c + 2*c^2) / 2 ∧ 
  ∀ x y : ℝ, x + y = t → x^2 + (y + c)^2 ≥ m :=
by sorry

end min_value_of_squares_l3113_311381


namespace doubleBracket_two_l3113_311316

-- Define the double bracket notation
def doubleBracket (x : ℝ) : ℝ := x^2 + 2*x + 4

-- State the theorem
theorem doubleBracket_two : doubleBracket 2 = 12 := by
  sorry

end doubleBracket_two_l3113_311316


namespace trains_crossing_time_l3113_311379

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length_A length_B speed_A speed_B : ℝ) 
  (h1 : length_A = 200)
  (h2 : length_B = 250)
  (h3 : speed_A = 72)
  (h4 : speed_B = 18) : 
  (length_A + length_B) / ((speed_A + speed_B) * (1000 / 3600)) = 18 := by
  sorry

#check trains_crossing_time

end trains_crossing_time_l3113_311379


namespace max_rabbit_population_l3113_311305

/-- Represents the properties of a rabbit population --/
structure RabbitPopulation where
  total : ℕ
  longEars : ℕ
  jumpFar : ℕ
  bothTraits : ℕ

/-- Checks if a rabbit population satisfies the given conditions --/
def isValidPopulation (pop : RabbitPopulation) : Prop :=
  pop.longEars = 13 ∧
  pop.jumpFar = 17 ∧
  pop.bothTraits ≥ 3 ∧
  pop.longEars + pop.jumpFar - pop.bothTraits ≤ pop.total

/-- Theorem stating that 27 is the maximum number of rabbits satisfying the conditions --/
theorem max_rabbit_population :
  ∀ (pop : RabbitPopulation), isValidPopulation pop → pop.total ≤ 27 :=
sorry

end max_rabbit_population_l3113_311305


namespace multiply_82519_by_9999_l3113_311341

theorem multiply_82519_by_9999 : 82519 * 9999 = 825117481 := by
  sorry

end multiply_82519_by_9999_l3113_311341


namespace omega_range_l3113_311394

theorem omega_range (ω : ℝ) (h1 : ω > 1/4) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) - Real.cos (ω * x)
  (∀ k : ℤ, (3*π/4 + k*π) / ω ∉ Set.Ioo (2*π) (3*π)) →
  ω ∈ Set.union (Set.Icc (3/8) (7/12)) (Set.Icc (7/8) (11/12)) :=
by sorry

end omega_range_l3113_311394


namespace minimum_values_l3113_311382

theorem minimum_values :
  (∀ x > 1, (x + 4 / (x - 1) ≥ 5) ∧ (x + 4 / (x - 1) = 5 ↔ x = 3)) ∧
  (∀ x, 0 < x → x < 1 → (4 / x + 1 / (1 - x) ≥ 9) ∧ (4 / x + 1 / (1 - x) = 9 ↔ x = 2/3)) :=
by sorry

end minimum_values_l3113_311382


namespace hexagon_triangle_ratio_l3113_311313

/-- A regular hexagon divided into six equal triangles -/
structure RegularHexagon where
  /-- The area of one of the six triangles -/
  s : ℝ
  /-- The area of a region formed by two adjacent triangles -/
  r : ℝ
  /-- The hexagon is divided into six equal triangles -/
  triangle_count : ℕ
  triangle_count_eq : triangle_count = 6
  /-- r is the area of two adjacent triangles -/
  r_eq : r = 2 * s

/-- The ratio of the area of two adjacent triangles to the area of one triangle in a regular hexagon is 2 -/
theorem hexagon_triangle_ratio (h : RegularHexagon) : r / s = 2 := by
  sorry

end hexagon_triangle_ratio_l3113_311313


namespace person_savings_l3113_311304

theorem person_savings (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 5 / 4 →
  income = 15000 →
  savings = income - expenditure →
  savings = 3000 := by
sorry

end person_savings_l3113_311304


namespace smallest_power_is_four_l3113_311323

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1],
    ![1,  0]]

theorem smallest_power_is_four :
  (∀ k : ℕ, k > 0 ∧ k < 4 → rotation_matrix ^ k ≠ 1) ∧
  rotation_matrix ^ 4 = 1 :=
sorry

end smallest_power_is_four_l3113_311323


namespace f_composition_negative_three_l3113_311374

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 / Real.sqrt x else x^2

-- State the theorem
theorem f_composition_negative_three : f (f (-3)) = 1/3 := by
  sorry

end f_composition_negative_three_l3113_311374


namespace unique_n_exists_l3113_311357

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem unique_n_exists : ∃! n : ℕ, n > 0 ∧ n + S n + S (S n) = 2023 := by
  sorry

end unique_n_exists_l3113_311357


namespace unique_solution_factorial_equation_l3113_311310

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem unique_solution_factorial_equation : 
  ∃! n : ℕ, n * factorial n - factorial n = 5040 - factorial n :=
by sorry

end unique_solution_factorial_equation_l3113_311310


namespace remainder_of_3_to_20_mod_7_l3113_311388

theorem remainder_of_3_to_20_mod_7 : 3^20 % 7 = 2 := by sorry

end remainder_of_3_to_20_mod_7_l3113_311388


namespace quoted_poetry_mismatch_l3113_311334

-- Define a type for poetry quotes
inductive PoetryQuote
| A
| B
| C
| D

-- Define a function to check if a quote matches its context
def matchesContext (quote : PoetryQuote) : Prop :=
  match quote with
  | PoetryQuote.A => True
  | PoetryQuote.B => True
  | PoetryQuote.C => True
  | PoetryQuote.D => False

-- Theorem statement
theorem quoted_poetry_mismatch :
  ∃ (q : PoetryQuote), ¬(matchesContext q) ∧ ∀ (p : PoetryQuote), p ≠ q → matchesContext p :=
by
  sorry

end quoted_poetry_mismatch_l3113_311334


namespace range_of_x_l3113_311336

theorem range_of_x (x : ℝ) : (1 / x < 3) ∧ (1 / x > -4) → x > 1/3 ∨ x < -1/4 := by
  sorry

end range_of_x_l3113_311336


namespace sum_of_coefficients_l3113_311348

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem sum_of_coefficients :
  (∀ x, f (x + 5) = 4 * x^2 + 9 * x + 6) →
  (∃ a b c, ∀ x, f x = a * x^2 + b * x + c) →
  (∃ a b c, (∀ x, f x = a * x^2 + b * x + c) ∧ a + b + c = 34) :=
by sorry

end sum_of_coefficients_l3113_311348


namespace factorial_simplification_l3113_311386

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_simplification :
  (factorial 12) / ((factorial 10) + 3 * (factorial 9)) = 132 / 13 :=
by sorry

end factorial_simplification_l3113_311386


namespace acute_triangle_properties_l3113_311332

/-- Properties of an acute triangle ABC with specific conditions -/
theorem acute_triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π / 2 ∧
  0 < B ∧ B < π / 2 ∧
  0 < C ∧ C < π / 2 ∧
  -- Sum of angles in a triangle is π
  A + B + C = π ∧
  -- Side lengths are positive
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  -- Given condition for b
  b = 2 * Real.sqrt 6 ∧
  -- Sine rule
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  -- Angle bisector theorem
  (Real.sqrt 3 * a * c) / (a + c) = b * Real.sin (B / 2) / Real.sin B →
  -- Conclusions to prove
  π / 6 < C ∧ C < π / 2 ∧
  2 * Real.sqrt 2 < c ∧ c < 4 * Real.sqrt 2 ∧
  16 < a * c ∧ a * c ≤ 24 := by
  sorry

end acute_triangle_properties_l3113_311332


namespace sum_of_rectangle_areas_l3113_311365

/-- The sum of the areas of six rectangles with specified dimensions -/
theorem sum_of_rectangle_areas : 
  let width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36]
  let areas := lengths.map (λ l => width * l)
  areas.sum = 182 := by
  sorry

end sum_of_rectangle_areas_l3113_311365


namespace fraction_subtraction_l3113_311375

theorem fraction_subtraction (a : ℝ) (ha : a ≠ 0) : 1 / a - 3 / a = -2 / a := by
  sorry

end fraction_subtraction_l3113_311375


namespace union_equals_B_intersection_equals_B_l3113_311366

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the set C
def C : Set ℝ := {a : ℝ | a ≤ -1 ∨ a = 1}

-- Theorem 1
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a = 1 := by sorry

-- Theorem 2
theorem intersection_equals_B : A ∩ B a = B a → a ∈ C := by sorry

end union_equals_B_intersection_equals_B_l3113_311366


namespace gold_found_per_hour_l3113_311369

def diving_time : ℕ := 8
def chest_gold : ℕ := 100
def num_small_bags : ℕ := 2

def gold_per_hour : ℚ :=
  let small_bag_gold := chest_gold / 2
  let total_gold := chest_gold + num_small_bags * small_bag_gold
  total_gold / diving_time

theorem gold_found_per_hour :
  gold_per_hour = 25 := by sorry

end gold_found_per_hour_l3113_311369


namespace box_price_difference_l3113_311322

/-- The price difference between box C and box A -/
def price_difference (a b c : ℚ) : ℚ := c - a

/-- The conditions from the problem -/
def problem_conditions (a b c : ℚ) : Prop :=
  a + b + c = 9 ∧ 3*a + 2*b + c = 16

theorem box_price_difference :
  ∀ a b c : ℚ, problem_conditions a b c → price_difference a b c = 2 := by
  sorry

end box_price_difference_l3113_311322


namespace equation_solution_l3113_311362

theorem equation_solution : ∃ n : ℚ, (6 / n) - (6 - 3) / 6 = 1 ∧ n = 4 := by sorry

end equation_solution_l3113_311362


namespace problem_statement_l3113_311317

theorem problem_statement (a b : ℤ) : 
  ({1, a, b / a} : Set ℤ) = {0, a^2, a + b} → a^2017 + b^2017 = -1 :=
by sorry

end problem_statement_l3113_311317


namespace no_real_solution_for_equation_l3113_311373

theorem no_real_solution_for_equation :
  ¬ ∃ (x : ℝ), x ≠ 0 ∧ (2 / x - (3 / x) * (6 / x) = 0.5) := by
  sorry

end no_real_solution_for_equation_l3113_311373


namespace quadratic_rational_solutions_l3113_311340

/-- For a positive integer k, the equation kx^2 + 30x + k = 0 has rational solutions
    if and only if k = 9 or k = 15. -/
theorem quadratic_rational_solutions (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 30 * x + k = 0) ↔ k = 9 ∨ k = 15 := by
  sorry

end quadratic_rational_solutions_l3113_311340


namespace anya_original_position_l3113_311325

def Friend := Fin 5

structure Seating :=
  (positions : Friend → Fin 5)
  (bijective : Function.Bijective positions)

def sum_positions (s : Seating) : Nat :=
  (List.range 5).sum

-- Define the movements
def move_right (s : Seating) (f : Friend) (n : Nat) : Seating := sorry
def move_left (s : Seating) (f : Friend) (n : Nat) : Seating := sorry
def swap (s : Seating) (f1 f2 : Friend) : Seating := sorry
def move_to_end (s : Seating) (f : Friend) : Seating := sorry

theorem anya_original_position 
  (initial : Seating) 
  (anya varya galya diana ella : Friend) 
  (h_distinct : anya ≠ varya ∧ anya ≠ galya ∧ anya ≠ diana ∧ anya ≠ ella ∧ 
                varya ≠ galya ∧ varya ≠ diana ∧ varya ≠ ella ∧ 
                galya ≠ diana ∧ galya ≠ ella ∧ 
                diana ≠ ella) 
  (final : Seating) 
  (h_movements : final = move_to_end (swap (move_left (move_right initial varya 3) galya 1) diana ella) anya) 
  (h_sum_equal : sum_positions initial = sum_positions final) :
  initial.positions anya = 3 := by sorry

end anya_original_position_l3113_311325


namespace surfers_ratio_l3113_311377

def surfers_problem (first_day : ℕ) (second_day_increase : ℕ) (average : ℕ) : Prop :=
  let second_day := first_day + second_day_increase
  let total := average * 3
  let third_day := total - first_day - second_day
  (third_day : ℚ) / first_day = 2 / 5

theorem surfers_ratio : 
  surfers_problem 1500 600 1400 := by sorry

end surfers_ratio_l3113_311377


namespace unique_k_solution_l3113_311330

theorem unique_k_solution : 
  ∃! (k : ℕ), k ≥ 1 ∧ (∃ (n m : ℤ), 9 * n^6 = 2^k + 5 * m^2 + 2) ∧ k = 1 := by
  sorry

end unique_k_solution_l3113_311330


namespace excluded_age_is_nine_l3113_311344

/-- A 5-digit number with distinct, consecutive digits in increasing order -/
def ConsecutiveDigitNumber := { n : ℕ | 
  12345 ≤ n ∧ n ≤ 98765 ∧ 
  ∃ (a b c d e : ℕ), n = 10000*a + 1000*b + 100*c + 10*d + e ∧
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e }

/-- The set of ages of Mrs. Smith's children -/
def ChildrenAges := { n : ℕ | 5 ≤ n ∧ n ≤ 13 }

theorem excluded_age_is_nine :
  ∃ (n : ConsecutiveDigitNumber),
    ∀ (k : ℕ), k ∈ ChildrenAges → k ≠ 9 → n % k = 0 :=
by sorry

end excluded_age_is_nine_l3113_311344


namespace actual_speed_calculation_l3113_311356

/-- 
Given a person who travels a certain distance at an unknown speed, 
this theorem proves that if walking at 10 km/hr would allow them 
to travel 20 km more in the same time, and the actual distance 
traveled is 20 km, then their actual speed is 5 km/hr.
-/
theorem actual_speed_calculation 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 20) 
  (h2 : faster_speed = 10) 
  (h3 : additional_distance = 20) 
  (h4 : actual_distance / actual_speed = (actual_distance + additional_distance) / faster_speed) :
  actual_speed = 5 :=
sorry

#check actual_speed_calculation

end actual_speed_calculation_l3113_311356


namespace bus_journey_l3113_311335

theorem bus_journey (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5.5)
  (h5 : ∀ x : ℝ, x / speed1 + (total_distance - x) / speed2 = total_time → x = 160) :
  ∃ x : ℝ, x / speed1 + (total_distance - x) / speed2 = total_time ∧ x = 160 := by
sorry

end bus_journey_l3113_311335


namespace only_sperm_has_one_set_l3113_311345

-- Define the types of cells
inductive CellType
  | Zygote
  | SomaticCell
  | Spermatogonium
  | Sperm

-- Define a function to represent the number of chromosome sets in a cell
def chromosomeSets : CellType → ℕ
  | CellType.Zygote => 2
  | CellType.SomaticCell => 2
  | CellType.Spermatogonium => 2
  | CellType.Sperm => 1

-- Define that spermatogonium is a type of somatic cell
axiom spermatogonium_is_somatic : chromosomeSets CellType.Spermatogonium = chromosomeSets CellType.SomaticCell

-- Define that sperm is formed through meiosis (implicitly resulting in one set of chromosomes)
axiom sperm_meiosis : chromosomeSets CellType.Sperm = 1

-- Theorem: Only sperm contains one set of chromosomes
theorem only_sperm_has_one_set :
  ∀ (cell : CellType), chromosomeSets cell = 1 ↔ cell = CellType.Sperm :=
by sorry


end only_sperm_has_one_set_l3113_311345


namespace parabola_vertex_specific_parabola_vertex_l3113_311378

/-- The vertex of a parabola in the form y = a(x-h)^2 + k is (h,k) -/
theorem parabola_vertex (a : ℝ) (h k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  (h, k) = (h, f h) ∧ ∀ x, f x ≥ f h := by sorry

/-- The vertex of the parabola y = 1/3 * (x-7)^2 + 5 is (7,5) -/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ (1/3) * (x - 7)^2 + 5
  (7, 5) = (7, f 7) ∧ ∀ x, f x ≥ f 7 := by sorry

end parabola_vertex_specific_parabola_vertex_l3113_311378


namespace common_element_in_sets_l3113_311326

theorem common_element_in_sets (n : ℕ) (S : Finset (Finset ℕ)) : 
  n = 50 →
  S.card = n →
  (∀ s ∈ S, s.card = 30) →
  (∀ T ⊆ S, T.card = 30 → ∃ x, ∀ s ∈ T, x ∈ s) →
  ∃ x, ∀ s ∈ S, x ∈ s :=
by sorry

end common_element_in_sets_l3113_311326


namespace no_real_solution_l3113_311391

theorem no_real_solution (a b c : ℝ) : ¬∃ (x y z : ℝ), 
  (a^2 + b^2 + c^2 + 3*(x^2 + y^2 + z^2) = 6) ∧ (a*x + b*y + c*z = 2) := by
  sorry

end no_real_solution_l3113_311391


namespace identify_counterfeit_coin_l3113_311314

/-- Represents the result of a weighing -/
inductive WeighResult
  | Left  : WeighResult  -- Left pan is heavier
  | Right : WeighResult  -- Right pan is heavier
  | Equal : WeighResult  -- Pans are balanced

/-- Represents a coin -/
inductive Coin
  | A : Coin
  | B : Coin
  | C : Coin
  | D : Coin

/-- Represents the state of a coin -/
inductive CoinState
  | Genuine : CoinState
  | Counterfeit : CoinState

/-- Represents whether the counterfeit coin is heavier or lighter -/
inductive CounterfeitWeight
  | Heavier : CounterfeitWeight
  | Lighter : CounterfeitWeight

/-- Function to perform a weighing -/
def weigh (left : List Coin) (right : List Coin) : WeighResult := sorry

/-- Function to determine the state of a coin -/
def determineCoinState (c : Coin) : CoinState := sorry

/-- Function to determine if the counterfeit coin is heavier or lighter -/
def determineCounterfeitWeight : CounterfeitWeight := sorry

/-- Theorem stating that the counterfeit coin can be identified in at most 3 weighings -/
theorem identify_counterfeit_coin :
  ∃ (counterfeit : Coin) (weight : CounterfeitWeight),
    (∀ c : Coin, c ≠ counterfeit → determineCoinState c = CoinState.Genuine) ∧
    (determineCoinState counterfeit = CoinState.Counterfeit) ∧
    (weight = determineCounterfeitWeight) ∧
    (∃ (w₁ w₂ w₃ : WeighResult),
      w₁ = weigh [Coin.A, Coin.B] [Coin.C, Coin.D] ∧
      w₂ = weigh [Coin.A, Coin.C] [Coin.B, Coin.D] ∧
      w₃ = weigh [Coin.A, Coin.D] [Coin.B, Coin.C]) :=
by
  sorry

end identify_counterfeit_coin_l3113_311314


namespace balloon_permutations_l3113_311355

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ := 1260

/-- The total number of letters in "balloon" -/
def total_letters : ℕ := 7

/-- The frequency of each letter in "balloon" -/
def letter_frequency : List ℕ := [1, 1, 2, 2, 1]

theorem balloon_permutations :
  balloon_arrangements = Nat.factorial total_letters / (List.prod letter_frequency) := by
  sorry

end balloon_permutations_l3113_311355


namespace scissors_in_drawer_final_scissors_count_l3113_311354

theorem scissors_in_drawer (initial : ℕ) (added : ℕ) (removed : ℕ) : ℕ :=
  initial + added - removed

theorem final_scissors_count : scissors_in_drawer 54 22 15 = 61 := by
  sorry

end scissors_in_drawer_final_scissors_count_l3113_311354


namespace speed_of_k_l3113_311367

-- Define the speeds and time delay
def speed_a : ℝ := 30
def speed_b : ℝ := 40
def delay : ℝ := 5

-- Define the theorem
theorem speed_of_k (speed_k : ℝ) : 
  -- a, b, k start from the same place and travel in the same direction
  -- a travels at speed_a km/hr
  -- b travels at speed_b km/hr
  -- b starts delay hours after a
  -- b and k overtake a at the same instant
  -- k starts at the same time as a
  (∃ (t : ℝ), t > 0 ∧ 
    speed_b * t = speed_a * (t + delay) ∧
    speed_k * (t + delay) = speed_a * (t + delay)) →
  -- Then the speed of k is 35 km/hr
  speed_k = 35 := by
sorry

end speed_of_k_l3113_311367


namespace spadesuit_example_l3113_311320

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a^2 - b^2|

-- Theorem statement
theorem spadesuit_example : spadesuit 3 (spadesuit 5 2) = 432 := by
  sorry

end spadesuit_example_l3113_311320


namespace intersection_complement_equals_set_l3113_311385

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 6}
def B : Set ℕ := {2, 4, 5, 6}

theorem intersection_complement_equals_set (h : Set ℕ) : A ∩ (U \ B) = {1, 3} := by
  sorry

end intersection_complement_equals_set_l3113_311385


namespace grocery_store_buyers_difference_l3113_311324

/-- Given information about buyers in a grocery store over three days, 
    prove the difference in buyers between today and yesterday --/
theorem grocery_store_buyers_difference 
  (buyers_day_before_yesterday : ℕ) 
  (buyers_yesterday : ℕ) 
  (buyers_today : ℕ) 
  (total_buyers : ℕ) 
  (h1 : buyers_day_before_yesterday = 50)
  (h2 : buyers_yesterday = buyers_day_before_yesterday / 2)
  (h3 : total_buyers = buyers_day_before_yesterday + buyers_yesterday + buyers_today)
  (h4 : total_buyers = 140) :
  buyers_today - buyers_yesterday = 40 := by
sorry


end grocery_store_buyers_difference_l3113_311324


namespace best_fit_model_l3113_311321

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  r_squared : ℝ

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

theorem best_fit_model (models : List RegressionModel) 
  (h1 : RegressionModel.mk 0.95 ∈ models)
  (h2 : RegressionModel.mk 0.70 ∈ models)
  (h3 : RegressionModel.mk 0.55 ∈ models)
  (h4 : RegressionModel.mk 0.30 ∈ models)
  (h5 : models.length = 4) :
  has_best_fit (RegressionModel.mk 0.95) models :=
sorry

end best_fit_model_l3113_311321


namespace sum_of_x_coordinates_l3113_311368

/-- Given three points X, Y, and Z in a plane satisfying certain conditions, 
    prove that the sum of X's coordinates is 34. -/
theorem sum_of_x_coordinates (X Y Z : ℝ × ℝ) : 
  (dist X Z) / (dist X Y) = 2/3 →
  (dist Z Y) / (dist X Y) = 1/3 →
  Y = (1, 9) →
  Z = (-1, 3) →
  X.1 + X.2 = 34 := by sorry


end sum_of_x_coordinates_l3113_311368


namespace cube_volume_from_surface_area_l3113_311359

theorem cube_volume_from_surface_area :
  ∀ s : ℝ, 
    s > 0 →
    6 * s^2 = 150 →
    s^3 = 125 :=
by
  sorry

end cube_volume_from_surface_area_l3113_311359


namespace train_length_l3113_311371

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 →
  time_s = 3.9996800255979523 →
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 ∧ 
    length_m = speed_kmh * (1000 / 3600) * time_s :=
by sorry

end train_length_l3113_311371


namespace talitha_took_108_pieces_l3113_311302

/-- Given an initial candy count, the number of pieces Solomon took, and the final candy count,
    calculate the number of pieces Talitha took. -/
def talitha_candy_count (initial : ℕ) (solomon_took : ℕ) (final : ℕ) : ℕ :=
  initial - solomon_took - final

/-- Theorem stating that Talitha took 108 pieces of candy. -/
theorem talitha_took_108_pieces :
  talitha_candy_count 349 153 88 = 108 := by
  sorry

end talitha_took_108_pieces_l3113_311302


namespace same_color_sock_pairs_l3113_311318

/-- The number of ways to choose a pair of socks of the same color from a drawer -/
theorem same_color_sock_pairs (white brown blue red : ℕ) 
  (h_white : white = 5)
  (h_brown : brown = 6)
  (h_blue : blue = 3)
  (h_red : red = 2) : 
  Nat.choose white 2 + Nat.choose brown 2 + Nat.choose blue 2 + Nat.choose red 2 = 29 := by
  sorry

end same_color_sock_pairs_l3113_311318
