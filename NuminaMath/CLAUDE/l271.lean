import Mathlib

namespace f_1993_of_3_eq_one_fifth_l271_27198

-- Define the function f
def f (x : ℚ) : ℚ := (1 + x) / (1 - 3 * x)

-- Define the iterated function f_n recursively
def f_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- State the theorem
theorem f_1993_of_3_eq_one_fifth : f_n 1993 3 = 1/5 := by sorry

end f_1993_of_3_eq_one_fifth_l271_27198


namespace semicircle_covering_l271_27163

theorem semicircle_covering (N : ℕ) (r : ℝ) : 
  N > 0 → 
  r > 0 → 
  let A := N * (π * r^2 / 2)
  let B := (π * (N * r)^2 / 2) - A
  A / B = 1 / 3 → 
  N = 4 := by
sorry

end semicircle_covering_l271_27163


namespace special_ellipse_major_twice_minor_l271_27103

/-- An ellipse where one focus and two vertices form an equilateral triangle -/
structure SpecialEllipse where
  -- Major axis length
  a : ℝ
  -- Minor axis length
  b : ℝ
  -- Distance from center to focus
  c : ℝ
  -- Constraint that one focus and two vertices form an equilateral triangle
  equilateral_triangle : c = a / 2
  -- Standard ellipse equation
  ellipse_equation : a^2 = b^2 + c^2

/-- The major axis is twice the minor axis in a special ellipse -/
theorem special_ellipse_major_twice_minor (e : SpecialEllipse) : e.a = 2 * e.b := by
  sorry

end special_ellipse_major_twice_minor_l271_27103


namespace x_eq_2_sufficient_not_necessary_l271_27134

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- The condition "x = 2" for the given vectors -/
def condition (x : ℝ) : Prop := x = 2

/-- The vectors a and b as functions of x -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

theorem x_eq_2_sufficient_not_necessary :
  (∀ x : ℝ, condition x → are_parallel (a x) (b x)) ∧
  ¬(∀ x : ℝ, are_parallel (a x) (b x) → condition x) := by
  sorry

end x_eq_2_sufficient_not_necessary_l271_27134


namespace range_of_f_real_l271_27109

def f (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem range_of_f_real : Set.range f = Set.Ici 1 := by sorry

end range_of_f_real_l271_27109


namespace tree_planting_theorem_l271_27168

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The number of trees planted by 6th graders -/
def trees_6th : ℕ := 3 * trees_5th - 30

/-- The total number of trees planted by all three grades -/
def total_trees : ℕ := trees_4th + trees_5th + trees_6th

theorem tree_planting_theorem : total_trees = 240 := by
  sorry

end tree_planting_theorem_l271_27168


namespace volume_of_specific_cuboid_l271_27160

/-- The volume of a cuboid formed by two identical cubes in a line --/
def cuboid_volume (edge_length : ℝ) : ℝ :=
  2 * (edge_length ^ 3)

/-- Theorem: The volume of a cuboid formed by two cubes with edge length 5 cm is 250 cm³ --/
theorem volume_of_specific_cuboid :
  cuboid_volume 5 = 250 := by
  sorry

end volume_of_specific_cuboid_l271_27160


namespace english_only_students_l271_27169

theorem english_only_students (total : Nat) (max_liz : Nat) (english : Nat) (french : Nat) : 
  total = 25 → 
  max_liz = 2 → 
  total = english + french - max_liz → 
  english = 2 * french → 
  english - french = 16 :=
by
  sorry

end english_only_students_l271_27169


namespace reflection_of_point_p_l271_27183

/-- The coordinates of a point with respect to the center of the coordinate origin -/
def reflection_through_origin (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- Theorem: The coordinates of P(-3,1) with respect to the center of the coordinate origin are (3,-1) -/
theorem reflection_of_point_p : reflection_through_origin (-3) 1 = (3, -1) := by
  sorry

end reflection_of_point_p_l271_27183


namespace sum_of_integers_l271_27189

theorem sum_of_integers (a b : ℕ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a^2 - b^2 = 2018 - 2*a) : a + b = 672 := by
  sorry

end sum_of_integers_l271_27189


namespace marble_jar_problem_l271_27101

/-- The number of marbles in the jar -/
def M : ℕ := 364

/-- The initial number of people -/
def initial_people : ℕ := 26

/-- The number of people who join later -/
def joining_people : ℕ := 2

/-- The number of marbles each person gets in the initial distribution -/
def initial_distribution : ℕ := M / initial_people

/-- The number of marbles each person would get after more people join -/
def later_distribution : ℕ := M / (initial_people + joining_people)

theorem marble_jar_problem :
  (M = initial_people * initial_distribution) ∧
  (M = (initial_people + joining_people) * (initial_distribution - 1)) :=
sorry

end marble_jar_problem_l271_27101


namespace point_outside_circle_l271_27145

theorem point_outside_circle (r OA : ℝ) (h1 : r = 3) (h2 : OA = 5) :
  OA > r := by sorry

end point_outside_circle_l271_27145


namespace prob_three_ones_in_four_rolls_eq_5_324_l271_27180

/-- A fair, regular six-sided die -/
def fair_die : Finset ℕ := Finset.range 6

/-- The probability of an event occurring when rolling a fair die -/
def prob (event : Finset ℕ) : ℚ :=
  event.card / fair_die.card

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of rolling a 1 exactly three times in four rolls of a fair die -/
def prob_three_ones_in_four_rolls : ℚ :=
  (choose 4 3 : ℚ) * (prob {0})^3 * (1 - prob {0})

theorem prob_three_ones_in_four_rolls_eq_5_324 :
  prob_three_ones_in_four_rolls = 5 / 324 := by
  sorry

end prob_three_ones_in_four_rolls_eq_5_324_l271_27180


namespace range_of_a_l271_27188

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (h_xy : x + y + 8 = x * y)
  (h_ineq : ∀ x y : ℝ, x > 0 → y > 0 → x + y + 8 = x * y → (x + y)^2 - a*(x + y) + 1 ≥ 0) :
  a ≤ 65/8 :=
sorry

end range_of_a_l271_27188


namespace min_snakes_is_three_l271_27171

/-- Represents the number of people owning a specific combination of pets -/
structure PetOwnership :=
  (total : ℕ)
  (onlyDogs : ℕ)
  (onlyCats : ℕ)
  (catsAndDogs : ℕ)
  (catsDogsSnakes : ℕ)

/-- The minimum number of snakes given the pet ownership information -/
def minSnakes (po : PetOwnership) : ℕ := po.catsDogsSnakes

/-- Theorem stating that the minimum number of snakes is 3 given the problem conditions -/
theorem min_snakes_is_three (po : PetOwnership)
  (h1 : po.total = 89)
  (h2 : po.onlyDogs = 15)
  (h3 : po.onlyCats = 10)
  (h4 : po.catsAndDogs = 5)
  (h5 : po.catsDogsSnakes = 3) :
  minSnakes po = 3 := by sorry

end min_snakes_is_three_l271_27171


namespace complex_equation_sum_l271_27126

theorem complex_equation_sum (x y : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (x - 2 : ℂ) + y * i = -1 + i) : x + y = 2 := by
  sorry

end complex_equation_sum_l271_27126


namespace prime_in_sequence_l271_27155

theorem prime_in_sequence (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∃ n : ℕ, p = Int.sqrt (24 * n + 1) := by
  sorry

end prime_in_sequence_l271_27155


namespace division_remainder_proof_l271_27139

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 141 →
  divisor = 17 →
  quotient = 8 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end division_remainder_proof_l271_27139


namespace sand_art_problem_l271_27102

/-- The amount of sand needed to fill one square inch -/
def sand_per_square_inch (rectangle_length rectangle_width square_side total_sand : ℕ) : ℚ :=
  total_sand / (rectangle_length * rectangle_width + square_side * square_side)

theorem sand_art_problem (rectangle_length rectangle_width square_side total_sand : ℕ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 7)
  (h3 : square_side = 5)
  (h4 : total_sand = 201) :
  sand_per_square_inch rectangle_length rectangle_width square_side total_sand = 3 := by
  sorry

end sand_art_problem_l271_27102


namespace card_probability_and_combinations_l271_27130

theorem card_probability_and_combinations : 
  -- Part 1: Probability of drawing two hearts
  (Nat.choose 13 2 : ℚ) / (Nat.choose 52 2) = 1 / 17 ∧ 
  -- Part 2: Number of ways to choose 15 from 17
  Nat.choose 17 15 = 136 ∧ 
  -- Part 3: Number of non-empty subsets of a 4-element set
  (2^4 - 1 : ℕ) = 15 ∧ 
  -- Part 4: Probability of drawing a red ball
  (3 : ℚ) / 15 = 1 / 5 := by
  sorry


end card_probability_and_combinations_l271_27130


namespace elevator_optimal_stop_l271_27151

def total_floors : ℕ := 12
def num_people : ℕ := 11

def dissatisfaction (n : ℕ) : ℕ :=
  let down_sum := (n - 2) * (n - 1) / 2
  let up_sum := (total_floors - n) * (total_floors - n + 1)
  down_sum + 2 * up_sum

theorem elevator_optimal_stop :
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ total_floors →
    dissatisfaction 9 ≤ dissatisfaction k :=
sorry

end elevator_optimal_stop_l271_27151


namespace units_digit_base_6_product_l271_27179

theorem units_digit_base_6_product (a b : ℕ) (ha : a = 312) (hb : b = 67) :
  (a * b) % 6 = 0 := by
sorry

end units_digit_base_6_product_l271_27179


namespace ryan_got_seven_books_l271_27143

/-- Calculates the number of books Ryan got from the library given the conditions -/
def ryans_books (ryan_total_pages : ℕ) (brother_daily_pages : ℕ) (days : ℕ) (ryan_extra_daily_pages : ℕ) : ℕ :=
  ryan_total_pages / (brother_daily_pages + ryan_extra_daily_pages)

/-- Theorem stating that Ryan got 7 books from the library -/
theorem ryan_got_seven_books :
  ryans_books 2100 200 7 100 = 7 := by
  sorry

end ryan_got_seven_books_l271_27143


namespace roger_cookie_price_l271_27105

/-- Represents a cookie batch -/
structure CookieBatch where
  shape : String
  count : ℕ
  price : ℚ

/-- Calculates the total earnings from a cookie batch -/
def totalEarnings (batch : CookieBatch) : ℚ :=
  batch.count * batch.price

theorem roger_cookie_price (art_batch roger_batch : CookieBatch) 
  (h1 : art_batch.shape = "rectangle")
  (h2 : roger_batch.shape = "square")
  (h3 : art_batch.count = 15)
  (h4 : roger_batch.count = 20)
  (h5 : art_batch.price = 75/100)
  (h6 : totalEarnings art_batch = totalEarnings roger_batch) :
  roger_batch.price = 5625/10000 := by
  sorry

#eval (5625 : ℚ) / 10000  -- Expected output: 0.5625

end roger_cookie_price_l271_27105


namespace complex_problem_l271_27115

theorem complex_problem (b : ℝ) (z : ℂ) (h1 : z = 3 + b * I) 
  (h2 : (Complex.I * (Complex.I * ((1 + 3 * I) * z))).re = 0) : 
  z = 3 + I ∧ Complex.abs (z / (2 + I)) = Real.sqrt 2 := by
  sorry

end complex_problem_l271_27115


namespace platform_length_calculation_l271_27131

/-- The speed of the goods train in kilometers per hour -/
def train_speed : ℝ := 72

/-- The time taken by the train to cross the platform in seconds -/
def crossing_time : ℝ := 26

/-- The length of the goods train in meters -/
def train_length : ℝ := 170.0416

/-- The length of the platform in meters -/
def platform_length : ℝ := 349.9584

/-- Theorem stating that the calculated platform length is correct -/
theorem platform_length_calculation :
  platform_length = (train_speed * 1000 / 3600 * crossing_time) - train_length := by
  sorry

end platform_length_calculation_l271_27131


namespace polynomial_sum_coefficients_l271_27158

theorem polynomial_sum_coefficients (d : ℝ) (a b c e : ℤ) : 
  d ≠ 0 → 
  (16 * d + 15 + 18 * d^2 + 5 * d^3) + (4 * d + 3 + 3 * d^3) = a * d + b + c * d^2 + e * d^3 →
  a + b + c + e = 64 := by
sorry

end polynomial_sum_coefficients_l271_27158


namespace students_with_dogs_and_amphibians_but_not_cats_l271_27196

theorem students_with_dogs_and_amphibians_but_not_cats (total_students : ℕ) 
  (students_with_dogs : ℕ) (students_with_cats : ℕ) (students_with_amphibians : ℕ) 
  (students_without_pets : ℕ) :
  total_students = 40 →
  students_with_dogs = 24 →
  students_with_cats = 10 →
  students_with_amphibians = 8 →
  students_without_pets = 6 →
  (∃ (x y z : ℕ),
    x + y = students_with_dogs ∧
    y + z = students_with_amphibians ∧
    x + y + z = total_students - students_without_pets ∧
    y = 0) :=
by sorry

end students_with_dogs_and_amphibians_but_not_cats_l271_27196


namespace sqrt_pattern_l271_27175

theorem sqrt_pattern (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 + (2 * n - 1) / (n^2 : ℝ)) = (n + 1 : ℝ) / n :=
by sorry

end sqrt_pattern_l271_27175


namespace newspaper_delivery_ratio_l271_27191

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- The number of newspapers Jake delivers in a week -/
def jake_weekly : ℕ := 234

/-- The additional number of newspapers Miranda delivers compared to Jake in a month -/
def miranda_monthly_extra : ℕ := 936

/-- The ratio of newspapers Miranda delivers to Jake's deliveries in a week -/
def delivery_ratio : ℚ := (jake_weekly * weeks_per_month + miranda_monthly_extra) / (jake_weekly * weeks_per_month)

theorem newspaper_delivery_ratio :
  delivery_ratio = 2 := by sorry

end newspaper_delivery_ratio_l271_27191


namespace smallest_positive_e_value_l271_27146

theorem smallest_positive_e_value (a b c d e : ℤ) : 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    (x = -3 ∨ x = 7 ∨ x = 8 ∨ x = -1/4)) →
  (∀ e' : ℤ, e' > 0 → e' ≥ 168) →
  e = 168 :=
sorry

end smallest_positive_e_value_l271_27146


namespace quadratic_inequality_l271_27172

theorem quadratic_inequality (x : ℝ) : x^2 - 40*x + 400 ≤ 10 ↔ 20 - Real.sqrt 10 ≤ x ∧ x ≤ 20 + Real.sqrt 10 := by
  sorry

end quadratic_inequality_l271_27172


namespace q_is_false_l271_27107

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p)) : ¬q :=
sorry

end q_is_false_l271_27107


namespace triangle_reconstruction_uniqueness_l271_27104

-- Define the necessary types
structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the given information
def B : Point := sorry
def G : Point := sorry  -- centroid
def L : Point := sorry  -- intersection of symmedian from B with circumcircle

-- Define the necessary concepts
def isCentroid (G : Point) (t : Triangle) : Prop := sorry
def isSymmedianIntersection (L : Point) (t : Triangle) : Prop := sorry

-- The theorem statement
theorem triangle_reconstruction_uniqueness :
  ∃! (t : Triangle), 
    t.B = B ∧ 
    isCentroid G t ∧ 
    isSymmedianIntersection L t :=
sorry

end triangle_reconstruction_uniqueness_l271_27104


namespace second_to_first_ratio_l271_27125

/-- Represents the guesses of four students for the number of jellybeans in a jar. -/
structure JellybeanGuesses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Defines the conditions for the jellybean guessing problem. -/
def valid_guesses (g : JellybeanGuesses) : Prop :=
  g.first = 100 ∧
  g.third = g.second - 200 ∧
  g.fourth = (g.first + g.second + g.third) / 3 + 25 ∧
  g.fourth = 525

/-- Theorem stating that for valid guesses, the ratio of the second to the first guess is 8:1. -/
theorem second_to_first_ratio (g : JellybeanGuesses) (h : valid_guesses g) :
  g.second / g.first = 8 := by
  sorry

#check second_to_first_ratio

end second_to_first_ratio_l271_27125


namespace expand_expression_l271_27142

theorem expand_expression (x : ℝ) : 5 * (x + 3) * (x + 6) = 5 * x^2 + 45 * x + 90 := by
  sorry

end expand_expression_l271_27142


namespace subtraction_division_equality_l271_27165

theorem subtraction_division_equality : 5020 - (502 / 100.4) = 5014.998 := by
  sorry

end subtraction_division_equality_l271_27165


namespace problem_statements_l271_27124

theorem problem_statements :
  (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) ∧
  (∃ x y : ℝ, (abs x > abs y ∧ x ≤ y) ∧ ∃ x y : ℝ, (x > y ∧ abs x ≤ abs y)) ∧
  (∃ x : ℤ, x^2 ≤ 0) ∧
  (∀ m : ℝ, (∃ x y : ℝ, x^2 - 2*x + m = 0 ∧ x > 0 ∧ y < 0) ↔ m < 0) :=
by sorry

end problem_statements_l271_27124


namespace worker_wages_l271_27184

theorem worker_wages (workers1 workers2 days1 days2 total_wages2 : ℕ)
  (hw1 : workers1 = 15)
  (hw2 : workers2 = 19)
  (hd1 : days1 = 6)
  (hd2 : days2 = 5)
  (ht2 : total_wages2 = 9975) :
  workers1 * days1 * (total_wages2 / (workers2 * days2)) = 9450 := by
  sorry

end worker_wages_l271_27184


namespace circular_garden_ratio_l271_27110

theorem circular_garden_ratio (r : ℝ) (h : r = 10) : 
  (2 * π * r) / (π * r^2) = 1 / 5 := by
  sorry

end circular_garden_ratio_l271_27110


namespace range_of_expression_l271_27156

theorem range_of_expression (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (∀ x y : ℝ, y = x^2 + 2*b*x + 1 → y ≠ 2*a*(x + b)) →
  ∀ θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) →
    1 < (a - Real.cos θ)^2 + (b - Real.sin θ)^2 ∧
    (a - Real.cos θ)^2 + (b - Real.sin θ)^2 < 4 := by
  sorry


end range_of_expression_l271_27156


namespace number_of_towns_l271_27153

theorem number_of_towns (n : ℕ) : Nat.choose n 2 = 15 → n = 6 := by
  sorry

end number_of_towns_l271_27153


namespace tangent_line_to_circle_l271_27141

theorem tangent_line_to_circle (θ : Real) (h1 : 0 < θ ∧ θ < π) :
  (∃ t : Real, ∀ x y : Real,
    (x = t * Real.cos θ ∧ y = t * Real.sin θ) →
    (∃ α : Real, x = 4 + 2 * Real.cos α ∧ y = 2 * Real.sin α) →
    (∀ x' y' : Real, (x' - 4)^2 + y'^2 = 4 →
      (y' - y) * Real.cos θ = (x' - x) * Real.sin θ)) →
  θ = π/6 ∨ θ = 5*π/6 := by
sorry

end tangent_line_to_circle_l271_27141


namespace lcm_gcf_problem_l271_27137

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 10 = 36 → Nat.gcd n 10 = 5 → n = 18 := by
  sorry

end lcm_gcf_problem_l271_27137


namespace godzilla_stitches_proof_l271_27121

/-- The number of stitches Carolyn can sew per minute -/
def stitches_per_minute : ℕ := 4

/-- The number of stitches required to embroider a flower -/
def stitches_per_flower : ℕ := 60

/-- The number of stitches required to embroider a unicorn -/
def stitches_per_unicorn : ℕ := 180

/-- The number of unicorns in the embroidery -/
def num_unicorns : ℕ := 3

/-- The number of flowers in the embroidery -/
def num_flowers : ℕ := 50

/-- The total time Carolyn spends embroidering (in minutes) -/
def total_time : ℕ := 1085

/-- The number of stitches required to embroider Godzilla -/
def stitches_for_godzilla : ℕ := 800

theorem godzilla_stitches_proof : 
  stitches_for_godzilla = 
    total_time * stitches_per_minute - 
    (num_unicorns * stitches_per_unicorn + num_flowers * stitches_per_flower) := by
  sorry

end godzilla_stitches_proof_l271_27121


namespace students_using_green_l271_27140

theorem students_using_green (total : ℕ) (both : ℕ) (red : ℕ) : 
  total = 70 → both = 38 → red = 56 → 
  total = (total - both) + red → 
  (total - both) = 52 := by sorry

end students_using_green_l271_27140


namespace yw_equals_two_l271_27129

/-- A right triangle with specific side lengths and a median -/
structure RightTriangleWithMedian where
  /-- The length of side XY -/
  xy : ℝ
  /-- The length of side YZ -/
  yz : ℝ
  /-- The point where the median from X meets YZ -/
  w : ℝ
  /-- XY equals 3 -/
  xy_eq : xy = 3
  /-- YZ equals 4 -/
  yz_eq : yz = 4

/-- The length YW in a right triangle with specific side lengths and median -/
def yw (t : RightTriangleWithMedian) : ℝ := t.w

/-- Theorem: In a right triangle XYZ with XY = 3 and YZ = 4, 
    if W is where the median from X meets YZ, then YW = 2 -/
theorem yw_equals_two (t : RightTriangleWithMedian) : yw t = 2 := by
  sorry

end yw_equals_two_l271_27129


namespace prob_two_hearts_one_spade_l271_27152

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Fin 52)
  (ranks : Fin 13)
  (suits : Fin 4)

/-- Represents the suits in a deck -/
inductive Suit
| hearts
| diamonds
| clubs
| spades

/-- Defines the color of a suit -/
def suitColor (s : Suit) : Bool :=
  match s with
  | Suit.hearts | Suit.diamonds => true  -- Red
  | Suit.clubs | Suit.spades => false    -- Black

/-- Calculates the probability of drawing two hearts followed by a spade -/
def probabilityTwoHeartsOneSpade (d : Deck) : ℚ :=
  13 / 850

/-- Theorem stating the probability of drawing two hearts followed by a spade -/
theorem prob_two_hearts_one_spade (d : Deck) :
  probabilityTwoHeartsOneSpade d = 13 / 850 := by
  sorry

end prob_two_hearts_one_spade_l271_27152


namespace subset_implies_a_values_l271_27132

def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}

theorem subset_implies_a_values :
  ∀ a : ℝ, B a ⊆ A → a ∈ ({0, 1, -1} : Set ℝ) := by
  sorry

end subset_implies_a_values_l271_27132


namespace cards_left_after_distribution_l271_27167

/-- Given the initial number of cards, number of cards given to each student,
    and number of students, prove that the number of cards left is 12. -/
theorem cards_left_after_distribution (initial_cards : ℕ) (cards_per_student : ℕ) (num_students : ℕ)
    (h1 : initial_cards = 357)
    (h2 : cards_per_student = 23)
    (h3 : num_students = 15) :
  initial_cards - (cards_per_student * num_students) = 12 := by
  sorry

#check cards_left_after_distribution

end cards_left_after_distribution_l271_27167


namespace first_digit_base_9_l271_27148

def base_3_digits : List Nat := [2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1]

def y : Nat := (List.reverse base_3_digits).enum.foldl (fun acc (i, digit) => acc + digit * (3 ^ i)) 0

theorem first_digit_base_9 : ∃ (k : Nat), 4 * (9 ^ k) ≤ y ∧ y < 5 * (9 ^ k) ∧ (∀ m, m > k → y < 4 * (9 ^ m)) :=
  sorry

end first_digit_base_9_l271_27148


namespace real_part_reciprocal_l271_27185

theorem real_part_reciprocal (z : ℂ) (h : z = 1 - 2*I) : 
  (1 / z).re = 1 / 5 := by sorry

end real_part_reciprocal_l271_27185


namespace remainder_of_2456789_div_7_l271_27181

theorem remainder_of_2456789_div_7 : 2456789 % 7 = 6 := by
  sorry

end remainder_of_2456789_div_7_l271_27181


namespace system_solution_l271_27108

variable (y : ℝ)
variable (x₁ x₂ x₃ x₄ x₅ : ℝ)

def system_equations (y x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₅ + x₂ = y * x₁) ∧
  (x₁ + x₃ = y * x₂) ∧
  (x₂ + x₄ = y * x₃) ∧
  (x₃ + x₅ = y * x₄) ∧
  (x₄ + x₁ = y * x₅)

theorem system_solution :
  (system_equations y x₁ x₂ x₃ x₄ x₅) →
  ((y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
   (y = 2 → ∃ t, x₁ = t ∧ x₂ = t ∧ x₃ = t ∧ x₄ = t ∧ x₅ = t) ∧
   (y^2 + y - 1 = 0 → ∃ u v, x₁ = u ∧ x₅ = v ∧ x₂ = y * u - v ∧ x₃ = -y * (u + v) ∧ x₄ = y * v - u ∧
                            (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2))) :=
by sorry


end system_solution_l271_27108


namespace quadratic_vertex_condition_l271_27157

theorem quadratic_vertex_condition (a b c x₀ y₀ : ℝ) (h_a : a ≠ 0) :
  (∀ m n : ℝ, n = a * m^2 + b * m + c → a * (y₀ - n) ≤ 0) →
  y₀ = a * x₀^2 + b * x₀ + c →
  2 * a * x₀ + b = 0 := by
sorry

end quadratic_vertex_condition_l271_27157


namespace f_minimum_and_a_range_l271_27113

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_minimum_and_a_range :
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = -1 / Real.exp 1) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ≥ 1 → f x ≥ a * x - 1) ↔ a ≤ 1) := by
  sorry

end f_minimum_and_a_range_l271_27113


namespace Q_representation_exists_zero_polynomial_l271_27147

variable (x₁ x₂ x₃ x₄ : ℝ)

def Q (x₁ x₂ x₃ x₄ : ℝ) : ℝ := 4 * (x₁^2 + x₂^2 + x₃^2 + x₄^2) - (x₁ + x₂ + x₃ + x₄)^2

def P₁ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ + x₂ - x₃ - x₄
def P₂ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ + x₃ - x₄
def P₃ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ - x₃ + x₄
def P₄ (x₁ x₂ x₃ x₄ : ℝ) : ℝ := 0

theorem Q_representation (x₁ x₂ x₃ x₄ : ℝ) :
  Q x₁ x₂ x₃ x₄ = (P₁ x₁ x₂ x₃ x₄)^2 + (P₂ x₁ x₂ x₃ x₄)^2 + (P₃ x₁ x₂ x₃ x₄)^2 + (P₄ x₁ x₂ x₃ x₄)^2 :=
sorry

theorem exists_zero_polynomial (f g h k : ℝ → ℝ → ℝ → ℝ → ℝ) 
  (hQ : ∀ x₁ x₂ x₃ x₄, Q x₁ x₂ x₃ x₄ = (f x₁ x₂ x₃ x₄)^2 + (g x₁ x₂ x₃ x₄)^2 + (h x₁ x₂ x₃ x₄)^2 + (k x₁ x₂ x₃ x₄)^2) :
  (f = λ _ _ _ _ => 0) ∨ (g = λ _ _ _ _ => 0) ∨ (h = λ _ _ _ _ => 0) ∨ (k = λ _ _ _ _ => 0) :=
sorry

end Q_representation_exists_zero_polynomial_l271_27147


namespace driver_license_exam_results_l271_27170

/-- Represents the probabilities of passing each subject in the driver's license exam -/
structure ExamProbabilities where
  subject1 : ℝ
  subject2 : ℝ
  subject3 : ℝ

/-- Calculates the probability of obtaining a driver's license -/
def probabilityOfObtainingLicense (p : ExamProbabilities) : ℝ :=
  p.subject1 * p.subject2 * p.subject3

/-- Calculates the expected number of attempts during the application process -/
def expectedAttempts (p : ExamProbabilities) : ℝ :=
  1 * (1 - p.subject1) +
  2 * (p.subject1 * (1 - p.subject2)) +
  3 * (p.subject1 * p.subject2)

/-- Theorem stating the probability of obtaining a license and expected attempts -/
theorem driver_license_exam_results (p : ExamProbabilities)
  (h1 : p.subject1 = 0.9)
  (h2 : p.subject2 = 0.7)
  (h3 : p.subject3 = 0.6) :
  probabilityOfObtainingLicense p = 0.378 ∧
  expectedAttempts p = 2.53 := by
  sorry


end driver_license_exam_results_l271_27170


namespace completing_square_transform_l271_27135

theorem completing_square_transform (x : ℝ) :
  x^2 - 2*x - 5 = 0 ↔ (x - 1)^2 = 6 :=
by sorry

end completing_square_transform_l271_27135


namespace xyz_sum_sqrt_l271_27176

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 13)
  (h2 : z + x = 14)
  (h3 : x + y = 15) :
  Real.sqrt (x * y * z * (x + y + z)) = 84 := by
  sorry

end xyz_sum_sqrt_l271_27176


namespace inverse_of_3_mod_229_l271_27194

theorem inverse_of_3_mod_229 : ∃ x : ℕ, x < 229 ∧ (3 * x) % 229 = 1 :=
  by
    use 153
    sorry

end inverse_of_3_mod_229_l271_27194


namespace min_value_theorem_l271_27111

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = 9 ∧ (1/x + 4/y ≥ min) ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1/x₀ + 4/y₀ = min :=
by sorry

end min_value_theorem_l271_27111


namespace rectangle_area_perimeter_relation_l271_27159

theorem rectangle_area_perimeter_relation (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := x + 7
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (area = 2 * perimeter) → (x = 2) :=
by
  sorry

end rectangle_area_perimeter_relation_l271_27159


namespace chessboard_dark_light_difference_l271_27120

/-- Represents a square on the chessboard -/
inductive Square
| Dark
| Light

/-- Represents a row on the chessboard -/
def Row := Vector Square 9

/-- Generates a row starting with the given square color -/
def generateRow (startSquare : Square) : Row := sorry

/-- The chessboard, consisting of 9 rows -/
def Chessboard := Vector Row 9

/-- Generates the chessboard with alternating row starts -/
def generateChessboard : Chessboard := sorry

/-- Counts the number of dark squares in a row -/
def countDarkSquares (row : Row) : Nat := sorry

/-- Counts the number of light squares in a row -/
def countLightSquares (row : Row) : Nat := sorry

/-- Counts the total number of dark squares on the chessboard -/
def totalDarkSquares (board : Chessboard) : Nat := sorry

/-- Counts the total number of light squares on the chessboard -/
def totalLightSquares (board : Chessboard) : Nat := sorry

theorem chessboard_dark_light_difference :
  let board := generateChessboard
  totalDarkSquares board = totalLightSquares board + 1 := by sorry

end chessboard_dark_light_difference_l271_27120


namespace negation_of_existence_negation_of_greater_than_prop_l271_27161

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n : ℕ, p n) ↔ (∀ n : ℕ, ¬ p n) := by sorry

theorem negation_of_greater_than_prop :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end negation_of_existence_negation_of_greater_than_prop_l271_27161


namespace painting_price_increase_l271_27182

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 20 / 100) = 104 / 100 → x = 30 := by
  sorry

end painting_price_increase_l271_27182


namespace inequality_range_l271_27119

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ a ≤ -1 ∨ a ≥ 4 := by
sorry

end inequality_range_l271_27119


namespace solution_set_when_a_is_one_range_of_a_when_f_leq_one_l271_27106

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

-- Theorem for part (2)
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} :=
sorry

end solution_set_when_a_is_one_range_of_a_when_f_leq_one_l271_27106


namespace vector_difference_magnitude_l271_27149

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end vector_difference_magnitude_l271_27149


namespace grapes_per_day_calculation_l271_27118

/-- The number of pickers -/
def num_pickers : ℕ := 235

/-- The number of drums of raspberries filled per day -/
def raspberries_per_day : ℕ := 100

/-- The number of days -/
def num_days : ℕ := 77

/-- The total number of drums filled in 77 days -/
def total_drums : ℕ := 17017

/-- The number of drums of grapes filled per day -/
def grapes_per_day : ℕ := 121

theorem grapes_per_day_calculation :
  grapes_per_day = (total_drums - raspberries_per_day * num_days) / num_days :=
by sorry

end grapes_per_day_calculation_l271_27118


namespace brownies_made_next_morning_l271_27128

def initial_brownies : ℕ := 2 * 12
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def total_next_morning : ℕ := 36

theorem brownies_made_next_morning :
  total_next_morning - (initial_brownies - father_ate - mooney_ate) = 24 := by
  sorry

end brownies_made_next_morning_l271_27128


namespace max_value_quadratic_l271_27197

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (max_val : ℝ), (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 - 2*x'*y' + 3*y'^2 = 10 
    → x'^2 + 2*x'*y' + 3*y'^2 ≤ max_val) ∧ max_val = 20 + 10 * Real.sqrt 3 := by
  sorry

end max_value_quadratic_l271_27197


namespace simultaneous_equations_solution_l271_27136

theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 3) ↔ (∃ x y : ℝ, y = 3 * m * x + 6 ∧ y = (4 * m - 3) * x + 9) :=
by sorry

end simultaneous_equations_solution_l271_27136


namespace greatest_product_sum_2024_l271_27199

theorem greatest_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144 ∧
    ∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) :=
by
  sorry

end greatest_product_sum_2024_l271_27199


namespace other_coin_denomination_l271_27116

/-- Proves that given the problem conditions, the denomination of the other type of coin is 25 paise -/
theorem other_coin_denomination (total_coins : ℕ) (total_value : ℕ) (twenty_paise_coins : ℕ) 
  (h1 : total_coins = 324)
  (h2 : total_value = 7100)  -- 71 Rs in paise
  (h3 : twenty_paise_coins = 200) :
  (total_value - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

end other_coin_denomination_l271_27116


namespace triangle_sum_of_squares_l271_27192

-- Define an equilateral triangle ABC with side length 10
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 10 ∧ dist B C = 10 ∧ dist C A = 10

-- Define points P and Q on AB and AC respectively
def PointP (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ dist A P = 2

def PointQ (A C Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • A + t • C ∧ dist A Q = 2

-- Theorem statement
theorem triangle_sum_of_squares 
  (A B C P Q : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_P : PointP A B P) 
  (h_Q : PointQ A C Q) : 
  (dist C P)^2 + (dist C Q)^2 = 168 := by
  sorry

end triangle_sum_of_squares_l271_27192


namespace correct_quotient_calculation_l271_27190

theorem correct_quotient_calculation (dividend : ℕ) (incorrect_quotient : ℕ) : 
  dividend > 0 →
  incorrect_quotient = 753 →
  dividend = 102 * (incorrect_quotient * 3) →
  dividend % 201 = 0 →
  dividend / 201 = 1146 :=
by
  sorry

end correct_quotient_calculation_l271_27190


namespace min_triangle_area_l271_27195

/-- An acute-angled triangle with an inscribed square --/
structure TriangleWithSquare where
  /-- The base length of the triangle --/
  b : ℝ
  /-- The height of the triangle --/
  h : ℝ
  /-- The side length of the inscribed square --/
  s : ℝ
  /-- The triangle is acute-angled --/
  acute : 0 < b ∧ 0 < h
  /-- The square is inscribed as described --/
  square_inscribed : s = (b * h) / (b + h)

/-- The theorem stating the minimum area of the triangle --/
theorem min_triangle_area (t : TriangleWithSquare) (h_area : t.s^2 = 2017) :
  2 * t.s^2 ≤ (t.b * t.h) / 2 ∧
  ∃ (t' : TriangleWithSquare), t'.s^2 = 2017 ∧ (t'.b * t'.h) / 2 = 2 * t'.s^2 := by
  sorry

#check min_triangle_area

end min_triangle_area_l271_27195


namespace sum_consecutive_odds_not_even_not_div_four_l271_27138

theorem sum_consecutive_odds_not_even_not_div_four (n : ℤ) (m : ℤ) :
  ¬(∃ (k : ℤ), 4 * (n + 1) = 2 * k ∧ ¬(∃ (l : ℤ), 2 * k = 4 * l)) :=
by sorry

end sum_consecutive_odds_not_even_not_div_four_l271_27138


namespace abs_sum_inequality_l271_27187

theorem abs_sum_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -13/2 < x ∧ x < 7/2 := by sorry

end abs_sum_inequality_l271_27187


namespace imaginary_part_of_z_l271_27178

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - I) = abs (1 - I) + I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end imaginary_part_of_z_l271_27178


namespace chlorine_cost_l271_27193

-- Define the pool dimensions
def pool_length : ℝ := 10
def pool_width : ℝ := 8
def pool_depth : ℝ := 6

-- Define the chlorine requirement
def cubic_feet_per_quart : ℝ := 120

-- Define the cost of chlorine
def cost_per_quart : ℝ := 3

-- Theorem statement
theorem chlorine_cost : 
  let pool_volume : ℝ := pool_length * pool_width * pool_depth
  let quarts_needed : ℝ := pool_volume / cubic_feet_per_quart
  let total_cost : ℝ := quarts_needed * cost_per_quart
  total_cost = 12 := by sorry

end chlorine_cost_l271_27193


namespace angle_terminal_side_l271_27127

theorem angle_terminal_side (x : ℝ) (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (x, 4) ∧ P.1 = x ∧ P.2 = 4) → 
  Real.sin α = 4/5 → 
  x = 3 ∨ x = -3 := by
sorry

end angle_terminal_side_l271_27127


namespace farmer_vegetable_difference_l271_27133

/-- Calculates the total difference between initial and remaining tomatoes and carrots --/
def total_difference (initial_tomatoes initial_carrots picked_tomatoes picked_carrots given_tomatoes given_carrots : ℕ) : ℕ :=
  (initial_tomatoes - (initial_tomatoes - picked_tomatoes + given_tomatoes)) +
  (initial_carrots - (initial_carrots - picked_carrots + given_carrots))

/-- Theorem stating the total difference for the given problem --/
theorem farmer_vegetable_difference :
  total_difference 17 13 5 6 3 2 = 6 := by
  sorry

end farmer_vegetable_difference_l271_27133


namespace total_paintable_area_l271_27174

/-- Calculate the total square feet of walls to be painted in bedrooms and hallway -/
theorem total_paintable_area (
  num_bedrooms : ℕ)
  (bedroom_length bedroom_width bedroom_height : ℝ)
  (hallway_length hallway_width hallway_height : ℝ)
  (unpaintable_area_per_bedroom : ℝ)
  (h1 : num_bedrooms = 4)
  (h2 : bedroom_length = 14)
  (h3 : bedroom_width = 11)
  (h4 : bedroom_height = 9)
  (h5 : hallway_length = 20)
  (h6 : hallway_width = 7)
  (h7 : hallway_height = 9)
  (h8 : unpaintable_area_per_bedroom = 70) :
  (num_bedrooms * (2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area_per_bedroom)) +
  (2 * (hallway_length * hallway_height + hallway_width * hallway_height)) = 2006 := by
  sorry

end total_paintable_area_l271_27174


namespace surface_area_of_sawed_cube_l271_27164

/-- The total surface area of rectangular blocks obtained by sawing a unit cube -/
def total_surface_area (length_cuts width_cuts height_cuts : ℕ) : ℝ :=
  let original_surface := 6
  let new_surface := (length_cuts + 1) * (width_cuts + 1) * 2 +
                     (length_cuts + 1) * (height_cuts + 1) * 2 +
                     (width_cuts + 1) * (height_cuts + 1) * 2
  original_surface + new_surface - 6

/-- Theorem: The total surface area of 24 rectangular blocks obtained by sawing a unit cube
    1 time along length, 2 times along width, and 3 times along height is 18 square meters -/
theorem surface_area_of_sawed_cube : total_surface_area 1 2 3 = 18 := by
  sorry

end surface_area_of_sawed_cube_l271_27164


namespace angle_reflection_l271_27112

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 180 + 360 * (k : Real) < α ∧ α < 270 + 360 * (k : Real)

def is_in_fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 270 + 360 * (k : Real) < α ∧ α < 360 + 360 * (k : Real)

theorem angle_reflection (α : Real) :
  is_in_third_quadrant α → is_in_fourth_quadrant (180 - α) := by
  sorry

end angle_reflection_l271_27112


namespace wire_service_reporters_theorem_l271_27154

/-- Represents the percentage of reporters in a wire service -/
structure ReporterPercentage where
  local_politics : Real
  not_politics : Real
  politics_not_local : Real

/-- Given the percentages of reporters covering local politics and not covering politics,
    calculates the percentage of reporters covering politics but not local politics -/
def calculate_politics_not_local (rp : ReporterPercentage) : Real :=
  100 - rp.not_politics - rp.local_politics

/-- Theorem stating that given the specific percentages in the problem,
    the percentage of reporters covering politics but not local politics is 2.14285714285714% -/
theorem wire_service_reporters_theorem (rp : ReporterPercentage)
  (h1 : rp.local_politics = 5)
  (h2 : rp.not_politics = 92.85714285714286) :
  calculate_politics_not_local rp = 2.14285714285714 := by
  sorry

#eval calculate_politics_not_local { local_politics := 5, not_politics := 92.85714285714286, politics_not_local := 0 }

end wire_service_reporters_theorem_l271_27154


namespace equation_solution_l271_27100

theorem equation_solution : ∃ x : ℚ, 
  (((5 - 4*x) / (5 + 4*x) + 3) / (3 + (5 + 4*x) / (5 - 4*x))) - 
  (((5 - 4*x) / (5 + 4*x) + 2) / (2 + (5 + 4*x) / (5 - 4*x))) = 
  ((5 - 4*x) / (5 + 4*x) + 1) / (1 + (5 + 4*x) / (5 - 4*x)) ∧ 
  x = -5/14 := by
sorry

end equation_solution_l271_27100


namespace opposite_of_negative_one_sixth_l271_27162

theorem opposite_of_negative_one_sixth (x : ℚ) : x = -1/6 → -x = 1/6 := by
  sorry

end opposite_of_negative_one_sixth_l271_27162


namespace max_self_intersections_correct_max_self_intersections_13_max_self_intersections_1950_l271_27122

/-- The maximum number of self-intersection points in a closed polygonal line with n segments -/
def max_self_intersections (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 3) / 2
  else
    n * (n - 4) / 2 + 1

theorem max_self_intersections_correct (n : ℕ) (h : n ≥ 3) :
  max_self_intersections n =
    if n % 2 = 1 then
      n * (n - 3) / 2
    else
      n * (n - 4) / 2 + 1 :=
by sorry

-- Specific cases
theorem max_self_intersections_13 :
  max_self_intersections 13 = 65 :=
by sorry

theorem max_self_intersections_1950 :
  max_self_intersections 1950 = 1897851 :=
by sorry

end max_self_intersections_correct_max_self_intersections_13_max_self_intersections_1950_l271_27122


namespace trucks_given_to_jeff_l271_27166

-- Define the variables
def initial_trucks : ℕ := 51
def remaining_trucks : ℕ := 38

-- Define the theorem
theorem trucks_given_to_jeff : 
  initial_trucks - remaining_trucks = 13 := by
  sorry

end trucks_given_to_jeff_l271_27166


namespace watch_cost_price_l271_27173

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (C : ℝ), 
  (C > 0) ∧ 
  (0.64 * C + 140 = 1.04 * C) ∧ 
  (C = 350) := by
  sorry

end watch_cost_price_l271_27173


namespace smallest_set_size_for_divisibility_by_20_l271_27117

theorem smallest_set_size_for_divisibility_by_20 :
  ∃ (n : ℕ), n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T →
    a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d ∨
    ¬(20 ∣ (a + b - c - d))) :=
by
  sorry

#check smallest_set_size_for_divisibility_by_20

end smallest_set_size_for_divisibility_by_20_l271_27117


namespace permutation_sum_squares_values_l271_27177

theorem permutation_sum_squares_values (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) : 
  ∃! (s : Finset ℝ), 
    s.card = 3 ∧ 
    (∀ (x y z t : ℝ), ({x, y, z, t} : Finset ℝ) = {a, b, c, d} → 
      ((x - y)^2 + (y - z)^2 + (z - t)^2 + (t - x)^2) ∈ s) := by
  sorry

end permutation_sum_squares_values_l271_27177


namespace reflection_about_y_eq_neg_x_l271_27123

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

/-- The original point -/
def original_point : ℝ × ℝ := (3, -7)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (7, -3)

theorem reflection_about_y_eq_neg_x :
  reflect_about_y_eq_neg_x original_point = reflected_point := by
  sorry

end reflection_about_y_eq_neg_x_l271_27123


namespace intersection_A_complement_B_l271_27150

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | x ≥ 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end intersection_A_complement_B_l271_27150


namespace simple_interest_problem_l271_27144

/-- Proves that given the conditions of the problem, the principal amount is 2800 --/
theorem simple_interest_problem (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2240 → P = 2800 := by sorry

end simple_interest_problem_l271_27144


namespace eve_last_student_l271_27186

/-- Represents the students in the circle -/
inductive Student
| Alan
| Bob
| Cara
| Dan
| Eve

/-- The order of students in the circle -/
def initialOrder : List Student := [Student.Alan, Student.Bob, Student.Cara, Student.Dan, Student.Eve]

/-- Checks if a number is a multiple of 7 or contains the digit 6 -/
def isEliminationNumber (n : Nat) : Bool :=
  n % 7 == 0 || n.repr.contains '6'

/-- Simulates the elimination process and returns the last student remaining -/
def lastStudent (order : List Student) : Student :=
  sorry

/-- Theorem stating that Eve is the last student remaining -/
theorem eve_last_student : lastStudent initialOrder = Student.Eve :=
  sorry

end eve_last_student_l271_27186


namespace fraction_sum_equality_l271_27114

theorem fraction_sum_equality (p q m n : ℕ+) (x : ℚ) 
  (h1 : (p : ℚ) / q = (m : ℚ) / n)
  (h2 : (p : ℚ) / q = 4 / 5)
  (h3 : x = 1 / 7) :
  x + ((2 * q - p + 3 * m - 2 * n) : ℚ) / (2 * q + p - m + n) = 71 / 105 := by
  sorry

end fraction_sum_equality_l271_27114
