import Mathlib

namespace trigonometric_equation_solution_l1900_190043

theorem trigonometric_equation_solution (x : ℝ) :
  0.5 * (Real.cos (5 * x) + Real.cos (7 * x)) - (Real.cos (2 * x))^2 + (Real.sin (3 * x))^2 = 0 ↔
  (∃ k : ℤ, x = π / 2 * (2 * k + 1)) ∨ (∃ k : ℤ, x = 2 * k * π / 11) :=
by sorry

end trigonometric_equation_solution_l1900_190043


namespace special_sequence_is_arithmetic_l1900_190020

/-- A sequence satisfying specific conditions -/
def SpecialSequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a (n + 1) > a n ∧ a n > 0) ∧
  (∀ n : ℕ+, a n - 1 / a n < n ∧ n < a n + 1 / a n) ∧
  (∃ m : ℕ+, m ≥ 2 ∧ ∀ n : ℕ+, a (m * n) = m * a n)

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Theorem: A special sequence is an arithmetic sequence -/
theorem special_sequence_is_arithmetic (a : ℕ+ → ℝ) 
    (h : SpecialSequence a) : ArithmeticSequence a := by
  sorry

end special_sequence_is_arithmetic_l1900_190020


namespace combine_like_terms_l1900_190060

theorem combine_like_terms (x y : ℝ) :
  -x^2 * y + 3/4 * x^2 * y = -(1/4) * x^2 * y := by
  sorry

end combine_like_terms_l1900_190060


namespace sqrt_450_simplified_l1900_190042

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplified_l1900_190042


namespace factorial_equality_l1900_190025

theorem factorial_equality : 5 * 8 * 2 * 63 = Nat.factorial 7 := by
  sorry

end factorial_equality_l1900_190025


namespace white_washing_cost_l1900_190062

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the window dimensions
def window_height : ℝ := 4
def window_width : ℝ := 3

-- Define the number of windows
def num_windows : ℕ := 3

-- Define the cost per square foot
def cost_per_sqft : ℝ := 7

-- Theorem statement
theorem white_washing_cost :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := window_height * window_width * num_windows
  let adjusted_wall_area := total_wall_area - door_area - window_area
  adjusted_wall_area * cost_per_sqft = 6342 := by
  sorry


end white_washing_cost_l1900_190062


namespace parabola_intercepts_sum_l1900_190026

-- Define the parabola
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define a as the x-intercept
def a : ℝ := parabola 0

-- Define b and c as y-intercepts
noncomputable def b : ℝ := (9 - Real.sqrt 33) / 6
noncomputable def c : ℝ := (9 + Real.sqrt 33) / 6

-- Theorem statement
theorem parabola_intercepts_sum : a + b + c = 7 := by
  sorry

end parabola_intercepts_sum_l1900_190026


namespace coin_problem_l1900_190041

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10
  | CoinType.Quarter => 25
  | CoinType.HalfDollar => 50

/-- A collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat

/-- The total number of coins in a collection --/
def CoinCollection.totalCoins (c : CoinCollection) : Nat :=
  c.pennies + c.nickels + c.dimes + c.quarters + c.halfDollars

/-- The total value of a coin collection in cents --/
def CoinCollection.totalValue (c : CoinCollection) : Nat :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter +
  c.halfDollars * coinValue CoinType.HalfDollar

/-- The main theorem to prove --/
theorem coin_problem :
  ∀ (c : CoinCollection),
    c.totalCoins = 12 ∧
    c.totalValue = 166 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.quarters ≥ 1 ∧
    c.halfDollars ≥ 1
    →
    c.quarters = 3 :=
by sorry

end coin_problem_l1900_190041


namespace sine_cosine_inequality_l1900_190015

theorem sine_cosine_inequality (n : ℕ+) (x : ℝ) :
  (Real.sin (2 * x))^(n : ℝ) + (Real.sin x^(n : ℝ) - Real.cos x^(n : ℝ))^2 ≤ 1 := by
  sorry

end sine_cosine_inequality_l1900_190015


namespace windows_preference_l1900_190066

/-- Given a survey of college students about computer brand preferences,
    this theorem proves the number of students preferring Windows. -/
theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) : 
  total = 210 →
  mac = 60 →
  no_pref = 90 →
  ∃ (windows : ℕ), 
    windows = total - (mac + mac / 3 + no_pref) ∧
    windows = 40 :=
by sorry

end windows_preference_l1900_190066


namespace symmetric_intersection_theorem_l1900_190037

/-- A line that intersects a circle at two points symmetric about another line -/
structure SymmetricIntersection where
  /-- The coefficient of x in the line equation ax + 2y - 2 = 0 -/
  a : ℝ
  /-- The first intersection point -/
  A : ℝ × ℝ
  /-- The second intersection point -/
  B : ℝ × ℝ

/-- The line ax + 2y - 2 = 0 intersects the circle (x-1)² + (y+1)² = 6 -/
def intersects_circle (si : SymmetricIntersection) : Prop :=
  let (x₁, y₁) := si.A
  let (x₂, y₂) := si.B
  si.a * x₁ + 2 * y₁ - 2 = 0 ∧
  si.a * x₂ + 2 * y₂ - 2 = 0 ∧
  (x₁ - 1)^2 + (y₁ + 1)^2 = 6 ∧
  (x₂ - 1)^2 + (y₂ + 1)^2 = 6

/-- A and B are symmetric with respect to the line x + y = 0 -/
def symmetric_about_line (si : SymmetricIntersection) : Prop :=
  let (x₁, y₁) := si.A
  let (x₂, y₂) := si.B
  x₁ + y₁ = -(x₂ + y₂)

/-- The main theorem: if the conditions are met, then a = -2 -/
theorem symmetric_intersection_theorem (si : SymmetricIntersection) :
  intersects_circle si → symmetric_about_line si → si.a = -2 :=
sorry

end symmetric_intersection_theorem_l1900_190037


namespace function_extension_l1900_190056

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem function_extension (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_symmetry : ∀ x, f (2 - x) = f x)
  (h_base : ∀ x ∈ Set.Ioo 0 1, f x = Real.log x) :
  (∀ x ∈ Set.Icc (-1) 0, f x = -Real.log (-x)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Ioo (4 * k) (4 * k + 1), f x = Real.log (x - 4 * k)) :=
sorry

end function_extension_l1900_190056


namespace sqrt_eighteen_div_sqrt_two_equals_three_l1900_190084

theorem sqrt_eighteen_div_sqrt_two_equals_three :
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end sqrt_eighteen_div_sqrt_two_equals_three_l1900_190084


namespace two_triangles_max_parts_two_rectangles_max_parts_two_n_gons_max_parts_l1900_190085

/-- The maximum number of parts into which two polygons can divide a plane -/
def max_parts (sides : ℕ) : ℕ := 2 * sides + 2

/-- Two triangles can divide a plane into at most 8 parts -/
theorem two_triangles_max_parts : max_parts 3 = 8 := by sorry

/-- Two rectangles can divide a plane into at most 10 parts -/
theorem two_rectangles_max_parts : max_parts 4 = 10 := by sorry

/-- Two convex n-gons can divide a plane into at most 2n + 2 parts -/
theorem two_n_gons_max_parts (n : ℕ) : max_parts n = 2 * n + 2 := by sorry

end two_triangles_max_parts_two_rectangles_max_parts_two_n_gons_max_parts_l1900_190085


namespace garden_area_l1900_190068

/-- Represents a rectangular garden with specific properties. -/
structure RectangularGarden where
  width : Real
  length : Real
  perimeter : Real
  area : Real
  length_condition : length = 3 * width + 10
  perimeter_condition : perimeter = 2 * (length + width)
  area_condition : area = length * width

/-- Theorem stating the area of a specific rectangular garden. -/
theorem garden_area (g : RectangularGarden) (h : g.perimeter = 400) :
  g.area = 7243.75 := by
  sorry


end garden_area_l1900_190068


namespace perpendicular_line_equation_triangle_height_equation_l1900_190089

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem 1
theorem perpendicular_line_equation (A : Point) (h : A.x = -2 ∧ A.y = 3) :
  ∃ l : Line, perpendicular l ⟨A.y, -A.x, 0⟩ ∧ on_line A l ∧ l.a = 2 ∧ l.b = -3 ∧ l.c = 13 :=
sorry

-- Theorem 2
theorem triangle_height_equation (A B C : Point) 
  (hA : A.x = 4 ∧ A.y = 0) (hB : B.x = 6 ∧ B.y = 7) (hC : C.x = 0 ∧ C.y = 3) :
  ∃ l : Line, perpendicular l ⟨B.y - A.y, A.x - B.x, B.x * A.y - A.x * B.y⟩ ∧ 
    on_line C l ∧ l.a = 2 ∧ l.b = 7 ∧ l.c = -21 :=
sorry

end perpendicular_line_equation_triangle_height_equation_l1900_190089


namespace equal_angles_point_exists_l1900_190009

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : Point2D) : ℝ := sorry

-- Define the theorem
theorem equal_angles_point_exists (A B C D : Point2D) 
  (h_collinear : ∃ (t : ℝ), B.x = A.x + t * (D.x - A.x) ∧ 
                             B.y = A.y + t * (D.y - A.y) ∧ 
                             C.x = A.x + t * (D.x - A.x) ∧ 
                             C.y = A.y + t * (D.y - A.y)) :
  ∃ (M : Point2D), angle A M B = angle B M C ∧ angle B M C = angle C M D := by
  sorry

end equal_angles_point_exists_l1900_190009


namespace min_sum_of_dimensions_l1900_190034

def is_valid_box (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1729

theorem min_sum_of_dimensions :
  ∃ (a b c : ℕ), is_valid_box a b c ∧
  ∀ (x y z : ℕ), is_valid_box x y z → a + b + c ≤ x + y + z ∧
  a + b + c = 39 :=
sorry

end min_sum_of_dimensions_l1900_190034


namespace copy_pages_with_discount_l1900_190018

/-- Calculates the number of pages that can be copied given a certain amount of cents,
    considering a discount where for every 100 pages, an additional 10 pages are free. -/
def pages_copied (cents : ℕ) : ℕ :=
  let base_pages := (cents * 5) / 10
  let free_pages := (base_pages / 100) * 10
  base_pages + free_pages

/-- Proves that 5000 cents allows copying 2750 pages with the given pricing and discount. -/
theorem copy_pages_with_discount :
  pages_copied 5000 = 2750 := by
  sorry

end copy_pages_with_discount_l1900_190018


namespace largest_three_digit_square_base_seven_l1900_190094

/-- The largest integer whose square has exactly 3 digits when written in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def to_base_seven (n : ℕ) : ℕ :=
  if n < 7 then n
  else 10 * to_base_seven (n / 7) + n % 7

theorem largest_three_digit_square_base_seven :
  (M^2 ≥ 7^2) ∧ 
  (M^2 < 7^3) ∧ 
  (∀ n : ℕ, n > M → n^2 ≥ 7^3) ∧
  (to_base_seven M = 66) :=
sorry

end largest_three_digit_square_base_seven_l1900_190094


namespace exists_integer_sqrt_8m_l1900_190082

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℕ) = k^2 := by
  sorry

end exists_integer_sqrt_8m_l1900_190082


namespace largest_possible_number_david_l1900_190005

/-- Represents a decimal number with up to two digits before and after the decimal point -/
structure DecimalNumber :=
  (beforeDecimal : Fin 100)
  (afterDecimal : Fin 100)

/-- Checks if a DecimalNumber has mutually different digits -/
def hasMutuallyDifferentDigits (n : DecimalNumber) : Prop :=
  sorry

/-- Checks if a DecimalNumber has exactly two identical digits -/
def hasExactlyTwoIdenticalDigits (n : DecimalNumber) : Prop :=
  sorry

/-- Converts a DecimalNumber to a rational number -/
def toRational (n : DecimalNumber) : ℚ :=
  sorry

/-- The sum of two DecimalNumbers -/
def sum (a b : DecimalNumber) : ℚ :=
  toRational a + toRational b

theorem largest_possible_number_david
  (jana david : DecimalNumber)
  (h_sum : sum jana david = 11.11)
  (h_david_digits : hasMutuallyDifferentDigits david)
  (h_jana_digits : hasExactlyTwoIdenticalDigits jana) :
  toRational david ≤ 0.9 :=
sorry

end largest_possible_number_david_l1900_190005


namespace tetrahedron_properties_l1900_190047

def A₁ : ℝ × ℝ × ℝ := (2, -1, 2)
def A₂ : ℝ × ℝ × ℝ := (1, 2, -1)
def A₃ : ℝ × ℝ × ℝ := (3, 2, 1)
def A₄ : ℝ × ℝ × ℝ := (-4, 2, 5)

def tetrahedron_volume (A B C D : ℝ × ℝ × ℝ) : ℝ := sorry

def tetrahedron_height (A B C D : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  tetrahedron_volume A₁ A₂ A₃ A₄ = 11 ∧
  tetrahedron_height A₄ A₁ A₂ A₃ = 3 * Real.sqrt (11 / 2) := by
  sorry

end tetrahedron_properties_l1900_190047


namespace product_sum_of_three_numbers_l1900_190092

theorem product_sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149) 
  (h2 : a + b + c = 17) : 
  a * b + b * c + c * a = 70 := by
sorry

end product_sum_of_three_numbers_l1900_190092


namespace amount_spent_first_shop_l1900_190074

/-- The amount spent on books from the first shop -/
def amount_first_shop : ℕ := 1500

/-- The number of books bought from the first shop -/
def books_first_shop : ℕ := 55

/-- The number of books bought from the second shop -/
def books_second_shop : ℕ := 60

/-- The amount spent on books from the second shop -/
def amount_second_shop : ℕ := 340

/-- The average price per book -/
def average_price : ℕ := 16

/-- Theorem stating that the amount spent on the first shop is 1500,
    given the conditions of the problem -/
theorem amount_spent_first_shop :
  amount_first_shop = 1500 :=
by
  sorry

end amount_spent_first_shop_l1900_190074


namespace absolute_value_equation_solution_difference_l1900_190059

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|x₁ - 4| = 15 ∧ |x₂ - 4| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end absolute_value_equation_solution_difference_l1900_190059


namespace line_passes_through_fixed_point_l1900_190024

/-- The line (k+1)x-(2k-1)y+3k=0 always passes through the point (-1, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 1) * (-1) - (2 * k - 1) * 1 + 3 * k = 0 := by
sorry

end line_passes_through_fixed_point_l1900_190024


namespace adam_bought_26_books_l1900_190073

/-- Represents Adam's bookcase and book shopping scenario -/
structure Bookcase where
  shelves : Nat
  booksPerShelf : Nat
  initialBooks : Nat
  leftoverBooks : Nat

/-- Calculates the number of books Adam bought -/
def booksBought (b : Bookcase) : Nat :=
  b.shelves * b.booksPerShelf + b.leftoverBooks - b.initialBooks

/-- Theorem stating that Adam bought 26 books -/
theorem adam_bought_26_books (b : Bookcase) 
    (h1 : b.shelves = 4)
    (h2 : b.booksPerShelf = 20)
    (h3 : b.initialBooks = 56)
    (h4 : b.leftoverBooks = 2) : 
  booksBought b = 26 := by
  sorry

end adam_bought_26_books_l1900_190073


namespace circle_equation_l1900_190021

/-- Given a circle with center at (a,1) tangent to two lines, prove its standard equation -/
theorem circle_equation (a : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x y : ℝ), (2*x - y + 4 = 0 ∨ 2*x - y - 6 = 0) → 
      ((x - a)^2 + (y - 1)^2 = r^2))) →
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 5) :=
by sorry

end circle_equation_l1900_190021


namespace words_per_page_l1900_190014

theorem words_per_page (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 150 →
  max_words_per_page = 120 →
  total_words_mod = 270 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 221 = total_words_mod % 221 ∧
    words_per_page = 107 :=
by sorry

end words_per_page_l1900_190014


namespace ellipse_and_circle_intersection_l1900_190069

/-- The ellipse C₁ -/
def C₁ (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- The parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle C₃ -/
def C₃ (x y x₀ y₀ r : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The point P -/
def P (x y : ℝ) : Prop := C₁ x y 2 (Real.sqrt 3) ∧ C₂ x y ∧ x > 0 ∧ y > 0

/-- The point T -/
def T (x y : ℝ) : Prop := C₂ x y

theorem ellipse_and_circle_intersection 
  (a b : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : ∃ x y, P x y ∧ (x - 1)^2 + y^2 = (5/3)^2) 
  (h₄ : ∀ x₀ y₀ r, T x₀ y₀ → C₃ 0 2 x₀ y₀ r → C₃ 0 (-2) x₀ y₀ r → r^2 = 4 + x₀^2) :
  (∀ x y, C₁ x y a b ↔ C₁ x y 2 (Real.sqrt 3)) ∧ 
  (∀ x₀ y₀ r, T x₀ y₀ → C₃ 2 0 x₀ y₀ r) :=
sorry

end ellipse_and_circle_intersection_l1900_190069


namespace rachel_homework_difference_l1900_190004

theorem rachel_homework_difference :
  let math_pages : ℕ := 2
  let reading_pages : ℕ := 3
  let total_pages : ℕ := 15
  let biology_pages : ℕ := total_pages - (math_pages + reading_pages)
  biology_pages - reading_pages = 7 :=
by sorry

end rachel_homework_difference_l1900_190004


namespace star_commutative_star_not_distributive_star_has_identity_star_identity_is_neg_one_l1900_190087

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Theorem for commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Theorem for non-distributivity
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

-- Theorem for existence of identity element
theorem star_has_identity : ∃ e : ℝ, ∀ x : ℝ, star x e = x := by sorry

-- Theorem that -1 is the identity element
theorem star_identity_is_neg_one : ∀ x : ℝ, star x (-1) = x := by sorry

end star_commutative_star_not_distributive_star_has_identity_star_identity_is_neg_one_l1900_190087


namespace hash_difference_eight_five_l1900_190022

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- Theorem statement
theorem hash_difference_eight_five : hash 8 5 - hash 5 8 = -12 := by sorry

end hash_difference_eight_five_l1900_190022


namespace hamburgers_left_over_example_l1900_190093

/-- Given a restaurant that made some hamburgers and served some of them,
    calculate the number of hamburgers left over. -/
def hamburgers_left_over (made served : ℕ) : ℕ :=
  made - served

/-- Theorem stating that if 9 hamburgers were made and 3 were served,
    then 6 hamburgers were left over. -/
theorem hamburgers_left_over_example :
  hamburgers_left_over 9 3 = 6 := by
  sorry

end hamburgers_left_over_example_l1900_190093


namespace original_price_calculation_l1900_190054

theorem original_price_calculation (reduced_price : ℝ) (reduction_percentage : ℝ) 
  (h1 : reduced_price = 6)
  (h2 : reduction_percentage = 0.25)
  (h3 : reduced_price = reduction_percentage * original_price) :
  original_price = 24 := by
  sorry

end original_price_calculation_l1900_190054


namespace abs_a_plus_inv_a_geq_two_l1900_190008

theorem abs_a_plus_inv_a_geq_two (a : ℝ) (h : a ≠ 0) : |a + 1/a| ≥ 2 := by
  sorry

end abs_a_plus_inv_a_geq_two_l1900_190008


namespace two_six_digit_squares_decomposable_l1900_190046

/-- A function that checks if a number is a two-digit square -/
def isTwoDigitSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, 4 ≤ k ∧ k ≤ 9 ∧ n = k^2

/-- A function that checks if a 6-digit number can be decomposed into three two-digit squares -/
def isDecomposableIntoThreeTwoDigitSquares (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    isTwoDigitSquare a ∧
    isTwoDigitSquare b ∧
    isTwoDigitSquare c ∧
    n = a * 10000 + b * 100 + c

/-- The main theorem stating that there are exactly two 6-digit perfect squares
    that can be decomposed into three two-digit perfect squares -/
theorem two_six_digit_squares_decomposable :
  ∃! (s : Finset ℕ),
    s.card = 2 ∧
    (∀ n ∈ s, 100000 ≤ n ∧ n < 1000000) ∧
    (∀ n ∈ s, ∃ k : ℕ, n = k^2) ∧
    (∀ n ∈ s, isDecomposableIntoThreeTwoDigitSquares n) :=
  sorry

end two_six_digit_squares_decomposable_l1900_190046


namespace population_characteristics_changeable_l1900_190088

/-- Represents a population of organisms -/
structure Population where
  species : Type
  individuals : Set species
  space : Type
  time : Type

/-- Characteristics of a population -/
structure PopulationCharacteristics where
  density : ℝ
  birth_rate : ℝ
  death_rate : ℝ
  immigration_rate : ℝ
  age_composition : Set ℕ
  sex_ratio : ℝ

/-- A population has characteristics that can change over time -/
def population_characteristics_can_change (p : Population) : Prop :=
  ∃ (t₁ t₂ : p.time) (c₁ c₂ : PopulationCharacteristics),
    t₁ ≠ t₂ → c₁ ≠ c₂

/-- The main theorem stating that population characteristics can change over time -/
theorem population_characteristics_changeable :
  ∀ (p : Population), population_characteristics_can_change p :=
sorry

end population_characteristics_changeable_l1900_190088


namespace product_zero_given_sum_and_seventh_power_sum_zero_l1900_190027

theorem product_zero_given_sum_and_seventh_power_sum_zero 
  (w x y z : ℝ) 
  (sum_zero : w + x + y + z = 0) 
  (seventh_power_sum_zero : w^7 + x^7 + y^7 + z^7 = 0) : 
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end product_zero_given_sum_and_seventh_power_sum_zero_l1900_190027


namespace parentheses_removal_l1900_190058

theorem parentheses_removal (a b c d : ℝ) : a - (b - c + d) = a - b + c - d := by
  sorry

end parentheses_removal_l1900_190058


namespace simplify_nested_roots_l1900_190028

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/8))^(1/4))^2 * (((b^16)^(1/4))^(1/8))^2 = b^2 := by
  sorry

end simplify_nested_roots_l1900_190028


namespace smallest_NPP_l1900_190016

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def last_two_digits_equal (n : ℕ) : Prop :=
  (n % 100) % 11 = 0

theorem smallest_NPP :
  ∃ (M N P : ℕ),
    is_two_digit_with_equal_digits (11 * M) ∧
    is_one_digit N ∧
    is_three_digit (100 * N + 10 * P + P) ∧
    11 * M * N = 100 * N + 10 * P + P ∧
    (∀ (M' N' P' : ℕ),
      is_two_digit_with_equal_digits (11 * M') →
      is_one_digit N' →
      is_three_digit (100 * N' + 10 * P' + P') →
      11 * M' * N' = 100 * N' + 10 * P' + P' →
      100 * N + 10 * P + P ≤ 100 * N' + 10 * P' + P') ∧
    M = 2 ∧ N = 3 ∧ P = 6 :=
by sorry

end smallest_NPP_l1900_190016


namespace women_married_fraction_l1900_190049

theorem women_married_fraction (total : ℕ) (h_total_pos : total > 0) :
  let women := (76 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  married_women / women = 13 / 19 := by
sorry

end women_married_fraction_l1900_190049


namespace circle_sum_radii_geq_rectangle_sides_l1900_190040

/-- Given a rectangle ABCD with sides a and b, and two circles k₁ and k₂ where:
    - k₁ passes through A and B and is tangent to CD
    - k₂ passes through A and D and is tangent to BC
    - r₁ and r₂ are the radii of k₁ and k₂ respectively
    Prove that r₁ + r₂ ≥ 5/8 * (a + b) -/
theorem circle_sum_radii_geq_rectangle_sides 
  (a b r₁ r₂ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (hr₁ : r₁ = (a^2 + 4*b^2) / (8*b))
  (hr₂ : r₂ = (b^2 + 4*a^2) / (8*a)) :
  r₁ + r₂ ≥ 5/8 * (a + b) := by
  sorry

end circle_sum_radii_geq_rectangle_sides_l1900_190040


namespace T_increasing_T_not_perfect_square_non_perfect_square_in_T_T_2012th_term_l1900_190095

/-- The sequence of positive integers that are not perfect squares -/
def T : ℕ → ℕ := sorry

/-- T is increasing -/
theorem T_increasing : ∀ n : ℕ, T n < T (n + 1) := sorry

/-- T consists of non-perfect squares -/
theorem T_not_perfect_square : ∀ n : ℕ, ¬ ∃ m : ℕ, T n = m^2 := sorry

/-- Every non-perfect square is in T -/
theorem non_perfect_square_in_T : ∀ k : ℕ, (¬ ∃ m : ℕ, k = m^2) → ∃ n : ℕ, T n = k := sorry

/-- The 2012th term of T is 2057 -/
theorem T_2012th_term : T 2011 = 2057 := sorry

end T_increasing_T_not_perfect_square_non_perfect_square_in_T_T_2012th_term_l1900_190095


namespace arithmetic_sequence_sum_specific_l1900_190053

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific : 
  arithmetic_sequence_sum 1 2 17 = 289 := by
  sorry

end arithmetic_sequence_sum_specific_l1900_190053


namespace expected_value_of_game_l1900_190011

-- Define the die
def die := Finset.range 10

-- Define prime numbers on the die
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define composite numbers on the die
def composites : Finset ℕ := {4, 6, 8, 9, 10}

-- Define the winnings function
def winnings (n : ℕ) : ℚ :=
  if n ∈ primes then n
  else if n ∈ composites then 0
  else -5

-- Theorem statement
theorem expected_value_of_game : 
  (die.sum (fun i => winnings i) : ℚ) / 10 = 18 / 100 := by sorry

end expected_value_of_game_l1900_190011


namespace sandwiches_per_person_l1900_190013

def mini_croissants_per_set : ℕ := 12
def cost_per_set : ℕ := 8
def committee_size : ℕ := 24
def total_spent : ℕ := 32

theorem sandwiches_per_person :
  (total_spent / cost_per_set) * mini_croissants_per_set / committee_size = 2 := by
  sorry

end sandwiches_per_person_l1900_190013


namespace foreign_trade_income_l1900_190080

/-- Foreign trade income problem -/
theorem foreign_trade_income 
  (m : ℝ) -- Foreign trade income in 2001 (billion yuan)
  (x : ℝ) -- Percentage increase in 2002
  (n : ℝ) -- Foreign trade income in 2003 (billion yuan)
  (h1 : x > 0) -- Ensure x is positive
  (h2 : m > 0) -- Ensure initial income is positive
  : n = m * (1 + x / 100) * (1 + 2 * x / 100) :=
by sorry

end foreign_trade_income_l1900_190080


namespace volume_of_tetrahedron_OCDE_l1900_190077

/-- Square ABCD with side length 2 -/
def square_ABCD : Set (ℝ × ℝ) := sorry

/-- Point E is the midpoint of AB -/
def point_E : ℝ × ℝ := sorry

/-- Point O is formed when A and B coincide after folding -/
def point_O : ℝ × ℝ := sorry

/-- Triangle OCD formed after folding -/
def triangle_OCD : Set (ℝ × ℝ) := sorry

/-- Tetrahedron O-CDE formed after folding -/
def tetrahedron_OCDE : Set (ℝ × ℝ × ℝ) := sorry

/-- Volume of a tetrahedron -/
def tetrahedron_volume (t : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem volume_of_tetrahedron_OCDE : 
  tetrahedron_volume tetrahedron_OCDE = Real.sqrt 3 / 3 := by
  sorry

end volume_of_tetrahedron_OCDE_l1900_190077


namespace f_max_min_l1900_190083

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem f_max_min :
  (∀ x, f x ≤ 15) ∧ (∃ x, f x = 15) ∧
  (∀ x, f x ≥ -1) ∧ (∃ x, f x = -1) :=
sorry

end f_max_min_l1900_190083


namespace inverse_of_proposition_l1900_190098

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop := x^2 > 0 → x < 0

-- Theorem stating that inverse_proposition is the inverse of original_proposition
theorem inverse_of_proposition :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, inverse_proposition x) :=
sorry

end inverse_of_proposition_l1900_190098


namespace work_days_per_week_l1900_190099

/-- Proves that Terry and Jordan work 7 days a week given their daily incomes and weekly income difference -/
theorem work_days_per_week 
  (terry_daily_income : ℕ) 
  (jordan_daily_income : ℕ) 
  (weekly_income_difference : ℕ) 
  (h1 : terry_daily_income = 24)
  (h2 : jordan_daily_income = 30)
  (h3 : weekly_income_difference = 42) :
  ∃ d : ℕ, d = 7 ∧ d * jordan_daily_income - d * terry_daily_income = weekly_income_difference := by
  sorry

end work_days_per_week_l1900_190099


namespace largest_number_with_sum_14_l1900_190006

def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def all_valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_sum_14 :
  ∀ n : ℕ,
    all_valid_digits n →
    digit_sum n = 14 →
    n ≤ 3332 :=
sorry

end largest_number_with_sum_14_l1900_190006


namespace symmetry_point_l1900_190086

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricToYAxis (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem symmetry_point : 
  let M : Point2D := ⟨3, -4⟩
  let N : Point2D := ⟨-3, -4⟩
  symmetricToYAxis M N → N = ⟨-3, -4⟩ := by
  sorry

end symmetry_point_l1900_190086


namespace age_sum_in_six_years_l1900_190023

/-- Melanie's current age -/
def melanie_age : ℕ := sorry

/-- Phil's current age -/
def phil_age : ℕ := sorry

/-- The sum of Melanie's and Phil's ages 6 years from now is 42, 
    given that in 10 years, the product of their ages will be 400 more than it is now. -/
theorem age_sum_in_six_years : 
  (melanie_age + 10) * (phil_age + 10) = melanie_age * phil_age + 400 →
  (melanie_age + 6) + (phil_age + 6) = 42 := by sorry

end age_sum_in_six_years_l1900_190023


namespace perpendicular_implies_parallel_perpendicular_plane_and_contained_implies_perpendicular_l1900_190052

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- Non-coincident lines and planes
variable (m n : Line)
variable (α β : Plane)
variable (h_diff_lines : m ≠ n)
variable (h_diff_planes : α ≠ β)

-- Theorem 1
theorem perpendicular_implies_parallel 
  (h1 : perpendicular_plane m α) 
  (h2 : perpendicular_plane m β) : 
  parallel_plane α β :=
sorry

-- Theorem 2
theorem perpendicular_plane_and_contained_implies_perpendicular 
  (h1 : perpendicular_plane m α) 
  (h2 : contained n α) : 
  perpendicular m n :=
sorry

end perpendicular_implies_parallel_perpendicular_plane_and_contained_implies_perpendicular_l1900_190052


namespace third_largest_three_digit_with_eight_ones_l1900_190002

/-- Given a list of digits, returns all three-digit numbers that can be formed using exactly three of those digits. -/
def threeDigitNumbers (digits : List Nat) : List Nat := sorry

/-- Checks if a number has 8 in the ones place. -/
def hasEightInOnes (n : Nat) : Bool := sorry

/-- The third largest element in a list of natural numbers. -/
def thirdLargest (numbers : List Nat) : Nat := sorry

theorem third_largest_three_digit_with_eight_ones : 
  let digits := [0, 1, 4, 8]
  let validNumbers := (threeDigitNumbers digits).filter hasEightInOnes
  thirdLargest validNumbers = 148 := by sorry

end third_largest_three_digit_with_eight_ones_l1900_190002


namespace special_quadrilateral_not_necessarily_square_l1900_190000

/-- A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perp_diagonals : Bool
  /-- A circle can be inscribed within the quadrilateral -/
  has_inscribed_circle : Bool
  /-- A circle can be circumscribed around the quadrilateral -/
  has_circumscribed_circle : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Prop :=
  -- A square has all sides equal and all angles right angles
  sorry

/-- Theorem: A quadrilateral with perpendicular diagonals, an inscribed circle, 
    and a circumscribed circle is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ q : SpecialQuadrilateral, q.perp_diagonals ∧ q.has_inscribed_circle ∧ q.has_circumscribed_circle ∧ ¬is_square q :=
by
  sorry


end special_quadrilateral_not_necessarily_square_l1900_190000


namespace fraction_equality_l1900_190003

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - b^2

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a + b - 2 * a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 8 3) / (hash_op 8 3) = -15 / 133 := by
  sorry

end fraction_equality_l1900_190003


namespace divisibility_implies_equality_l1900_190071

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end divisibility_implies_equality_l1900_190071


namespace l_shape_surface_area_l1900_190078

/-- Represents the "L" shaped solid formed by unit cubes -/
structure LShape where
  bottom_row : ℕ
  vertical_stack : ℕ
  total_cubes : ℕ

/-- Calculates the surface area of the L-shaped solid -/
def surface_area (shape : LShape) : ℕ :=
  let bottom_exposure := shape.bottom_row + (shape.bottom_row - 1)
  let vertical_stack_exposure := 4 * shape.vertical_stack + 1
  let bottom_sides := 2 + shape.bottom_row
  bottom_exposure + vertical_stack_exposure + bottom_sides

/-- Theorem stating that the surface area of the specific L-shaped solid is 26 square units -/
theorem l_shape_surface_area :
  let shape : LShape := ⟨4, 3, 7⟩
  surface_area shape = 26 := by
  sorry

end l_shape_surface_area_l1900_190078


namespace rectangle_breadth_l1900_190001

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) :
  square_area = 625 →
  rectangle_area = 100 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  rectangle_area = rectangle_length * (10 : ℝ) :=
by
  sorry

end rectangle_breadth_l1900_190001


namespace equation_solution_l1900_190035

theorem equation_solution : ∃! x : ℝ, (9 - x)^2 = x^2 ∧ x = (9 : ℝ) / 2 := by
  sorry

end equation_solution_l1900_190035


namespace room_width_calculation_l1900_190050

theorem room_width_calculation (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 10 ∧ length = 5 ∧ area = length * width → width = 2 := by
  sorry

end room_width_calculation_l1900_190050


namespace set_A_properties_l1900_190031

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem set_A_properties : 
  (1 ∈ A) ∧ (∅ ⊆ A) ∧ ({1, -1} ⊆ A) := by sorry

end set_A_properties_l1900_190031


namespace not_equal_1990_l1900_190070

/-- Count of positive integers ≤ pqn that have a common divisor with pq -/
def f (p q n : ℕ) : ℕ := 
  (n * p) + (n * q) - n

theorem not_equal_1990 (p q n : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) (hn : n > 0) :
  (f p q n : ℚ) / n ≠ 1990 := by
  sorry

end not_equal_1990_l1900_190070


namespace robert_reading_capacity_l1900_190019

/-- Represents the number of books Robert can read in a given time -/
def books_read (reading_speed : ℕ) (available_time : ℕ) (book_type1_pages : ℕ) (book_type2_pages : ℕ) : ℕ :=
  let books_type1 := available_time / (book_type1_pages / reading_speed)
  let books_type2 := available_time / (book_type2_pages / reading_speed)
  books_type1 + books_type2

/-- Theorem stating that Robert can read 5 books in 6 hours given the specified conditions -/
theorem robert_reading_capacity : 
  books_read 120 6 240 360 = 5 := by
  sorry

end robert_reading_capacity_l1900_190019


namespace cricket_average_l1900_190067

theorem cricket_average (initial_average : ℝ) : 
  (8 * initial_average + 90) / 9 = initial_average + 6 → 
  initial_average + 6 = 42 := by
  sorry

end cricket_average_l1900_190067


namespace asterisk_replacement_l1900_190081

theorem asterisk_replacement : ∃ x : ℝ, (x / 21) * (x / 84) = 1 ∧ x = 42 := by sorry

end asterisk_replacement_l1900_190081


namespace oil_in_partial_tank_l1900_190076

theorem oil_in_partial_tank (tank_capacity : ℕ) (total_oil : ℕ) : 
  tank_capacity = 32 → total_oil = 728 → 
  total_oil % tank_capacity = 24 := by sorry

end oil_in_partial_tank_l1900_190076


namespace smallest_number_is_three_l1900_190048

/-- Represents a systematic sampling of units. -/
structure SystematicSampling where
  total_units : Nat
  selected_units : Nat
  sum_of_selected : Nat

/-- Calculates the smallest number drawn in a systematic sampling. -/
def smallest_number_drawn (s : SystematicSampling) : Nat :=
  (s.sum_of_selected - (s.selected_units - 1) * s.selected_units * (s.total_units / s.selected_units) / 2) / s.selected_units

/-- Theorem stating that for the given systematic sampling, the smallest number drawn is 3. -/
theorem smallest_number_is_three :
  let s : SystematicSampling := ⟨28, 4, 54⟩
  smallest_number_drawn s = 3 := by
  sorry


end smallest_number_is_three_l1900_190048


namespace percentage_sum_l1900_190057

theorem percentage_sum : (28 / 100) * 400 + (45 / 100) * 250 = 224.5 := by
  sorry

end percentage_sum_l1900_190057


namespace perpendicular_and_tangent_l1900_190045

-- Define the given curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the given line
def l₁ (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the line we want to prove
def l₂ (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- l₂ is perpendicular to l₁
    (∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₁ x₂ y₂ → l₂ x₁ y₁ → l₂ x₂ y₂ →
      (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
      ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) *
      ((x₂ - x₁) * (3) + (y₂ - y₁) * (1)) = 0) ∧
    -- l₂ is tangent to f at (x₀, y₀)
    (l₂ x₀ y₀ ∧ f x₀ = y₀ ∧
      ∀ (x : ℝ), x ≠ x₀ → l₂ x (f x) → False) :=
sorry

end perpendicular_and_tangent_l1900_190045


namespace root_relation_iff_p_values_l1900_190017

-- Define the quadratic equation
def quadratic_equation (p : ℝ) (x : ℝ) : ℝ := x^2 + p*x + 2*p

-- Define the condition for one root being three times the other
def root_condition (p : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), 
    quadratic_equation p x₁ = 0 ∧ 
    quadratic_equation p x₂ = 0 ∧ 
    x₂ = 3 * x₁

-- Theorem statement
theorem root_relation_iff_p_values :
  ∀ p : ℝ, root_condition p ↔ (p = 0 ∨ p = 32/3) :=
by sorry

end root_relation_iff_p_values_l1900_190017


namespace figure_to_square_l1900_190065

/-- Represents a figure on a graph paper -/
structure Figure where
  area : ℕ

/-- Represents a cut of the figure -/
structure Cut where
  parts : ℕ

/-- Represents the result of reassembling cut parts -/
structure Reassembly where
  isSquare : Bool

/-- A function that cuts a figure into parts -/
def cutFigure (f : Figure) (c : Cut) : Cut :=
  c

/-- A function that reassembles cut parts -/
def reassemble (c : Cut) : Reassembly :=
  { isSquare := true }

/-- Theorem stating that a figure with area 18 can be cut into 3 parts
    and reassembled into a square -/
theorem figure_to_square (f : Figure) (h : f.area = 18) :
  ∃ (c : Cut), c.parts = 3 ∧ (reassemble (cutFigure f c)).isSquare = true := by
  sorry

end figure_to_square_l1900_190065


namespace box_depth_calculation_l1900_190097

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube -/
structure Cube where
  sideLength : ℕ

def BoxDimensions.volume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.depth

def Cube.volume (c : Cube) : ℕ :=
  c.sideLength ^ 3

/-- The theorem to be proved -/
theorem box_depth_calculation (box : BoxDimensions) (cube : Cube) :
  box.length = 30 →
  box.width = 48 →
  (80 * cube.volume = box.volume) →
  (box.length % cube.sideLength = 0) →
  (box.width % cube.sideLength = 0) →
  (box.depth % cube.sideLength = 0) →
  box.depth = 12 := by
  sorry

end box_depth_calculation_l1900_190097


namespace trivia_game_total_score_luke_total_score_l1900_190075

/-- 
Given a player in a trivia game who:
- Plays a certain number of rounds
- Scores the same number of points each round
- Scores a specific number of points per round

This theorem proves that the total points scored is equal to 
the product of the number of rounds and the points per round.
-/
theorem trivia_game_total_score 
  (rounds : ℕ) 
  (points_per_round : ℕ) : 
  rounds * points_per_round = rounds * points_per_round := by
  sorry

/-- 
This theorem applies the general trivia_game_total_score theorem 
to Luke's specific case, where he played 5 rounds and scored 60 points per round.
-/
theorem luke_total_score : 5 * 60 = 300 := by
  sorry

end trivia_game_total_score_luke_total_score_l1900_190075


namespace intersection_points_imply_c_value_l1900_190010

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem intersection_points_imply_c_value :
  ∀ c : ℝ, (∃! (a b : ℝ), a ≠ b ∧ f c a = 0 ∧ f c b = 0 ∧ 
    (∀ x : ℝ, f c x = 0 → x = a ∨ x = b)) →
  c = -2 ∨ c = 2 :=
by sorry

end intersection_points_imply_c_value_l1900_190010


namespace expression_value_l1900_190030

theorem expression_value : (85 + 32 / 113) * 113 = 9637 := by
  sorry

end expression_value_l1900_190030


namespace polynomial_divisibility_l1900_190090

-- Define the polynomial
def polynomial (p : ℝ) (x : ℝ) : ℝ := 4 * x^3 - 12 * x^2 + p * x - 16

-- Define divisibility condition
def is_divisible_by (f : ℝ → ℝ) (a : ℝ) : Prop := f a = 0

-- Theorem statement
theorem polynomial_divisibility (p : ℝ) :
  (is_divisible_by (polynomial p) 2) →
  (is_divisible_by (polynomial p) 4 ↔ p = 16) :=
by sorry

end polynomial_divisibility_l1900_190090


namespace min_value_of_function_l1900_190012

theorem min_value_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min_val : ℝ), min_val = (a^(2/3) + b^(2/3))^(3/2) ∧
  ∀ θ : ℝ, θ ∈ Set.Ioo 0 (π/2) →
    a / Real.sin θ + b / Real.cos θ ≥ min_val :=
by sorry

end min_value_of_function_l1900_190012


namespace haley_cupcakes_l1900_190036

theorem haley_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 11)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 3) :
  todd_ate + packages * cupcakes_per_package = 20 := by
  sorry

end haley_cupcakes_l1900_190036


namespace sarah_walk_probability_l1900_190061

/-- The number of gates at the airport -/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet -/
def gate_distance : ℕ := 80

/-- The maximum distance Sarah is willing to walk in feet -/
def max_walk_distance : ℕ := 320

/-- The probability that Sarah walks 320 feet or less to her new gate -/
theorem sarah_walk_probability : 
  (num_gates : ℚ) * (max_walk_distance / gate_distance * 2 : ℚ) / 
  ((num_gates : ℚ) * (num_gates - 1 : ℚ)) = 4 / 7 := by sorry

end sarah_walk_probability_l1900_190061


namespace sum_to_base3_l1900_190029

def base10_to_base3 (n : ℕ) : List ℕ :=
  sorry

theorem sum_to_base3 :
  base10_to_base3 (36 + 25 + 2) = [2, 1, 0, 0] :=
sorry

end sum_to_base3_l1900_190029


namespace folded_paper_area_ratio_l1900_190091

/-- Represents a square piece of paper -/
structure Paper where
  side : ℝ
  area : ℝ
  area_eq : area = side ^ 2

/-- Represents the folded paper -/
structure FoldedPaper where
  original : Paper
  new_area : ℝ

/-- Theorem stating the ratio of areas after folding -/
theorem folded_paper_area_ratio (p : Paper) (fp : FoldedPaper) 
  (h_fp : fp.original = p) 
  (h_fold : fp.new_area = (7 / 8) * p.area) : 
  fp.new_area / p.area = 7 / 8 := by
  sorry

end folded_paper_area_ratio_l1900_190091


namespace eleven_steps_seven_moves_l1900_190063

/-- The number of ways to climb a staircase with a given number of steps in a fixed number of moves. -/
def climbStairs (totalSteps : ℕ) (requiredMoves : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating that there are 35 ways to climb 11 steps in 7 moves -/
theorem eleven_steps_seven_moves : climbStairs 11 7 = 35 := by
  sorry

end eleven_steps_seven_moves_l1900_190063


namespace inequality_proof_l1900_190039

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  Real.exp a - 1 > a ∧ a > a ^ Real.exp 1 := by
  sorry

end inequality_proof_l1900_190039


namespace length_of_EG_l1900_190064

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the radius of the circle
def radius : ℝ := 7

-- Define the points E, F, and G on the circle
def E : Point := sorry
def F : Point := sorry
def G : Point := sorry

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define the central angle subtended by an arc
def centralAngle (p q : Point) (c : Circle) : ℝ := sorry

-- State the theorem
theorem length_of_EG (c : Circle) : 
  distance E F = 8 → 
  centralAngle E G c = π / 3 → 
  distance E G = 7 := by sorry

end length_of_EG_l1900_190064


namespace smallest_sum_l1900_190079

theorem smallest_sum (E F G H : ℕ+) : 
  (∃ d : ℤ, (E : ℤ) + d = F ∧ (F : ℤ) + d = G) →  -- arithmetic sequence
  (∃ r : ℚ, F * r = G ∧ G * r = H) →  -- geometric sequence
  G = (7 : ℚ) / 4 * F →  -- G/F = 7/4
  E + F + G + H ≥ 97 :=
sorry

end smallest_sum_l1900_190079


namespace sum_binary_digits_350_1350_l1900_190007

/-- The number of digits in the binary representation of a positive integer -/
def binaryDigits (n : ℕ+) : ℕ :=
  Nat.log2 n + 1

/-- The sum of binary digits for 350 and 1350 -/
def sumBinaryDigits : ℕ := binaryDigits 350 + binaryDigits 1350

theorem sum_binary_digits_350_1350 : sumBinaryDigits = 20 := by
  sorry

end sum_binary_digits_350_1350_l1900_190007


namespace smallest_multiple_of_9_and_21_l1900_190051

theorem smallest_multiple_of_9_and_21 :
  ∃ (b : ℕ), b > 0 ∧ 9 ∣ b ∧ 21 ∣ b ∧ ∀ (x : ℕ), x > 0 ∧ 9 ∣ x ∧ 21 ∣ x → b ≤ x :=
by sorry

end smallest_multiple_of_9_and_21_l1900_190051


namespace number_calculation_l1900_190072

theorem number_calculation (x y : ℝ) (h1 : 0.5 * x = 0.25 * y - 30) (h2 : x = 690) : y = 1500 := by
  sorry

end number_calculation_l1900_190072


namespace simplify_expression_l1900_190055

theorem simplify_expression : 15 * (7 / 10) * (1 / 9) = 7 / 6 := by
  sorry

end simplify_expression_l1900_190055


namespace floor_ceil_sum_l1900_190033

theorem floor_ceil_sum : ⌊(0.99 : ℝ)⌋ + ⌈(2.99 : ℝ)⌉ + 2 = 5 := by
  sorry

end floor_ceil_sum_l1900_190033


namespace sum_of_two_before_last_l1900_190038

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j < n → a (j + 1) - a j = a (i + 1) - a i

theorem sum_of_two_before_last (a : ℕ → ℕ) :
  arithmetic_sequence a 7 →
  a 0 = 3 →
  a 1 = 8 →
  a 2 = 13 →
  a 6 = 33 →
  a 4 + a 5 = 51 := by
  sorry

end sum_of_two_before_last_l1900_190038


namespace average_height_theorem_l1900_190044

/-- The height difference between Itzayana and Zora in inches -/
def height_diff_itzayana_zora : ℝ := 4

/-- The height difference between Brixton and Zora in inches -/
def height_diff_brixton_zora : ℝ := 8

/-- Zara's height in inches -/
def height_zara : ℝ := 64

/-- Jaxon's height in centimeters -/
def height_jaxon_cm : ℝ := 170

/-- Conversion factor from centimeters to inches -/
def cm_to_inch : ℝ := 2.54

/-- The number of people -/
def num_people : ℕ := 5

theorem average_height_theorem :
  let height_brixton : ℝ := height_zara
  let height_zora : ℝ := height_brixton - height_diff_brixton_zora
  let height_itzayana : ℝ := height_zora + height_diff_itzayana_zora
  let height_jaxon : ℝ := height_jaxon_cm / cm_to_inch
  (height_itzayana + height_zora + height_brixton + height_zara + height_jaxon) / num_people = 62.2 := by
  sorry

end average_height_theorem_l1900_190044


namespace cone_radius_from_slant_height_and_surface_area_l1900_190032

theorem cone_radius_from_slant_height_and_surface_area :
  ∀ (slant_height curved_surface_area : ℝ),
    slant_height = 22 →
    curved_surface_area = 483.80526865282815 →
    curved_surface_area = Real.pi * (7 : ℝ) * slant_height :=
by
  sorry

end cone_radius_from_slant_height_and_surface_area_l1900_190032


namespace square_properties_l1900_190096

theorem square_properties (a b : ℤ) (h : 2*a^2 + a = 3*b^2 + b) :
  ∃ (x y : ℤ), (a - b = x^2) ∧ (2*a + 2*b + 1 = y^2) := by
  sorry

end square_properties_l1900_190096
