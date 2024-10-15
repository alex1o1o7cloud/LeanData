import Mathlib

namespace NUMINAMATH_CALUDE_pants_cost_theorem_l811_81176

/-- Represents the cost and pricing strategy for a pair of pants -/
structure PantsPricing where
  cost : ℝ
  profit_percentage : ℝ
  discount_percentage : ℝ
  final_price : ℝ

/-- Calculates the selling price before discount -/
def selling_price (p : PantsPricing) : ℝ :=
  p.cost * (1 + p.profit_percentage)

/-- Calculates the final selling price after discount -/
def discounted_price (p : PantsPricing) : ℝ :=
  selling_price p * (1 - p.discount_percentage)

/-- Theorem stating the relationship between the cost and final price -/
theorem pants_cost_theorem (p : PantsPricing) 
  (h1 : p.profit_percentage = 0.30)
  (h2 : p.discount_percentage = 0.20)
  (h3 : p.final_price = 130)
  : p.cost = 125 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_theorem_l811_81176


namespace NUMINAMATH_CALUDE_like_terms_mn_value_l811_81124

/-- 
Given two algebraic terms are like terms, prove that m^n = 8.
-/
theorem like_terms_mn_value (n m : ℕ) : 
  (∃ (k : ℚ), k * X^n * Y^2 = X^3 * Y^m) → m^n = 8 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_mn_value_l811_81124


namespace NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l811_81145

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 130 → xy = 45 → x + y ≤ 10 * Real.sqrt 2.2 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l811_81145


namespace NUMINAMATH_CALUDE_jamal_book_cart_l811_81186

theorem jamal_book_cart (history_books fiction_books childrens_books wrong_place_books remaining_books : ℕ) :
  history_books = 12 →
  fiction_books = 19 →
  childrens_books = 8 →
  wrong_place_books = 4 →
  remaining_books = 16 →
  history_books + fiction_books + childrens_books + wrong_place_books + remaining_books = 59 := by
  sorry

end NUMINAMATH_CALUDE_jamal_book_cart_l811_81186


namespace NUMINAMATH_CALUDE_min_value_expression_l811_81195

theorem min_value_expression (x y : ℝ) (h : 4 - 16*x^2 - 8*x*y - y^2 > 0) :
  (13*x^2 + 24*x*y + 13*y^2 - 14*x - 16*y + 61) / (4 - 16*x^2 - 8*x*y - y^2)^(7/2) ≥ 7/16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l811_81195


namespace NUMINAMATH_CALUDE_clocks_chime_together_l811_81160

def clock1_interval : ℕ := 15
def clock2_interval : ℕ := 25

theorem clocks_chime_together : Nat.lcm clock1_interval clock2_interval = 75 := by
  sorry

end NUMINAMATH_CALUDE_clocks_chime_together_l811_81160


namespace NUMINAMATH_CALUDE_video_game_cost_l811_81167

def allowance_period1 : ℕ := 8
def allowance_rate1 : ℕ := 5
def allowance_period2 : ℕ := 6
def allowance_rate2 : ℕ := 6
def remaining_money : ℕ := 3

def total_savings : ℕ := allowance_period1 * allowance_rate1 + allowance_period2 * allowance_rate2

def money_after_clothes : ℕ := total_savings / 2

theorem video_game_cost : money_after_clothes - remaining_money = 35 := by
  sorry

end NUMINAMATH_CALUDE_video_game_cost_l811_81167


namespace NUMINAMATH_CALUDE_money_ratio_proof_l811_81165

def ram_money : ℝ := 490
def krishan_money : ℝ := 2890

/-- The ratio of money between two people -/
structure MoneyRatio where
  person1 : ℝ
  person2 : ℝ

/-- The condition that two money ratios are equal -/
def equal_ratios (r1 r2 : MoneyRatio) : Prop :=
  r1.person1 / r1.person2 = r2.person1 / r2.person2

theorem money_ratio_proof 
  (ram_gopal : MoneyRatio) 
  (gopal_krishan : MoneyRatio) 
  (h1 : ram_gopal.person1 = ram_money)
  (h2 : gopal_krishan.person2 = krishan_money)
  (h3 : equal_ratios ram_gopal gopal_krishan) :
  ∃ (n : ℕ), 
    ram_gopal.person1 / ram_gopal.person2 = 49 / 119 ∧ 
    n * ram_gopal.person1 = 49 ∧ 
    n * ram_gopal.person2 = 119 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l811_81165


namespace NUMINAMATH_CALUDE_staircase_shape_perimeter_l811_81144

/-- A shape formed by cutting out a staircase from a rectangle --/
structure StaircaseShape where
  width : ℝ
  height : ℝ
  step_size : ℝ
  num_steps : ℕ
  total_area : ℝ

/-- Calculate the perimeter of a StaircaseShape --/
def perimeter (shape : StaircaseShape) : ℝ :=
  shape.width + shape.height + shape.step_size * (2 * shape.num_steps)

/-- The main theorem --/
theorem staircase_shape_perimeter : 
  ∀ (shape : StaircaseShape), 
    shape.width = 11 ∧ 
    shape.step_size = 2 ∧ 
    shape.num_steps = 10 ∧ 
    shape.total_area = 130 →
    perimeter shape = 54.45 := by
  sorry


end NUMINAMATH_CALUDE_staircase_shape_perimeter_l811_81144


namespace NUMINAMATH_CALUDE_geometric_sequence_and_sum_l811_81185

/-- Represents the sum of the first n terms in a geometric sequence -/
def S (n : ℕ) : ℚ := sorry

/-- Represents the nth term of the geometric sequence -/
def a (n : ℕ) : ℚ := sorry

/-- Represents the nth term of the sequence b_n -/
def b (n : ℕ) : ℚ := 1 / (a n) + n

/-- Represents the sum of the first n terms of the sequence b_n -/
def T (n : ℕ) : ℚ := sorry

theorem geometric_sequence_and_sum :
  (S 3 = 7/2) → (S 6 = 63/16) →
  (∀ n, a n = (1/2)^(n-2)) ∧
  (∀ n, T n = (2^n + n^2 + n - 1) / 2) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_sum_l811_81185


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l811_81171

theorem washing_machine_capacity 
  (num_families : ℕ) 
  (people_per_family : ℕ) 
  (vacation_days : ℕ) 
  (towels_per_person_per_day : ℕ) 
  (total_loads : ℕ) 
  (h1 : num_families = 3) 
  (h2 : people_per_family = 4) 
  (h3 : vacation_days = 7) 
  (h4 : towels_per_person_per_day = 1) 
  (h5 : total_loads = 6) : 
  (num_families * people_per_family * vacation_days * towels_per_person_per_day) / total_loads = 14 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l811_81171


namespace NUMINAMATH_CALUDE_songs_on_mp3_player_l811_81129

theorem songs_on_mp3_player (initial : ℕ) (deleted : ℕ) (added : ℕ) :
  initial ≥ deleted →
  (initial - deleted + added : ℕ) = initial - deleted + added :=
by sorry

end NUMINAMATH_CALUDE_songs_on_mp3_player_l811_81129


namespace NUMINAMATH_CALUDE_frog_jump_probability_l811_81182

/-- Represents a jump as a vector in 3D space -/
structure Jump where
  x : ℝ
  y : ℝ
  z : ℝ
  magnitude_is_one : x^2 + y^2 + z^2 = 1

/-- Represents the frog's position after a series of jumps -/
def FinalPosition (jumps : List Jump) : ℝ × ℝ × ℝ :=
  let sum := jumps.foldl (fun (ax, ay, az) j => (ax + j.x, ay + j.y, az + j.z)) (0, 0, 0)
  sum

/-- The probability of the frog's final position being exactly 1 meter from the start -/
noncomputable def probability_one_meter_away (num_jumps : ℕ) : ℝ :=
  sorry

/-- Theorem stating the probability for 4 jumps is 1/8 -/
theorem frog_jump_probability :
  probability_one_meter_away 4 = 1/8 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l811_81182


namespace NUMINAMATH_CALUDE_butter_amount_is_480_l811_81110

/-- Represents the ingredients in a recipe --/
structure Ingredients where
  flour : ℝ
  butter : ℝ
  sugar : ℝ

/-- Represents the ratios of ingredients in a recipe --/
structure Ratio where
  flour : ℝ
  butter : ℝ
  sugar : ℝ

/-- Calculates the total ingredients after mixing two recipes and adding extra flour --/
def mixRecipes (cake : Ingredients) (cream : Ingredients) (extraFlour : ℝ) : Ingredients :=
  { flour := cake.flour + extraFlour
  , butter := cake.butter + cream.butter
  , sugar := cake.sugar + cream.sugar }

/-- Checks if the given ingredients satisfy the required ratio --/
def satisfiesRatio (ingredients : Ingredients) (ratio : Ratio) : Prop :=
  ingredients.flour / ratio.flour = ingredients.butter / ratio.butter ∧
  ingredients.flour / ratio.flour = ingredients.sugar / ratio.sugar

/-- Main theorem: The amount of butter used is 480 grams --/
theorem butter_amount_is_480 
  (cake_ratio : Ratio)
  (cream_ratio : Ratio)
  (cookie_ratio : Ratio)
  (cake : Ingredients)
  (cream : Ingredients)
  (h1 : satisfiesRatio cake cake_ratio)
  (h2 : satisfiesRatio cream cream_ratio)
  (h3 : cake_ratio = { flour := 3, butter := 2, sugar := 1 })
  (h4 : cream_ratio = { flour := 0, butter := 2, sugar := 3 })
  (h5 : cookie_ratio = { flour := 5, butter := 3, sugar := 2 })
  (h6 : satisfiesRatio (mixRecipes cake cream 200) cookie_ratio) :
  cake.butter + cream.butter = 480 := by
  sorry


end NUMINAMATH_CALUDE_butter_amount_is_480_l811_81110


namespace NUMINAMATH_CALUDE_periodic_odd_function_sum_l811_81123

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def MinimumPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  HasPeriod f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬HasPeriod f q

theorem periodic_odd_function_sum (f : ℝ → ℝ) :
  IsOdd f →
  MinimumPositivePeriod f 3 →
  (∀ x, f x = Real.log (1 - x)) →
  f 2010 + f 2011 = 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_sum_l811_81123


namespace NUMINAMATH_CALUDE_square_sum_pairs_l811_81194

theorem square_sum_pairs : 
  {(a, b) : ℕ × ℕ | ∃ (m n : ℕ), a^2 + 3*b = m^2 ∧ b^2 + 3*a = n^2} = 
  {(1, 1), (11, 11), (16, 11)} := by
sorry

end NUMINAMATH_CALUDE_square_sum_pairs_l811_81194


namespace NUMINAMATH_CALUDE_august_tips_multiple_l811_81187

theorem august_tips_multiple (total_months : Nat) (august_ratio : Real) 
  (h1 : total_months = 7)
  (h2 : august_ratio = 0.4) :
  let other_months := total_months - 1
  let august_tips := august_ratio * total_months
  august_tips / other_months = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_august_tips_multiple_l811_81187


namespace NUMINAMATH_CALUDE_fgh_supermarket_count_l811_81126

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 41

/-- The difference between the number of US and Canadian supermarkets -/
def difference : ℕ := 22

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - difference

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := us_supermarkets + canada_supermarkets

theorem fgh_supermarket_count : total_supermarkets = 60 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_count_l811_81126


namespace NUMINAMATH_CALUDE_remaining_calories_l811_81172

def calories_per_serving : ℕ := 110
def servings_per_block : ℕ := 16
def servings_eaten : ℕ := 5

theorem remaining_calories : 
  (servings_per_block - servings_eaten) * calories_per_serving = 1210 := by
  sorry

end NUMINAMATH_CALUDE_remaining_calories_l811_81172


namespace NUMINAMATH_CALUDE_inequality_system_solution_l811_81166

/-- Given an inequality system with solution set x < 1, find the range of a -/
theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x - 1 < 0 ∧ x < a + 3) ↔ x < 1) → a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l811_81166


namespace NUMINAMATH_CALUDE_exists_double_application_square_l811_81108

theorem exists_double_application_square : 
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 := by sorry

end NUMINAMATH_CALUDE_exists_double_application_square_l811_81108


namespace NUMINAMATH_CALUDE_willow_playing_time_l811_81177

/-- Calculates the total playing time in hours given the time spent on football and basketball in minutes -/
def total_playing_time (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Proves that given Willow played football for 60 minutes and basketball for 60 minutes, 
    the total time he played is 2 hours -/
theorem willow_playing_time :
  total_playing_time 60 60 = 2 := by sorry

end NUMINAMATH_CALUDE_willow_playing_time_l811_81177


namespace NUMINAMATH_CALUDE_vertices_form_parabola_l811_81147

/-- A parabola in the family of parabolas described by y = x^2 + 2ax + a for all real a -/
structure Parabola where
  a : ℝ

/-- The vertex of a parabola in the family -/
def vertex (p : Parabola) : ℝ × ℝ :=
  (-p.a, p.a - p.a^2)

/-- The set of all vertices of parabolas in the family -/
def vertex_set : Set (ℝ × ℝ) :=
  {v | ∃ p : Parabola, v = vertex p}

/-- The equation of the curve on which the vertices lie -/
def vertex_curve (x y : ℝ) : Prop :=
  y = -x^2 - x

theorem vertices_form_parabola :
  ∀ v ∈ vertex_set, vertex_curve v.1 v.2 := by
  sorry

#check vertices_form_parabola

end NUMINAMATH_CALUDE_vertices_form_parabola_l811_81147


namespace NUMINAMATH_CALUDE_axis_of_symmetry_implies_r_equals_s_l811_81106

/-- Represents a rational function of the form (px + q) / (rx + s) -/
structure RationalFunction (α : Type) [Field α] where
  p : α
  q : α
  r : α
  s : α
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  r_nonzero : r ≠ 0
  s_nonzero : s ≠ 0

/-- Defines the property of y = -x being an axis of symmetry for a given rational function -/
def isAxisOfSymmetry {α : Type} [Field α] (f : RationalFunction α) : Prop :=
  ∀ (x y : α), y = (f.p * x + f.q) / (f.r * x + f.s) → (-x) = (f.p * (-y) + f.q) / (f.r * (-y) + f.s)

/-- Theorem stating that if y = -x is an axis of symmetry for the rational function,
    then r - s = 0 -/
theorem axis_of_symmetry_implies_r_equals_s {α : Type} [Field α] (f : RationalFunction α) :
  isAxisOfSymmetry f → f.r = f.s :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_implies_r_equals_s_l811_81106


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l811_81162

theorem square_sum_given_product_and_sum (r s : ℝ) 
  (h1 : r * s = 16) 
  (h2 : r + s = 8) : 
  r^2 + s^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l811_81162


namespace NUMINAMATH_CALUDE_simplify_expression_1_l811_81136

theorem simplify_expression_1 (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (x^2 / (-2*y)) * (6*x*y^2 / x^4) = -3*y/x :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_1_l811_81136


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l811_81149

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≤ 1} = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l811_81149


namespace NUMINAMATH_CALUDE_inequality_proof_l811_81122

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l811_81122


namespace NUMINAMATH_CALUDE_solution_is_twelve_l811_81103

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

/-- Theorem stating that 12 is the solution to the equation -/
theorem solution_is_twelve :
  ∃ (x : ℝ), custom_mul 3 (custom_mul 6 x) = 12 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_twelve_l811_81103


namespace NUMINAMATH_CALUDE_factorial_inequality_l811_81113

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_inequality :
  factorial (factorial 100) < (factorial 99)^(factorial 100) * (factorial 100)^(factorial 99) :=
by sorry

end NUMINAMATH_CALUDE_factorial_inequality_l811_81113


namespace NUMINAMATH_CALUDE_sum_of_first_four_terms_l811_81135

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_four_terms
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_5th : a 5 = 11)
  (h_6th : a 6 = 17)
  (h_7th : a 7 = 23) :
  a 1 + a 2 + a 3 + a 4 = -16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_four_terms_l811_81135


namespace NUMINAMATH_CALUDE_right_triangle_base_length_l811_81169

theorem right_triangle_base_length 
  (height : ℝ) 
  (perimeter : ℝ) 
  (is_right_triangle : Bool) 
  (h1 : height = 3) 
  (h2 : perimeter = 12) 
  (h3 : is_right_triangle = true) : 
  ∃ (base : ℝ), base = 4 ∧ 
  ∃ (hypotenuse : ℝ), 
    perimeter = base + height + hypotenuse ∧
    hypotenuse^2 = base^2 + height^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_base_length_l811_81169


namespace NUMINAMATH_CALUDE_reflection_line_equation_l811_81161

/-- The line of reflection for a triangle --/
structure ReflectionLine where
  equation : ℝ → Prop

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points --/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- Reflects a point about a horizontal line --/
def reflect (p : Point) (y : ℝ) : Point :=
  { x := p.x, y := 2 * y - p.y }

/-- The theorem stating the equation of the reflection line --/
theorem reflection_line_equation 
  (t : Triangle)
  (t' : Triangle)
  (h1 : t.p = Point.mk 1 4)
  (h2 : t.q = Point.mk 8 9)
  (h3 : t.r = Point.mk (-3) 7)
  (h4 : t'.p = Point.mk 1 (-6))
  (h5 : t'.q = Point.mk 8 (-11))
  (h6 : t'.r = Point.mk (-3) (-9))
  (h7 : ∃ (y : ℝ), t'.p = reflect t.p y ∧ 
                   t'.q = reflect t.q y ∧ 
                   t'.r = reflect t.r y) :
  ∃ (m : ReflectionLine), m.equation = λ y => y = -1 :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l811_81161


namespace NUMINAMATH_CALUDE_ten_day_search_cost_l811_81159

/-- Tom's charging scheme for item search -/
def search_cost (days : ℕ) : ℕ :=
  let initial_rate := 100
  let discounted_rate := 60
  let initial_period := 5
  if days ≤ initial_period then
    days * initial_rate
  else
    initial_period * initial_rate + (days - initial_period) * discounted_rate

/-- The theorem stating the total cost for a 10-day search -/
theorem ten_day_search_cost : search_cost 10 = 800 := by
  sorry

end NUMINAMATH_CALUDE_ten_day_search_cost_l811_81159


namespace NUMINAMATH_CALUDE_YZ_squared_equals_33_l811_81117

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB BC CA : ℝ)
  (AB_pos : AB > 0)
  (BC_pos : BC > 0)
  (CA_pos : CA > 0)

/-- Circumcircle of a triangle -/
def Circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Incircle of a triangle -/
def Incircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Circle tangent to circumcircle and two sides of the triangle -/
def TangentCircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Intersection point of TangentCircle and Circumcircle -/
def X (t : Triangle) : ℝ × ℝ := sorry

/-- Points Y and Z on the circumcircle such that XY and YZ are tangent to the incircle -/
def Y (t : Triangle) : ℝ × ℝ := sorry
def Z (t : Triangle) : ℝ × ℝ := sorry

/-- Square of the distance between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ := sorry

theorem YZ_squared_equals_33 (t : Triangle) 
  (h1 : t.AB = 4) 
  (h2 : t.BC = 5) 
  (h3 : t.CA = 6) : 
  dist_squared (Y t) (Z t) = 33 := by sorry

end NUMINAMATH_CALUDE_YZ_squared_equals_33_l811_81117


namespace NUMINAMATH_CALUDE_system_solution_l811_81163

theorem system_solution :
  ∃ x y : ℝ, 
    (4 * x - 3 * y = -0.75) ∧
    (5 * x + 3 * y = 5.35) ∧
    (abs (x - 0.5111) < 0.0001) ∧
    (abs (y - 0.9315) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l811_81163


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l811_81112

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_iff_same_slope {a b c d : ℝ} (l1 l2 : ℝ → ℝ → Prop) :
  (∀ x y, l1 x y ↔ a * x + b * y = 0) →
  (∀ x y, l2 x y ↔ c * x + d * y = 1) →
  (∀ x y, l1 x y → l2 x y) ↔ a / b = c / d

/-- The line ax + 2y = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y = 0

/-- The line x + y = 1 -/
def line2 (x y : ℝ) : Prop := x + y = 1

/-- Theorem: a = 2 is both sufficient and necessary for line1 to be parallel to line2 -/
theorem parallel_iff_a_eq_two (a : ℝ) :
  (∀ x y, line1 a x y → line2 x y) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l811_81112


namespace NUMINAMATH_CALUDE_marcus_initial_mileage_l811_81168

/-- Represents the mileage and fuel efficiency of a car --/
structure Car where
  mpg : ℕ  -- Miles per gallon
  tankCapacity : ℕ  -- Gallons
  currentMileage : ℕ  -- Current mileage

/-- Calculates the initial mileage of a car before a road trip --/
def initialMileage (c : Car) (numFillUps : ℕ) : ℕ :=
  c.currentMileage - (c.mpg * c.tankCapacity * numFillUps)

/-- Theorem: Given the conditions of Marcus's road trip, his car's initial mileage was 1728 miles --/
theorem marcus_initial_mileage :
  let marcusCar : Car := { mpg := 30, tankCapacity := 20, currentMileage := 2928 }
  initialMileage marcusCar 2 = 1728 := by
  sorry

#eval initialMileage { mpg := 30, tankCapacity := 20, currentMileage := 2928 } 2

end NUMINAMATH_CALUDE_marcus_initial_mileage_l811_81168


namespace NUMINAMATH_CALUDE_remainder_x7_plus_2_div_x_plus_1_l811_81119

theorem remainder_x7_plus_2_div_x_plus_1 :
  ∃ q : Polynomial ℤ, (X ^ 7 + 2 : Polynomial ℤ) = (X + 1) * q + 1 :=
sorry

end NUMINAMATH_CALUDE_remainder_x7_plus_2_div_x_plus_1_l811_81119


namespace NUMINAMATH_CALUDE_triangle_side_length_l811_81193

theorem triangle_side_length (x : ℕ+) : 
  (5 + 15 > x^3 ∧ x^3 + 5 > 15 ∧ x^3 + 15 > 5) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l811_81193


namespace NUMINAMATH_CALUDE_fill_large_bottle_l811_81151

/-- The volume of shampoo in milliliters that a medium-sized bottle holds -/
def medium_bottle_volume : ℕ := 150

/-- The volume of shampoo in milliliters that a large bottle holds -/
def large_bottle_volume : ℕ := 1200

/-- The number of medium-sized bottles needed to fill a large bottle -/
def bottles_needed : ℕ := large_bottle_volume / medium_bottle_volume

theorem fill_large_bottle : bottles_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_fill_large_bottle_l811_81151


namespace NUMINAMATH_CALUDE_eel_cost_l811_81130

theorem eel_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : E = 180 := by
  sorry

end NUMINAMATH_CALUDE_eel_cost_l811_81130


namespace NUMINAMATH_CALUDE_final_balance_proof_l811_81199

def bank_transactions (initial_balance : ℚ) : ℚ :=
  let balance1 := initial_balance - 300
  let balance2 := balance1 - 150
  let balance3 := balance2 + (3/5 * balance2)
  let balance4 := balance3 - 250
  balance4 + (2/3 * balance4)

theorem final_balance_proof :
  ∃ (initial_balance : ℚ),
    (300 = (3/7) * initial_balance) ∧
    (150 = (1/3) * (initial_balance - 300)) ∧
    (250 = (1/4) * (initial_balance - 300 - 150 + (3/5) * (initial_balance - 300 - 150))) ∧
    (bank_transactions initial_balance = 1250) :=
by sorry

end NUMINAMATH_CALUDE_final_balance_proof_l811_81199


namespace NUMINAMATH_CALUDE_amanda_notebooks_l811_81180

/-- Calculate the final number of notebooks Amanda has -/
def final_notebooks (initial ordered lost : ℕ) : ℕ :=
  initial + ordered - lost

/-- Theorem stating that Amanda's final number of notebooks is 74 -/
theorem amanda_notebooks : final_notebooks 65 23 14 = 74 := by
  sorry

end NUMINAMATH_CALUDE_amanda_notebooks_l811_81180


namespace NUMINAMATH_CALUDE_continuous_n_times_iff_odd_l811_81102

/-- A function that takes every real value exactly n times. -/
def ExactlyNTimes (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ∧ (∃ (S : Finset ℝ), S.card = n ∧ ∀ x : ℝ, f x = y ↔ x ∈ S)

/-- Main theorem: A continuous function that takes every real value exactly n times exists if and only if n is odd. -/
theorem continuous_n_times_iff_odd (n : ℕ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ExactlyNTimes f n) ↔ Odd n :=
sorry


end NUMINAMATH_CALUDE_continuous_n_times_iff_odd_l811_81102


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l811_81127

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l811_81127


namespace NUMINAMATH_CALUDE_largest_good_number_all_greater_bad_smallest_bad_number_all_lesser_good_l811_81146

/-- Definition of a good number -/
def is_good (M : ℕ) : Prop :=
  ∃ a b c d : ℤ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

/-- 576 is a good number -/
theorem largest_good_number : is_good 576 := by sorry

/-- All numbers greater than 576 are bad numbers -/
theorem all_greater_bad (M : ℕ) : M > 576 → ¬ is_good M := by sorry

/-- 443 is a bad number -/
theorem smallest_bad_number : ¬ is_good 443 := by sorry

/-- All numbers less than 443 are good numbers -/
theorem all_lesser_good (M : ℕ) : M < 443 → is_good M := by sorry

end NUMINAMATH_CALUDE_largest_good_number_all_greater_bad_smallest_bad_number_all_lesser_good_l811_81146


namespace NUMINAMATH_CALUDE_production_quantity_for_36000_min_production_for_profit_8500_l811_81196

-- Define the production cost function
def C (n : ℕ) : ℝ := 4000 + 50 * n

-- Define the profit function
def P (n : ℕ) : ℝ := 40 * n - 4000

-- Theorem 1: Production quantity when cost is 36,000
theorem production_quantity_for_36000 :
  ∃ n : ℕ, C n = 36000 ∧ n = 640 := by sorry

-- Theorem 2: Minimum production for profit ≥ 8,500
theorem min_production_for_profit_8500 :
  ∃ n : ℕ, (∀ m : ℕ, P m ≥ 8500 → m ≥ n) ∧ P n ≥ 8500 ∧ n = 313 := by sorry

end NUMINAMATH_CALUDE_production_quantity_for_36000_min_production_for_profit_8500_l811_81196


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_l811_81183

theorem tangent_line_to_ln (k : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ = Real.log x₀ ∧ k = 1 / x₀) → k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_l811_81183


namespace NUMINAMATH_CALUDE_trajectory_is_circle_l811_81156

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define the points
variable (F₁ F₂ P Q : E)

-- Define the ellipse
def is_on_ellipse (P : E) (F₁ F₂ : E) (a : ℝ) : Prop :=
  dist P F₁ + dist P F₂ = 2 * a

-- Define the condition for Q
def extends_to_Q (P Q : E) (F₁ F₂ : E) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = F₁ + t • (P - F₁) ∧ dist P Q = dist P F₂

-- Theorem statement
theorem trajectory_is_circle 
  (a : ℝ) 
  (h_ellipse : is_on_ellipse P F₁ F₂ a) 
  (h_extends : extends_to_Q P Q F₁ F₂) :
  ∃ (center : E) (radius : ℝ), dist Q center = radius :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_circle_l811_81156


namespace NUMINAMATH_CALUDE_team_selection_with_girl_l811_81116

theorem team_selection_with_girl (n m k : ℕ) (hn : n = 5) (hm : m = 5) (hk : k = 3) :
  Nat.choose (n + m) k - Nat.choose n k = 110 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_with_girl_l811_81116


namespace NUMINAMATH_CALUDE_rhombus_with_60_degree_angles_l811_81141

/-- A configuration of four points in the plane -/
structure QuadConfig where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ
  A₄ : ℝ × ℝ

/-- The sum of the smallest angles in the four triangles formed by the points -/
def sumSmallestAngles (q : QuadConfig) : ℝ := sorry

/-- Predicate to check if four points form a rhombus -/
def isRhombus (q : QuadConfig) : Prop := sorry

/-- Predicate to check if all angles in a quadrilateral are at least 60° -/
def allAnglesAtLeast60 (q : QuadConfig) : Prop := sorry

/-- The main theorem -/
theorem rhombus_with_60_degree_angles 
  (q : QuadConfig) 
  (h : sumSmallestAngles q = π) : 
  isRhombus q ∧ allAnglesAtLeast60 q := by
  sorry

end NUMINAMATH_CALUDE_rhombus_with_60_degree_angles_l811_81141


namespace NUMINAMATH_CALUDE_curvilinear_triangle_area_half_triangle_area_l811_81154

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle type
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Function to calculate the area of a curvilinear triangle formed by three circles
def curvilinearTriangleArea (c1 c2 c3 : Circle) : ℝ := sorry

-- Theorem statement
theorem curvilinear_triangle_area_half_triangle_area 
  (c1 c2 c3 : Circle) 
  (t : Triangle) 
  (h1 : c1.radius = c2.radius ∧ c2.radius = c3.radius)
  (h2 : c1.center = t.a ∧ c2.center = t.b ∧ c3.center = t.c) :
  curvilinearTriangleArea c1 c2 c3 = (1/2) * triangleArea t := by sorry

end NUMINAMATH_CALUDE_curvilinear_triangle_area_half_triangle_area_l811_81154


namespace NUMINAMATH_CALUDE_ice_pack_price_is_three_l811_81197

/-- The price of a pack of 10 bags of ice for Chad's BBQ -/
def ice_pack_price (total_people : ℕ) (ice_per_person : ℕ) (bags_per_pack : ℕ) (total_spent : ℚ) : ℚ :=
  let total_ice := total_people * ice_per_person
  total_spent / (total_ice / bags_per_pack)

/-- Theorem: The price of a pack of 10 bags of ice is $3 -/
theorem ice_pack_price_is_three :
  ice_pack_price 15 2 10 9 = 3 := by
  sorry

#eval ice_pack_price 15 2 10 9

end NUMINAMATH_CALUDE_ice_pack_price_is_three_l811_81197


namespace NUMINAMATH_CALUDE_walters_age_calculation_l811_81140

/-- Walter's age at the end of 2000 -/
def walters_age_2000 : ℝ := 37.5

/-- Walter's grandmother's age at the end of 2000 -/
def grandmothers_age_2000 : ℝ := 3 * walters_age_2000

/-- The sum of Walter's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3850

/-- Walter's age at the end of 2010 -/
def walters_age_2010 : ℝ := walters_age_2000 + 10

theorem walters_age_calculation :
  (2000 - walters_age_2000) + (2000 - grandmothers_age_2000) = birth_years_sum ∧
  walters_age_2010 = 47.5 := by
  sorry

#eval walters_age_2010

end NUMINAMATH_CALUDE_walters_age_calculation_l811_81140


namespace NUMINAMATH_CALUDE_equation_solutions_l811_81100

variable (a : ℝ)

theorem equation_solutions :
  {x : ℝ | x * (x + a)^3 * (5 - x) = 0} = {0, -a, 5} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l811_81100


namespace NUMINAMATH_CALUDE_no_consecutive_red_probability_l811_81150

def num_lights : ℕ := 8
def red_prob : ℝ := 0.4
def green_prob : ℝ := 1 - red_prob

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def prob_no_consecutive_red : ℝ :=
  (green_prob ^ num_lights) * (binomial (num_lights + 1 - 0) 0) +
  (green_prob ^ 7) * red_prob * (binomial (num_lights + 1 - 1) 1) +
  (green_prob ^ 6) * (red_prob ^ 2) * (binomial (num_lights + 1 - 2) 2) +
  (green_prob ^ 5) * (red_prob ^ 3) * (binomial (num_lights + 1 - 3) 3) +
  (green_prob ^ 4) * (red_prob ^ 4) * (binomial (num_lights + 1 - 4) 4)

theorem no_consecutive_red_probability :
  prob_no_consecutive_red = 0.3499456 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_red_probability_l811_81150


namespace NUMINAMATH_CALUDE_expected_sum_of_three_marbles_l811_81152

def marbles : Finset ℕ := Finset.range 7

def draw_size : ℕ := 3

theorem expected_sum_of_three_marbles :
  let all_draws := marbles.powerset.filter (λ s => s.card = draw_size)
  let sum_of_draws := all_draws.sum (λ s => s.sum id)
  let num_of_draws := all_draws.card
  (sum_of_draws : ℚ) / num_of_draws = 12 := by sorry

end NUMINAMATH_CALUDE_expected_sum_of_three_marbles_l811_81152


namespace NUMINAMATH_CALUDE_subscription_period_l811_81138

/-- Proves that the subscription period is 18 months given the promotion conditions -/
theorem subscription_period (normal_price : ℚ) (discount_per_issue : ℚ) (total_discount : ℚ) :
  normal_price = 34 →
  discount_per_issue = 0.25 →
  total_discount = 9 →
  ∃ (period : ℕ), period * 2 * discount_per_issue = total_discount ∧ period = 18 :=
by sorry

end NUMINAMATH_CALUDE_subscription_period_l811_81138


namespace NUMINAMATH_CALUDE_collinearity_of_special_points_l811_81191

-- Define the triangle and points
variable (A B C A' B' C' A'' B'' C'' : ℝ × ℝ)

-- Define the conditions
def is_scalene_triangle (A B C : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def is_angle_bisector_point (X Y Z X' : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), X' = t • Y + (1 - t) • Z ∧ 
  (X.1 - Y.1) * (X'.1 - Z.1) = (X.2 - Y.2) * (X'.2 - Z.2)

def is_perpendicular_bisector_point (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2) = 0 ∧
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2

-- State the theorem
theorem collinearity_of_special_points 
  (h_scalene : is_scalene_triangle A B C)
  (h_A' : is_angle_bisector_point A B C A')
  (h_B' : is_angle_bisector_point B C A B')
  (h_C' : is_angle_bisector_point C A B C')
  (h_A'' : is_perpendicular_bisector_point A A' A'')
  (h_B'' : is_perpendicular_bisector_point B B' B'')
  (h_C'' : is_perpendicular_bisector_point C C' C'') :
  ∃ (m b : ℝ), A''.2 = m * A''.1 + b ∧ 
               B''.2 = m * B''.1 + b ∧ 
               C''.2 = m * C''.1 + b :=
sorry

end NUMINAMATH_CALUDE_collinearity_of_special_points_l811_81191


namespace NUMINAMATH_CALUDE_no_integer_solution_l811_81137

theorem no_integer_solution :
  ¬ ∃ (A B C : ℤ),
    (A - B = 1620) ∧
    ((75 : ℚ) / 1000 * A = (125 : ℚ) / 1000 * B) ∧
    (A + B = (1 : ℚ) / 2 * C^4) ∧
    (A^2 + B^2 = C^2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l811_81137


namespace NUMINAMATH_CALUDE_total_wheels_is_142_l811_81153

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of wheels on a unicycle -/
def unicycle_wheels : ℕ := 1

/-- The number of wheels on a four-wheeler -/
def four_wheeler_wheels : ℕ := 4

/-- The number of bicycles in Storage Area A -/
def bicycles_A : ℕ := 16

/-- The number of tricycles in Storage Area A -/
def tricycles_A : ℕ := 7

/-- The number of unicycles in Storage Area A -/
def unicycles_A : ℕ := 10

/-- The number of four-wheelers in Storage Area A -/
def four_wheelers_A : ℕ := 5

/-- The number of bicycles in Storage Area B -/
def bicycles_B : ℕ := 12

/-- The number of tricycles in Storage Area B -/
def tricycles_B : ℕ := 5

/-- The number of unicycles in Storage Area B -/
def unicycles_B : ℕ := 8

/-- The number of four-wheelers in Storage Area B -/
def four_wheelers_B : ℕ := 3

/-- The total number of wheels in both storage areas -/
def total_wheels : ℕ := 
  (bicycles_A * bicycle_wheels + tricycles_A * tricycle_wheels + unicycles_A * unicycle_wheels + four_wheelers_A * four_wheeler_wheels) +
  (bicycles_B * bicycle_wheels + tricycles_B * tricycle_wheels + unicycles_B * unicycle_wheels + four_wheelers_B * four_wheeler_wheels)

theorem total_wheels_is_142 : total_wheels = 142 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_142_l811_81153


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l811_81104

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 6)^2 + (y - 7)^2 + (z - 8)^2 = 0 → x + y + z = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l811_81104


namespace NUMINAMATH_CALUDE_probability_of_y_selection_l811_81125

theorem probability_of_y_selection (p_x p_both : ℝ) (h1 : p_x = 1/7) 
  (h2 : p_both = 0.031746031746031744) : 
  ∃ p_y : ℝ, p_y = 0.2222222222222222 ∧ p_both = p_x * p_y :=
sorry

end NUMINAMATH_CALUDE_probability_of_y_selection_l811_81125


namespace NUMINAMATH_CALUDE_largest_constant_for_good_array_l811_81128

def isGoodArray (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a < b ∧
  Nat.lcm a b + Nat.lcm (a + 2) (b + 2) = 2 * Nat.lcm (a + 1) (b + 1)

theorem largest_constant_for_good_array :
  (∃ c : ℚ, c > 0 ∧
    (∀ a b : ℕ, isGoodArray a b → b > c * a^3) ∧
    (∀ ε > 0, ∃ a b : ℕ, isGoodArray a b ∧ b ≤ (c + ε) * a^3)) ∧
  (let c := (1/2 : ℚ); 
   (∀ a b : ℕ, isGoodArray a b → b > c * a^3) ∧
   (∀ ε > 0, ∃ a b : ℕ, isGoodArray a b ∧ b ≤ (c + ε) * a^3)) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_for_good_array_l811_81128


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l811_81181

/-- Given an arithmetic sequence {a_n} where S_n is the sum of the first n terms,
    if (S_2016 / 2016) - (S_2015 / 2015) = 3, then a_2016 - a_2014 = 12. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →
  (S 2016 / 2016 - S 2015 / 2015 = 3) →
  a 2016 - a 2014 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l811_81181


namespace NUMINAMATH_CALUDE_sisters_age_difference_l811_81158

/-- The age difference between two sisters, Denise and Diane -/
def ageDifference (deniseFutureAge deniseFutureYears dianeFutureAge dianeFutureYears : ℕ) : ℕ :=
  (deniseFutureAge - deniseFutureYears) - (dianeFutureAge - dianeFutureYears)

/-- Theorem stating that the age difference between Denise and Diane is 4 years -/
theorem sisters_age_difference :
  ageDifference 25 2 25 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sisters_age_difference_l811_81158


namespace NUMINAMATH_CALUDE_right_triangle_9_40_41_l811_81118

theorem right_triangle_9_40_41 : 
  ∀ (a b c : ℝ), a = 9 ∧ b = 40 ∧ c = 41 → a^2 + b^2 = c^2 :=
by
  sorry

#check right_triangle_9_40_41

end NUMINAMATH_CALUDE_right_triangle_9_40_41_l811_81118


namespace NUMINAMATH_CALUDE_hannah_adblock_efficiency_l811_81120

/-- The percentage of ads not blocked by Hannah's AdBlock -/
def ads_not_blocked : ℝ := sorry

/-- The percentage of not blocked ads that are interesting -/
def interesting_not_blocked_ratio : ℝ := 0.20

/-- The percentage of all ads that are not interesting and not blocked -/
def not_interesting_not_blocked_ratio : ℝ := 0.16

theorem hannah_adblock_efficiency :
  ads_not_blocked = 0.20 :=
sorry

end NUMINAMATH_CALUDE_hannah_adblock_efficiency_l811_81120


namespace NUMINAMATH_CALUDE_rider_distance_l811_81107

/-- The distance traveled by a rider moving back and forth along a moving caravan -/
theorem rider_distance (caravan_length caravan_distance : ℝ) 
  (h_length : caravan_length = 1)
  (h_distance : caravan_distance = 1) : 
  ∃ (rider_speed : ℝ), 
    rider_speed > 0 ∧ 
    (1 / (rider_speed - 1) + 1 / (rider_speed + 1) = 1) ∧
    rider_speed * caravan_distance = 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_rider_distance_l811_81107


namespace NUMINAMATH_CALUDE_cycle_original_price_l811_81109

/-- Given a cycle sold for Rs. 1080 with a gain of 8%, prove that the original price was Rs. 1000 -/
theorem cycle_original_price (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1080)
  (h2 : gain_percentage = 8) :
  let original_price := selling_price / (1 + gain_percentage / 100)
  original_price = 1000 := by sorry

end NUMINAMATH_CALUDE_cycle_original_price_l811_81109


namespace NUMINAMATH_CALUDE_smallest_r_value_l811_81175

theorem smallest_r_value (p q r : ℕ) : 
  0 < p ∧ p < q ∧ q < r ∧                   -- p, q, r are positive integers and p < q < r
  (2 * q = p + r) ∧                         -- arithmetic progression
  (r * r = p * q) →                         -- geometric progression
  r ≥ 5 ∧ ∃ (p' q' r' : ℕ), 
    0 < p' ∧ p' < q' ∧ q' < r' ∧ 
    (2 * q' = p' + r') ∧ 
    (r' * r' = p' * q') ∧ 
    r' = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_r_value_l811_81175


namespace NUMINAMATH_CALUDE_basketball_free_throws_l811_81134

theorem basketball_free_throws 
  (two_point_shots three_point_shots free_throws : ℕ) :
  (3 * three_point_shots = 2 * two_point_shots) →
  (three_point_shots = two_point_shots - 2) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 68) →
  free_throws = 44 := by
sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l811_81134


namespace NUMINAMATH_CALUDE_soda_price_ratio_l811_81131

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio (v : ℝ) (p : ℝ) (h1 : v > 0) (h2 : p > 0) : 
  (0.85 * p) / (1.25 * v) / (p / v) = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l811_81131


namespace NUMINAMATH_CALUDE_probability_two_female_contestants_l811_81198

theorem probability_two_female_contestants (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 8 →
  female = 5 →
  male = 3 →
  (female.choose 2 : ℚ) / (total.choose 2) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_female_contestants_l811_81198


namespace NUMINAMATH_CALUDE_complement_union_theorem_l811_81178

open Set

-- Define the sets
def U : Set ℝ := univ
def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_union_theorem : 
  (U \ (A ∪ B)) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l811_81178


namespace NUMINAMATH_CALUDE_candy_mixture_total_candy_mixture_total_proof_l811_81190

/-- Proves that the total amount of candy needed is 80 pounds given the problem conditions --/
theorem candy_mixture_total (price_cheap price_expensive price_mixture : ℚ) 
                            (amount_cheap : ℚ) (total_amount : ℚ) : Prop :=
  price_cheap = 2 ∧ 
  price_expensive = 3 ∧ 
  price_mixture = 2.2 ∧
  amount_cheap = 64 ∧
  total_amount = 80 ∧
  ∃ (amount_expensive : ℚ),
    amount_cheap + amount_expensive = total_amount ∧
    (amount_cheap * price_cheap + amount_expensive * price_expensive) / total_amount = price_mixture
    
theorem candy_mixture_total_proof : 
  candy_mixture_total 2 3 2.2 64 80 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_total_candy_mixture_total_proof_l811_81190


namespace NUMINAMATH_CALUDE_population_reaches_limit_l811_81157

/-- The number of years it takes for the population to reach or exceed the sustainable limit -/
def years_to_sustainable_limit : ℕ :=
  -- We'll define this later in the proof
  210

/-- The amount of acres required per person for sustainable living -/
def acres_per_person : ℕ := 2

/-- The total available acres for human habitation -/
def total_acres : ℕ := 35000

/-- The initial population in 2005 -/
def initial_population : ℕ := 150

/-- The number of years it takes for the population to double -/
def years_to_double : ℕ := 30

/-- The maximum sustainable population -/
def max_sustainable_population : ℕ := total_acres / acres_per_person

/-- The population after a given number of years -/
def population_after_years (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / years_to_double))

/-- Theorem stating that the population reaches or exceeds the sustainable limit in the specified number of years -/
theorem population_reaches_limit :
  population_after_years years_to_sustainable_limit ≥ max_sustainable_population ∧
  population_after_years (years_to_sustainable_limit - years_to_double) < max_sustainable_population :=
by sorry

end NUMINAMATH_CALUDE_population_reaches_limit_l811_81157


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l811_81133

theorem absolute_value_sum_difference (a b : ℝ) 
  (ha : |a| = 4) (hb : |b| = 3) : 
  ((a * b < 0 → |a + b| = 1) ∧ (a * b > 0 → |a - b| = 1)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l811_81133


namespace NUMINAMATH_CALUDE_correct_regression_l811_81142

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if two variables are positively correlated -/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Calculates the sample mean of a variable -/
def sample_mean (x : ℝ → ℝ) : ℝ := sorry

/-- Checks if a linear regression equation is valid for given data -/
def is_valid_regression (reg : LinearRegression) (x y : ℝ → ℝ) : Prop :=
  positively_correlated x y ∧
  sample_mean x = 3 ∧
  sample_mean y = 3.5 ∧
  reg.slope > 0 ∧
  reg.slope * (sample_mean x) + reg.intercept = sample_mean y

theorem correct_regression :
  is_valid_regression ⟨0.4, 2.3⟩ (λ _ => sorry) (λ _ => sorry) := by sorry

end NUMINAMATH_CALUDE_correct_regression_l811_81142


namespace NUMINAMATH_CALUDE_employee_satisfaction_theorem_l811_81164

-- Define the total number of people surveyed
def total_people : ℕ := 200

-- Define the number of people satisfied with both
def satisfied_both : ℕ := 50

-- Define the number of people satisfied with employee dedication
def satisfied_dedication : ℕ := (40 * total_people) / 100

-- Define the number of people satisfied with management level
def satisfied_management : ℕ := (45 * total_people) / 100

-- Define the chi-square function
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value
def critical_value : ℚ := 6635 / 1000

-- Define the probability of being satisfied with both
def prob_both : ℚ := satisfied_both / total_people

-- Theorem statement
theorem employee_satisfaction_theorem :
  let a := satisfied_both
  let b := satisfied_dedication - satisfied_both
  let c := satisfied_management - satisfied_both
  let d := total_people - satisfied_dedication - satisfied_management + satisfied_both
  (chi_square a b c d > critical_value) ∧
  (3 * prob_both * (1 - prob_both)^2 + 2 * 3 * prob_both^2 * (1 - prob_both) + 3 * prob_both^3 = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_employee_satisfaction_theorem_l811_81164


namespace NUMINAMATH_CALUDE_roots_of_g_are_cubes_of_roots_of_f_l811_81115

/-- The original polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

/-- The polynomial g(x) whose roots are the cubes of the roots of f(x) -/
def g (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

/-- Theorem stating that the roots of g are the cubes of the roots of f -/
theorem roots_of_g_are_cubes_of_roots_of_f :
  ∀ r : ℝ, f r = 0 → g (r^3) = 0 := by sorry

end NUMINAMATH_CALUDE_roots_of_g_are_cubes_of_roots_of_f_l811_81115


namespace NUMINAMATH_CALUDE_goods_train_length_l811_81121

/-- The length of a goods train passing a man on another train --/
theorem goods_train_length
  (man_train_speed : ℝ)
  (goods_train_speed : ℝ)
  (passing_time : ℝ)
  (h1 : man_train_speed = 40)
  (h2 : goods_train_speed = 72)
  (h3 : passing_time = 9) :
  (man_train_speed + goods_train_speed) * passing_time * 1000 / 3600 = 280 := by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l811_81121


namespace NUMINAMATH_CALUDE_poster_purchase_l811_81139

theorem poster_purchase (regular_price : ℕ) (budget : ℕ) : 
  budget = 24 * regular_price → 
  (∃ (num_posters : ℕ), 
    num_posters * regular_price + (num_posters / 2) * (regular_price / 2) = budget ∧ 
    num_posters = 32) :=
by sorry

end NUMINAMATH_CALUDE_poster_purchase_l811_81139


namespace NUMINAMATH_CALUDE_machine_selling_price_l811_81143

/-- Calculates the selling price of a machine given its costs and profit percentage -/
def calculate_selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 28500 Rs -/
theorem machine_selling_price :
  calculate_selling_price 13000 5000 1000 50 = 28500 := by
  sorry

#eval calculate_selling_price 13000 5000 1000 50

end NUMINAMATH_CALUDE_machine_selling_price_l811_81143


namespace NUMINAMATH_CALUDE_parabola_point_relation_l811_81155

theorem parabola_point_relation (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = x₁^2 - 4*x₁ + 3 →
  y₂ = x₂^2 - 4*x₂ + 3 →
  x₁ > x₂ →
  x₂ > 2 →
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_relation_l811_81155


namespace NUMINAMATH_CALUDE_two_out_of_three_probability_l811_81174

-- Define the success rate of the basketball player
def success_rate : ℚ := 3 / 5

-- Define the number of shots taken
def total_shots : ℕ := 3

-- Define the number of successful shots we're interested in
def successful_shots : ℕ := 2

-- Theorem statement
theorem two_out_of_three_probability :
  (Nat.choose total_shots successful_shots : ℚ) * success_rate ^ successful_shots * (1 - success_rate) ^ (total_shots - successful_shots) = 54 / 125 := by
  sorry

end NUMINAMATH_CALUDE_two_out_of_three_probability_l811_81174


namespace NUMINAMATH_CALUDE_binary_101011_equals_43_l811_81101

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101011_equals_43 :
  binary_to_decimal [true, true, false, true, false, true] = 43 := by
  sorry

end NUMINAMATH_CALUDE_binary_101011_equals_43_l811_81101


namespace NUMINAMATH_CALUDE_slope_problem_l811_81189

theorem slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = 2 * m) : m = (3 + Real.sqrt 41) / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_problem_l811_81189


namespace NUMINAMATH_CALUDE_volunteer_selection_l811_81170

theorem volunteer_selection (boys girls : ℕ) (positions : ℕ) : 
  boys = 4 → girls = 3 → positions = 4 → 
  (Nat.choose (boys + girls) positions) - (Nat.choose boys positions) = 34 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_selection_l811_81170


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l811_81114

theorem fraction_sum_equality : (20 : ℚ) / 50 - 3 / 8 + 1 / 4 = 11 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l811_81114


namespace NUMINAMATH_CALUDE_cos_sin_pi_12_product_l811_81184

theorem cos_sin_pi_12_product (π : Real) : 
  (Real.cos (π / 12) - Real.sin (π / 12)) * (Real.cos (π / 12) + Real.sin (π / 12)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_pi_12_product_l811_81184


namespace NUMINAMATH_CALUDE_remainder_problem_l811_81148

theorem remainder_problem (x : ℤ) : x % 63 = 27 → x % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l811_81148


namespace NUMINAMATH_CALUDE_remaining_crops_l811_81173

/-- Calculates the total number of remaining crops for a farmer after pest damage --/
theorem remaining_crops (corn_per_row potato_per_row wheat_per_row : ℕ)
  (corn_destroyed potato_destroyed wheat_destroyed : ℚ)
  (corn_rows potato_rows wheat_rows : ℕ)
  (h_corn : corn_per_row = 12)
  (h_potato : potato_per_row = 40)
  (h_wheat : wheat_per_row = 60)
  (h_corn_dest : corn_destroyed = 30 / 100)
  (h_potato_dest : potato_destroyed = 40 / 100)
  (h_wheat_dest : wheat_destroyed = 25 / 100)
  (h_corn_rows : corn_rows = 25)
  (h_potato_rows : potato_rows = 15)
  (h_wheat_rows : wheat_rows = 20) :
  (corn_rows * corn_per_row * (1 - corn_destroyed)).floor +
  (potato_rows * potato_per_row * (1 - potato_destroyed)).floor +
  (wheat_rows * wheat_per_row * (1 - wheat_destroyed)).floor = 1470 := by
  sorry

end NUMINAMATH_CALUDE_remaining_crops_l811_81173


namespace NUMINAMATH_CALUDE_cups_sold_calculation_l811_81188

def lemon_cost : ℕ := 10
def sugar_cost : ℕ := 5
def cup_cost : ℕ := 3
def price_per_cup : ℕ := 4
def profit : ℕ := 66

theorem cups_sold_calculation : 
  ∃ (cups : ℕ), 
    cups * price_per_cup = (lemon_cost + sugar_cost + cup_cost + profit) ∧ 
    cups = 21 := by
  sorry

end NUMINAMATH_CALUDE_cups_sold_calculation_l811_81188


namespace NUMINAMATH_CALUDE_value_of_z_l811_81132

theorem value_of_z (x y z : ℝ) : 
  y = 3 * x - 5 → 
  z = 3 * x + 3 → 
  y = 1 → 
  z = 9 := by
sorry

end NUMINAMATH_CALUDE_value_of_z_l811_81132


namespace NUMINAMATH_CALUDE_C_is_largest_l811_81105

-- Define A, B, and C
def A : ℚ := 2010/2009 + 2010/2011
def B : ℚ := (2010/2011) * (2012/2011)
def C : ℚ := 2011/2010 + 2011/2012 + 1/10000

-- Theorem statement
theorem C_is_largest : C > A ∧ C > B := by
  sorry

end NUMINAMATH_CALUDE_C_is_largest_l811_81105


namespace NUMINAMATH_CALUDE_survey_result_l811_81192

/-- The percentage of parents who agree to the tuition fee increase -/
def agree_percentage : ℝ := 0.20

/-- The number of parents who disagree with the tuition fee increase -/
def disagree_count : ℕ := 640

/-- The total number of parents surveyed -/
def total_parents : ℕ := 800

/-- Theorem stating that the total number of parents surveyed is 800 -/
theorem survey_result : total_parents = 800 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l811_81192


namespace NUMINAMATH_CALUDE_print_325_pages_time_l811_81111

/-- Calculates the time required to print a given number of pages with a printer that has a specific print rate and delay after every 100 pages. -/
def print_time (total_pages : ℕ) (pages_per_minute : ℕ) (delay_minutes : ℕ) : ℕ :=
  let print_time := total_pages / pages_per_minute
  let num_delays := total_pages / 100
  print_time + num_delays * delay_minutes

/-- Theorem stating that printing 325 pages takes 16 minutes with the given conditions. -/
theorem print_325_pages_time :
  print_time 325 25 1 = 16 :=
by sorry

end NUMINAMATH_CALUDE_print_325_pages_time_l811_81111


namespace NUMINAMATH_CALUDE_set_A_at_most_one_element_iff_a_in_range_l811_81179

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + a = 0}

-- Theorem statement
theorem set_A_at_most_one_element_iff_a_in_range :
  ∀ a : ℝ, (∃ (x y : ℝ), x ∈ A a ∧ y ∈ A a ∧ x ≠ y) ↔ a ∈ {a : ℝ | a < -1 ∨ (-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)} :=
by sorry

end NUMINAMATH_CALUDE_set_A_at_most_one_element_iff_a_in_range_l811_81179
