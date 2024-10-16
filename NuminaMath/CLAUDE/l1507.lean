import Mathlib

namespace NUMINAMATH_CALUDE_teddy_material_in_tons_l1507_150766

/-- The amount of fluffy foam material Teddy uses for each pillow in pounds -/
def material_per_pillow : ℝ := 5 - 3

/-- The number of pillows Teddy can make -/
def number_of_pillows : ℕ := 3000

/-- The number of pounds in a ton -/
def pounds_per_ton : ℝ := 2000

/-- The theorem stating the amount of fluffy foam material Teddy has in tons -/
theorem teddy_material_in_tons : 
  (material_per_pillow * number_of_pillows) / pounds_per_ton = 3 := by
  sorry

end NUMINAMATH_CALUDE_teddy_material_in_tons_l1507_150766


namespace NUMINAMATH_CALUDE_total_photos_taken_is_46_l1507_150730

/-- Represents the number of photos on Toby's camera roll at different stages --/
structure PhotoCount where
  initial : Nat
  deletedBadShots : Nat
  catPictures : Nat
  deletedAfterEditing : Nat
  additionalShots : Nat
  secondSession : Nat
  thirdSession : Nat
  deletedFromSecond : Nat
  deletedFromThird : Nat
  final : Nat

/-- Calculates the total number of photos taken in all photo shoots --/
def totalPhotosTaken (p : PhotoCount) : Nat :=
  let firstSessionPhotos := p.final - p.initial + p.deletedBadShots - p.catPictures + 
                            p.deletedAfterEditing - p.additionalShots - 
                            (p.secondSession - p.deletedFromSecond) - 
                            (p.thirdSession - p.deletedFromThird)
  firstSessionPhotos + p.secondSession + p.thirdSession

/-- Theorem stating that the total number of photos taken in all photo shoots is 46 --/
theorem total_photos_taken_is_46 (p : PhotoCount) 
  (h1 : p.initial = 63)
  (h2 : p.deletedBadShots = 7)
  (h3 : p.catPictures = 15)
  (h4 : p.deletedAfterEditing = 3)
  (h5 : p.additionalShots = 5)
  (h6 : p.secondSession = 11)
  (h7 : p.thirdSession = 8)
  (h8 : p.deletedFromSecond = 6)
  (h9 : p.deletedFromThird = 4)
  (h10 : p.final = 112) :
  totalPhotosTaken p = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_photos_taken_is_46_l1507_150730


namespace NUMINAMATH_CALUDE_point_2_4_is_D_l1507_150716

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the diagram points
def F : Point2D := ⟨5, 5⟩
def D : Point2D := ⟨2, 4⟩

-- Theorem statement
theorem point_2_4_is_D : 
  ∃ (p : Point2D), p.x = 2 ∧ p.y = 4 ∧ p = D :=
sorry

end NUMINAMATH_CALUDE_point_2_4_is_D_l1507_150716


namespace NUMINAMATH_CALUDE_impossible_all_positive_4x4_impossible_all_positive_8x8_l1507_150709

/-- Represents a grid of signs -/
def Grid (n : Nat) := Fin n → Fin n → Bool

/-- Represents a line (row, column, or diagonal) in the grid -/
inductive Line (n : Nat)
| Row : Fin n → Line n
| Col : Fin n → Line n
| Diag : Bool → Line n

/-- Flips the signs along a given line in the grid -/
def flipLine (n : Nat) (g : Grid n) (l : Line n) : Grid n :=
  sorry

/-- Checks if all signs in the grid are positive -/
def allPositive (n : Nat) (g : Grid n) : Prop :=
  ∀ i j, g i j = true

/-- Initial configuration for the 8x8 grid with one negative sign -/
def initialConfig : Grid 8 :=
  sorry

/-- Theorem for the 4x4 grid -/
theorem impossible_all_positive_4x4 (g : Grid 4) :
  ¬∃ (flips : List (Line 4)), allPositive 4 (flips.foldl (flipLine 4) g) :=
  sorry

/-- Theorem for the 8x8 grid -/
theorem impossible_all_positive_8x8 :
  ¬∃ (flips : List (Line 8)), allPositive 8 (flips.foldl (flipLine 8) initialConfig) :=
  sorry

end NUMINAMATH_CALUDE_impossible_all_positive_4x4_impossible_all_positive_8x8_l1507_150709


namespace NUMINAMATH_CALUDE_remainder_theorem_l1507_150707

theorem remainder_theorem (y : ℤ) 
  (h1 : (2 + y) % (3^3) = 3^2 % (3^3))
  (h2 : (4 + y) % (5^3) = 2^3 % (5^3))
  (h3 : (6 + y) % (7^3) = 7^2 % (7^3)) :
  y % 105 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1507_150707


namespace NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l1507_150746

theorem square_sum_from_sum_and_product (x y : ℚ) 
  (h1 : x + y = 5/6) (h2 : x * y = 7/36) : x^2 + y^2 = 11/36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l1507_150746


namespace NUMINAMATH_CALUDE_part_one_part_two_l1507_150733

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := abs x * (x + a)

-- Part I
theorem part_one (a : ℝ) (h : ∀ x, f a x = -f a (-x)) : a = 0 := by
  sorry

-- Part II
theorem part_two (b : ℝ) (h1 : b > 0) 
  (h2 : ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-b) b, f 0 x ≤ max ∧ min ≤ f 0 x) ∧ max - min = b) :
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1507_150733


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1507_150705

theorem basketball_lineup_combinations : 
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let lineup_size : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  let captain_in_lineup : ℕ := 1

  (Nat.choose quadruplets quadruplets_in_lineup) * 
  (Nat.choose (total_players - quadruplets - captain_in_lineup) 
              (lineup_size - quadruplets_in_lineup - captain_in_lineup)) = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1507_150705


namespace NUMINAMATH_CALUDE_cube_sum_product_l1507_150741

theorem cube_sum_product : (3^3 * 4^3) + (3^3 * 2^3) = 1944 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_product_l1507_150741


namespace NUMINAMATH_CALUDE_relay_team_count_l1507_150714

/-- The number of sprinters --/
def total_sprinters : ℕ := 6

/-- The number of sprinters to be selected --/
def selected_sprinters : ℕ := 4

/-- The number of ways to form the relay team --/
def relay_team_formations : ℕ := 252

/-- Theorem stating the number of ways to form the relay team --/
theorem relay_team_count :
  (total_sprinters = 6) →
  (selected_sprinters = 4) →
  (∃ A B : ℕ, A ≠ B ∧ A ≤ total_sprinters ∧ B ≤ total_sprinters) →
  relay_team_formations = 252 :=
by sorry

end NUMINAMATH_CALUDE_relay_team_count_l1507_150714


namespace NUMINAMATH_CALUDE_delta_airlines_discount_percentage_l1507_150780

theorem delta_airlines_discount_percentage 
  (delta_price : ℝ) 
  (united_price : ℝ) 
  (united_discount : ℝ) 
  (price_difference : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  united_discount = 0.3 →
  price_difference = 90 →
  let united_discounted_price := united_price * (1 - united_discount)
  let delta_discounted_price := united_discounted_price - price_difference
  let delta_discount_amount := delta_price - delta_discounted_price
  let delta_discount_percentage := delta_discount_amount / delta_price
  delta_discount_percentage = 0.2 := by sorry

end NUMINAMATH_CALUDE_delta_airlines_discount_percentage_l1507_150780


namespace NUMINAMATH_CALUDE_solution_set_for_a_l1507_150706

/-- The solution set for parameter a in the given equation with domain restrictions -/
theorem solution_set_for_a (x a : ℝ) : 
  x ≠ 2 → x ≠ 6 → a - 7*x + 39 ≥ 0 →
  (x^2 - 4*x - 21 + ((|x-2|)/(x-2) + (|x-6|)/(x-6) + a)^2 = 0) →
  a ∈ Set.Ioo (-5) (-4) ∪ Set.Ioo (-3) 3 ∪ Set.Ico 5 7 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_l1507_150706


namespace NUMINAMATH_CALUDE_expansion_terms_count_l1507_150786

def expandedTerms (N : ℕ) : ℕ := Nat.choose N 4

theorem expansion_terms_count : expandedTerms 14 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l1507_150786


namespace NUMINAMATH_CALUDE_pythagorean_triple_for_eleven_l1507_150757

theorem pythagorean_triple_for_eleven : ∃ b c : ℕ, 11^2 + b^2 = c^2 ∧ c = 61 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_for_eleven_l1507_150757


namespace NUMINAMATH_CALUDE_shirt_discount_price_l1507_150794

theorem shirt_discount_price (original_price discount_percentage : ℝ) 
  (h1 : original_price = 80)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 68 := by
  sorry

end NUMINAMATH_CALUDE_shirt_discount_price_l1507_150794


namespace NUMINAMATH_CALUDE_relay_arrangements_count_l1507_150763

/-- Represents the number of people in the class -/
def class_size : ℕ := 5

/-- Represents the number of people needed for the relay -/
def relay_size : ℕ := 4

/-- Represents the number of options for the first runner -/
def first_runner_options : ℕ := 3

/-- Represents the number of options for the last runner -/
def last_runner_options : ℕ := 2

/-- Calculates the number of relay arrangements given the constraints -/
def relay_arrangements : ℕ := 24

/-- Theorem stating that the number of relay arrangements is 24 -/
theorem relay_arrangements_count : 
  relay_arrangements = 24 := by sorry

end NUMINAMATH_CALUDE_relay_arrangements_count_l1507_150763


namespace NUMINAMATH_CALUDE_vectors_parallel_when_m_neg_one_l1507_150790

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Vector a parameterized by m -/
def a (m : ℝ) : ℝ × ℝ := (2*m - 1, m)

/-- Vector b -/
def b : ℝ × ℝ := (3, 1)

/-- Theorem stating that vectors a and b are parallel when m = -1 -/
theorem vectors_parallel_when_m_neg_one :
  are_parallel (a (-1)) b := by sorry

end NUMINAMATH_CALUDE_vectors_parallel_when_m_neg_one_l1507_150790


namespace NUMINAMATH_CALUDE_stormi_car_wash_l1507_150736

/-- Proves the number of cars Stormi washed to save for a bicycle --/
theorem stormi_car_wash : 
  ∀ (car_wash_price lawn_mow_income bicycle_price additional_needed : ℕ),
  car_wash_price = 10 →
  lawn_mow_income = 26 →
  bicycle_price = 80 →
  additional_needed = 24 →
  (bicycle_price - additional_needed - lawn_mow_income) / car_wash_price = 3 := by
sorry

end NUMINAMATH_CALUDE_stormi_car_wash_l1507_150736


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_seven_l1507_150742

theorem same_solution_implies_a_equals_seven (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = 3 ∧ 3 - (a - x) / 3 = 1) →
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_seven_l1507_150742


namespace NUMINAMATH_CALUDE_connie_blue_markers_l1507_150781

/-- Given that Connie has 2315 red markers and 3343 markers in total, 
    prove that she has 1028 blue markers. -/
theorem connie_blue_markers 
  (total_markers : ℕ) 
  (red_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : red_markers = 2315) :
  total_markers - red_markers = 1028 := by
  sorry

end NUMINAMATH_CALUDE_connie_blue_markers_l1507_150781


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1507_150765

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 1)
  f (-1) = 1 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1507_150765


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1507_150727

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1507_150727


namespace NUMINAMATH_CALUDE_product_of_two_fifteens_l1507_150789

theorem product_of_two_fifteens : ∀ (a b : ℕ), a = 15 → b = 15 → a * b = 225 := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_fifteens_l1507_150789


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1507_150737

/-- The weight of the new person when the average weight of a group increases -/
def new_person_weight (num_people : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + num_people * avg_increase

/-- Theorem stating the weight of the new person under given conditions -/
theorem weight_of_new_person :
  new_person_weight 12 4 65 = 113 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1507_150737


namespace NUMINAMATH_CALUDE_city_partition_theorem_l1507_150751

/-- A directed graph where each vertex has outdegree 2 -/
structure CityGraph (V : Type) :=
  (edges : V → V → Prop)
  (outdegree_two : ∀ v : V, ∃ u w : V, u ≠ w ∧ edges v u ∧ edges v w ∧ ∀ x : V, edges v x → (x = u ∨ x = w))

/-- A partition of the vertices into 1014 sets -/
def ValidPartition (V : Type) (G : CityGraph V) :=
  ∃ (f : V → Fin 1014),
    (∀ v w : V, G.edges v w → f v ≠ f w) ∧
    (∀ i j : Fin 1014, i ≠ j →
      (∀ v w : V, f v = i ∧ f w = j → G.edges v w) ∨
      (∀ v w : V, f v = i ∧ f w = j → G.edges w v))

/-- The main theorem: every CityGraph has a ValidPartition -/
theorem city_partition_theorem (V : Type) (G : CityGraph V) :
  ValidPartition V G :=
sorry

end NUMINAMATH_CALUDE_city_partition_theorem_l1507_150751


namespace NUMINAMATH_CALUDE_sphere_in_cylinder_ratio_l1507_150728

theorem sphere_in_cylinder_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (π * r^2 * h = 3 * (4/3 * π * r^3)) → (h / (2 * r) = 2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cylinder_ratio_l1507_150728


namespace NUMINAMATH_CALUDE_linda_current_age_l1507_150770

/-- Represents the ages of Sarah, Jake, and Linda -/
structure Ages where
  sarah : ℚ
  jake : ℚ
  linda : ℚ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 11
  (ages.sarah + ages.jake + ages.linda) / 3 = 11 ∧
  -- Five years ago, Linda was the same age as Sarah is now
  ages.linda - 5 = ages.sarah ∧
  -- In 4 years, Jake's age will be 3/4 of Sarah's age at that time
  ages.jake + 4 = 3 / 4 * (ages.sarah + 4)

/-- The theorem stating Linda's current age -/
theorem linda_current_age (ages : Ages) (h : age_conditions ages) : 
  ages.linda = 14 := by
  sorry

end NUMINAMATH_CALUDE_linda_current_age_l1507_150770


namespace NUMINAMATH_CALUDE_museum_ticket_price_museum_ticket_price_is_6_l1507_150778

theorem museum_ticket_price (friday_price : ℝ) (saturday_visitors : ℕ) 
  (saturday_visitor_ratio : ℝ) (saturday_revenue_ratio : ℝ) : ℝ :=
let friday_visitors : ℕ := saturday_visitors / 2
let friday_revenue : ℝ := friday_visitors * friday_price
let saturday_revenue : ℝ := friday_revenue * saturday_revenue_ratio
let k : ℝ := saturday_revenue / saturday_visitors
k

theorem museum_ticket_price_is_6 :
  museum_ticket_price 9 200 2 (4/3) = 6 := by
sorry

end NUMINAMATH_CALUDE_museum_ticket_price_museum_ticket_price_is_6_l1507_150778


namespace NUMINAMATH_CALUDE_angle_complement_supplement_l1507_150731

theorem angle_complement_supplement (x : ℝ) : 
  (90 - x) = 3 * (180 - x) → (180 - x = 135 ∧ 90 - x = 45) := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_l1507_150731


namespace NUMINAMATH_CALUDE_base_equation_solution_l1507_150791

/-- Represents a digit in base b --/
def Digit (b : ℕ) := Fin b

/-- Converts a natural number to its representation in base b --/
def toBase (n : ℕ) (b : ℕ) : List (Digit b) :=
  sorry

/-- Adds two numbers in base b --/
def addBase (x y : List (Digit b)) : List (Digit b) :=
  sorry

/-- Checks if a list of digits is equal to another list of digits --/
def digitListEq (x y : List (Digit b)) : Prop :=
  sorry

theorem base_equation_solution :
  ∀ b : ℕ, b > 1 →
    (digitListEq (addBase (toBase 295 b) (toBase 467 b)) (toBase 762 b)) ↔ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_equation_solution_l1507_150791


namespace NUMINAMATH_CALUDE_fruit_shopping_cost_l1507_150717

/-- Calculates the price per unit of fruit given the number of fruits and their total price in cents. -/
def price_per_unit (num_fruits : ℕ) (total_price : ℕ) : ℚ :=
  total_price / num_fruits

/-- Determines the cheaper fruit given their prices per unit. -/
def cheaper_fruit (apple_price : ℚ) (orange_price : ℚ) : ℚ :=
  min apple_price orange_price

theorem fruit_shopping_cost :
  let apple_price := price_per_unit 10 200  -- 10 apples for $2 (200 cents)
  let orange_price := price_per_unit 5 150  -- 5 oranges for $1.50 (150 cents)
  let cheaper_price := cheaper_fruit apple_price orange_price
  (12 : ℕ) * (cheaper_price : ℚ) = 240
  := by sorry

end NUMINAMATH_CALUDE_fruit_shopping_cost_l1507_150717


namespace NUMINAMATH_CALUDE_square_sum_implies_sum_l1507_150754

theorem square_sum_implies_sum (x : ℝ) (h : x > 0) :
  Real.sqrt x + (Real.sqrt x)⁻¹ = 3 → x + x⁻¹ = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_sum_l1507_150754


namespace NUMINAMATH_CALUDE_max_x_plus_z_l1507_150725

theorem max_x_plus_z (x y z t : ℝ) 
  (h1 : x^2 + y^2 = 4)
  (h2 : z^2 + t^2 = 9)
  (h3 : x*t + y*z = 6) :
  x + z ≤ Real.sqrt 13 ∧ ∃ x y z t, x^2 + y^2 = 4 ∧ z^2 + t^2 = 9 ∧ x*t + y*z = 6 ∧ x + z = Real.sqrt 13 := by
  sorry

#check max_x_plus_z

end NUMINAMATH_CALUDE_max_x_plus_z_l1507_150725


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1507_150722

-- Define the rectangle dimensions
def rectangle_length : ℕ := 9
def rectangle_width : ℕ := 4

-- Define the number of parts
def num_parts : ℕ := 3

-- Define the square side length
def square_side : ℕ := 6

-- Theorem statement
theorem rectangle_to_square :
  ∃ (part1 part2 part3 : ℕ × ℕ),
    -- The parts fit within the original rectangle
    part1.1 ≤ rectangle_length ∧ part1.2 ≤ rectangle_width ∧
    part2.1 ≤ rectangle_length ∧ part2.2 ≤ rectangle_width ∧
    part3.1 ≤ rectangle_length ∧ part3.2 ≤ rectangle_width ∧
    -- The total area of the parts equals the area of the original rectangle
    part1.1 * part1.2 + part2.1 * part2.2 + part3.1 * part3.2 = rectangle_length * rectangle_width ∧
    -- The parts can form a square
    (part1.1 = square_side ∨ part1.2 = square_side) ∧
    (part2.1 + part3.1 = square_side ∨ part2.2 + part3.2 = square_side) :=
by sorry

#check rectangle_to_square

end NUMINAMATH_CALUDE_rectangle_to_square_l1507_150722


namespace NUMINAMATH_CALUDE_books_sold_l1507_150793

/-- Given Paul's initial and final number of books, prove that he sold 42 books. -/
theorem books_sold (initial_books final_books : ℕ) 
  (h1 : initial_books = 108) 
  (h2 : final_books = 66) : 
  initial_books - final_books = 42 := by
  sorry

#check books_sold

end NUMINAMATH_CALUDE_books_sold_l1507_150793


namespace NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_for_x_geq_one_l1507_150788

theorem x_squared_geq_one_necessary_not_sufficient_for_x_geq_one :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) := by
sorry

end NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_for_x_geq_one_l1507_150788


namespace NUMINAMATH_CALUDE_pizzas_bought_l1507_150747

def total_slices : ℕ := 32
def slices_left : ℕ := 7
def slices_per_pizza : ℕ := 8

theorem pizzas_bought : (total_slices - slices_left) / slices_per_pizza = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_bought_l1507_150747


namespace NUMINAMATH_CALUDE_wendy_album_pics_l1507_150721

def pictures_per_album (phone_pics camera_pics num_albums : ℕ) : ℕ :=
  (phone_pics + camera_pics) / num_albums

theorem wendy_album_pics : pictures_per_album 22 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_wendy_album_pics_l1507_150721


namespace NUMINAMATH_CALUDE_vector_collinearity_l1507_150739

theorem vector_collinearity (k : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.sqrt 3, 1]
  let b : Fin 2 → ℝ := ![0, -1]
  let c : Fin 2 → ℝ := ![k, Real.sqrt 3]
  (∃ (t : ℝ), a + 2 • b = t • c) → k = -3 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1507_150739


namespace NUMINAMATH_CALUDE_night_rides_total_l1507_150704

def total_ferris_rides : ℕ := 13
def total_roller_coaster_rides : ℕ := 9
def day_ferris_rides : ℕ := 7
def day_roller_coaster_rides : ℕ := 4

theorem night_rides_total : 
  (total_ferris_rides - day_ferris_rides) + (total_roller_coaster_rides - day_roller_coaster_rides) = 11 := by
  sorry

end NUMINAMATH_CALUDE_night_rides_total_l1507_150704


namespace NUMINAMATH_CALUDE_book_pricing_loss_percentage_l1507_150798

theorem book_pricing_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h1 : cost_price > 0) 
  (h2 : 5 * cost_price = 20 * selling_price) : 
  (cost_price - selling_price) / cost_price = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_book_pricing_loss_percentage_l1507_150798


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1507_150726

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ (x y : ℝ), y^2 = 12*x ∧ y = 3*x + c

/-- If the line y = 3x + d is tangent to the parabola y^2 = 12x, then d = 1 -/
theorem tangent_line_to_parabola (d : ℝ) : 
  (∃ (x y : ℝ), y^2 = 12*x ∧ y = 3*x + d) → d = 1 := by
  sorry

#check tangent_line_to_parabola

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1507_150726


namespace NUMINAMATH_CALUDE_no_articles_in_general_context_l1507_150756

/-- Represents the possible article choices for a noun in a sentence -/
inductive Article
  | Definite   -- represents "the"
  | Indefinite -- represents "a" or "an"
  | None       -- represents no article

/-- Represents the context of a sentence -/
inductive Context
  | General
  | Specific

/-- Represents a noun in the sentence -/
inductive Noun
  | College
  | Prison

/-- Determines the correct article for a noun given the context -/
def correctArticle (context : Context) (noun : Noun) : Article :=
  match context, noun with
  | Context.General, _ => Article.None
  | Context.Specific, _ => Article.Definite

/-- The main theorem stating that in a general context, 
    both "college" and "prison" should have no article -/
theorem no_articles_in_general_context : 
  ∀ (context : Context),
    context = Context.General →
    correctArticle context Noun.College = Article.None ∧
    correctArticle context Noun.Prison = Article.None :=
by sorry

end NUMINAMATH_CALUDE_no_articles_in_general_context_l1507_150756


namespace NUMINAMATH_CALUDE_problem_statement_l1507_150782

theorem problem_statement : 2 * Real.sin (π / 3) + (-1/2)⁻¹ + |2 - Real.sqrt 3| = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1507_150782


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l1507_150761

theorem stratified_sampling_survey (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (sample_female : ℕ) (sample_size : ℕ) : 
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  sample_female = 80 → 
  sample_size * (female_students / (teachers + male_students + female_students)) = sample_female → 
  sample_size = 192 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l1507_150761


namespace NUMINAMATH_CALUDE_wine_card_probability_l1507_150783

theorem wine_card_probability : 
  let n_card_types : ℕ := 3
  let n_bottles : ℕ := 5
  let total_outcomes : ℕ := n_card_types^n_bottles
  let two_type_outcomes : ℕ := Nat.choose n_card_types 2 * 2^n_bottles
  let one_type_outcomes : ℕ := n_card_types
  let favorable_outcomes : ℕ := total_outcomes - (two_type_outcomes - one_type_outcomes)
  (favorable_outcomes : ℚ) / total_outcomes = 50 / 81 :=
by sorry

end NUMINAMATH_CALUDE_wine_card_probability_l1507_150783


namespace NUMINAMATH_CALUDE_expand_expression_l1507_150784

theorem expand_expression (x : ℝ) : (3 * x + 5) * (4 * x - 2) = 12 * x^2 + 14 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1507_150784


namespace NUMINAMATH_CALUDE_floor_power_divisibility_l1507_150720

theorem floor_power_divisibility (n : ℕ) : 
  (2^(n+1) : ℤ) ∣ ⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_power_divisibility_l1507_150720


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1507_150743

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
def prob_less_than (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normal random variable ξ with mean 40, 
    if P(ξ < 30) = 0.2, then P(30 < ξ < 50) = 0.6 -/
theorem normal_distribution_probability 
  (ξ : NormalRandomVariable) 
  (h_mean : ξ.μ = 40) 
  (h_prob : prob_less_than ξ 30 = 0.2) : 
  prob_between ξ 30 50 = 0.6 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1507_150743


namespace NUMINAMATH_CALUDE_sin_10_50_70_eq_one_eighth_l1507_150792

/-- The value of sin 10° * sin 50° * sin 70° is equal to 1/8 -/
theorem sin_10_50_70_eq_one_eighth :
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_10_50_70_eq_one_eighth_l1507_150792


namespace NUMINAMATH_CALUDE_expression_value_l1507_150785

theorem expression_value : 
  let x : ℝ := 26
  let y : ℝ := 3 * x / 2
  let z : ℝ := 11
  (x - (y - z)) - ((x - y) - z) = 22 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1507_150785


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1507_150760

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : B ∩ (U \ A) = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1507_150760


namespace NUMINAMATH_CALUDE_max_intersection_points_l1507_150775

theorem max_intersection_points (x_points y_points : ℕ) : x_points = 15 → y_points = 6 → 
  (x_points.choose 2) * (y_points.choose 2) = 1575 := by sorry

end NUMINAMATH_CALUDE_max_intersection_points_l1507_150775


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l1507_150753

/-- Represents the points on the circle -/
inductive Point
| one | two | three | four | five | six | seven

/-- Calculates the next point based on the jumping rules -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.four
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.seven
  | Point.five => Point.one
  | Point.six => Point.two
  | Point.seven => Point.three

/-- Calculates the point after n jumps -/
def jumpNTimes (start : Point) (n : ℕ) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpNTimes start n)

theorem bug_position_after_2023_jumps :
  jumpNTimes Point.seven 2023 = Point.one :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l1507_150753


namespace NUMINAMATH_CALUDE_digit_difference_when_reversed_l1507_150755

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a 3-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a 3-digit number -/
def ThreeDigitNumber.reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

theorem digit_difference_when_reversed (n : ThreeDigitNumber) 
  (h : (n.reversed_value - n.value : ℚ) / 10 = 19.8) : 
  n.hundreds - n.units = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_when_reversed_l1507_150755


namespace NUMINAMATH_CALUDE_reciprocal_of_recurring_decimal_l1507_150764

/-- The decimal representation of the recurring decimal 0.363636... -/
def recurring_decimal : ℚ := 36 / 99

/-- The reciprocal of the common fraction form of 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_recurring_decimal : 
  (recurring_decimal)⁻¹ = reciprocal := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_recurring_decimal_l1507_150764


namespace NUMINAMATH_CALUDE_third_fourth_product_l1507_150762

def arithmetic_sequence (a : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem third_fourth_product (a : ℝ) (d : ℝ) :
  arithmetic_sequence a d 5 = 17 ∧ d = 2 →
  (arithmetic_sequence a d 2) * (arithmetic_sequence a d 3) = 143 := by
sorry

end NUMINAMATH_CALUDE_third_fourth_product_l1507_150762


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l1507_150724

theorem tan_sum_reciprocal (a b : ℝ) 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 2)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 44/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l1507_150724


namespace NUMINAMATH_CALUDE_digit_sum_l1507_150777

theorem digit_sum (P Q R S T : Nat) : 
  (P < 10 ∧ Q < 10 ∧ R < 10 ∧ S < 10 ∧ T < 10) → 
  (4 * (P * 10000 + Q * 1000 + R * 100 + S * 10 + T) = 41024) → 
  (P + Q + R + S + T = 14) := by
sorry

end NUMINAMATH_CALUDE_digit_sum_l1507_150777


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1507_150708

theorem min_value_quadratic (x y : ℝ) :
  y = 3 * x^2 + 6 * x + 9 →
  ∀ z : ℝ, y ≥ 6 ∧ ∃ w : ℝ, 3 * w^2 + 6 * w + 9 = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1507_150708


namespace NUMINAMATH_CALUDE_ballCombinations_2005_l1507_150769

/-- The number of ways to choose n balls from red, green, and yellow balls
    such that the number of red balls is even or the number of green balls is odd. -/
def ballCombinations (n : ℕ) : ℕ := sorry

/-- The main theorem stating the number of combinations for 2005 balls. -/
theorem ballCombinations_2005 : ballCombinations 2005 = Nat.choose 2007 2 - Nat.choose 1004 2 := by
  sorry

end NUMINAMATH_CALUDE_ballCombinations_2005_l1507_150769


namespace NUMINAMATH_CALUDE_distance_45N_90long_diff_l1507_150796

/-- The spherical distance between two points on Earth --/
def spherical_distance (R : ℝ) (lat1 lat2 long1 long2 : ℝ) : ℝ := sorry

/-- Theorem: The spherical distance between two points at 45°N with 90° longitude difference --/
theorem distance_45N_90long_diff (R : ℝ) :
  spherical_distance R (π/4) (π/4) (π/9) (11*π/18) = π*R/3 := by sorry

end NUMINAMATH_CALUDE_distance_45N_90long_diff_l1507_150796


namespace NUMINAMATH_CALUDE_medical_staff_composition_l1507_150729

theorem medical_staff_composition :
  ∀ (a b c d : ℕ),
    a + b + c + d = 17 →
    a + b ≥ c + d →
    d > a →
    a > b →
    c ≥ 2 →
    a = 5 ∧ b = 4 ∧ c = 2 ∧ d = 6 :=
by sorry

end NUMINAMATH_CALUDE_medical_staff_composition_l1507_150729


namespace NUMINAMATH_CALUDE_max_mineral_worth_l1507_150776

-- Define the mineral types
inductive Mineral
| Sapphire
| Ruby
| Emerald

-- Define the properties of each mineral
def weight (m : Mineral) : Nat :=
  match m with
  | Mineral.Sapphire => 6
  | Mineral.Ruby => 3
  | Mineral.Emerald => 2

def value (m : Mineral) : Nat :=
  match m with
  | Mineral.Sapphire => 18
  | Mineral.Ruby => 9
  | Mineral.Emerald => 4

-- Define the maximum carrying capacity
def maxWeight : Nat := 20

-- Define the minimum available quantity of each mineral
def minQuantity : Nat := 30

-- Define a function to calculate the total weight of a combination of minerals
def totalWeight (s r e : Nat) : Nat :=
  s * weight Mineral.Sapphire + r * weight Mineral.Ruby + e * weight Mineral.Emerald

-- Define a function to calculate the total value of a combination of minerals
def totalValue (s r e : Nat) : Nat :=
  s * value Mineral.Sapphire + r * value Mineral.Ruby + e * value Mineral.Emerald

-- Theorem: The maximum worth of minerals Joe can carry is $58
theorem max_mineral_worth :
  ∃ s r e : Nat,
    s ≤ minQuantity ∧ r ≤ minQuantity ∧ e ≤ minQuantity ∧
    totalWeight s r e ≤ maxWeight ∧
    totalValue s r e = 58 ∧
    ∀ s' r' e' : Nat,
      s' ≤ minQuantity → r' ≤ minQuantity → e' ≤ minQuantity →
      totalWeight s' r' e' ≤ maxWeight →
      totalValue s' r' e' ≤ 58 :=
by sorry

end NUMINAMATH_CALUDE_max_mineral_worth_l1507_150776


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1507_150713

/-- 
Given an arithmetic sequence of consecutive integers where:
- k is a natural number
- The first term is k^2 - 1
- The number of terms is 2k - 1

The sum of all terms in this sequence is equal to 2k^3 + k^2 - 5k + 2
-/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let first_term := k^2 - 1
  let num_terms := 2*k - 1
  let last_term := first_term + (num_terms - 1)
  (num_terms : ℝ) * (first_term + last_term) / 2 = 2*k^3 + k^2 - 5*k + 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1507_150713


namespace NUMINAMATH_CALUDE_squirrel_acorns_l1507_150718

theorem squirrel_acorns (total_acorns : ℕ) (num_months : ℕ) (acorns_per_month : ℕ) :
  total_acorns = 210 →
  num_months = 3 →
  acorns_per_month = 60 →
  total_acorns - num_months * acorns_per_month = 30 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l1507_150718


namespace NUMINAMATH_CALUDE_area_of_triangle_GAB_l1507_150702

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define points P, Q, and G
def point_P : ℝ × ℝ := (2, 0)
def point_Q : ℝ × ℝ := (0, -2)
def point_G : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem area_of_triangle_GAB :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧
    curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    line_l point_Q.1 point_Q.2 →
    let area := (1/2) * ‖A - B‖ * (2 * Real.sqrt 2)
    area = 16 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_area_of_triangle_GAB_l1507_150702


namespace NUMINAMATH_CALUDE_jordan_empty_boxes_l1507_150711

/-- A structure representing the distribution of items in boxes -/
structure BoxDistribution where
  total : ℕ
  pencils : ℕ
  pens : ℕ
  markers : ℕ
  pencils_and_pens : ℕ
  pencils_and_markers : ℕ
  pens_and_markers : ℕ

/-- The number of boxes with no items, given a box distribution -/
def empty_boxes (d : BoxDistribution) : ℕ :=
  d.total - (d.pencils + d.pens + d.markers - d.pencils_and_pens - d.pencils_and_markers - d.pens_and_markers)

/-- The specific box distribution from the problem -/
def jordan_boxes : BoxDistribution :=
  { total := 15
  , pencils := 8
  , pens := 5
  , markers := 3
  , pencils_and_pens := 2
  , pencils_and_markers := 1
  , pens_and_markers := 1 }

/-- Theorem stating that the number of empty boxes in Jordan's distribution is 3 -/
theorem jordan_empty_boxes :
    empty_boxes jordan_boxes = 3 := by
  sorry


end NUMINAMATH_CALUDE_jordan_empty_boxes_l1507_150711


namespace NUMINAMATH_CALUDE_f_difference_l1507_150774

/-- The function f(x) = x^4 + x^2 + 5x^3 -/
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x^3

/-- Theorem: f(5) - f(-5) = 1250 -/
theorem f_difference : f 5 - f (-5) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l1507_150774


namespace NUMINAMATH_CALUDE_students_taking_physics_or_chemistry_but_not_both_l1507_150715

theorem students_taking_physics_or_chemistry_but_not_both 
  (both : ℕ) 
  (physics : ℕ) 
  (only_chemistry : ℕ) 
  (h1 : both = 12) 
  (h2 : physics = 22) 
  (h3 : only_chemistry = 9) : 
  (physics - both) + only_chemistry = 19 := by
sorry

end NUMINAMATH_CALUDE_students_taking_physics_or_chemistry_but_not_both_l1507_150715


namespace NUMINAMATH_CALUDE_probability_is_one_third_l1507_150740

/-- The set of digits used to form the number -/
def digits : Finset Nat := {2, 4, 6, 7}

/-- A function to check if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function to check if a number is not a multiple of 3 -/
def notMultipleOf3 (n : Nat) : Bool := n % 3 ≠ 0

/-- The set of all four-digit numbers that can be formed using the given digits -/
def allNumbers : Finset Nat := sorry

/-- The set of favorable numbers (odd with hundreds digit not multiple of 3) -/
def favorableNumbers : Finset Nat := sorry

/-- The probability of forming a favorable number -/
def probability : Rat := (Finset.card favorableNumbers : Rat) / (Finset.card allNumbers : Rat)

theorem probability_is_one_third :
  probability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l1507_150740


namespace NUMINAMATH_CALUDE_fence_length_15m_l1507_150749

/-- The length of a fence surrounding a square swimming pool -/
def fence_length (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The length of a fence surrounding a square swimming pool with side length 15 meters is 60 meters -/
theorem fence_length_15m : fence_length 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fence_length_15m_l1507_150749


namespace NUMINAMATH_CALUDE_equal_ratios_sum_ratio_l1507_150759

theorem equal_ratios_sum_ratio (x y z : ℚ) : 
  x / 2 = y / 3 ∧ y / 3 = z / 4 → (x + y + z) / (2 * z) = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_sum_ratio_l1507_150759


namespace NUMINAMATH_CALUDE_walk_legs_count_l1507_150748

/-- The number of legs of a human -/
def human_legs : ℕ := 2

/-- The number of legs of a dog -/
def dog_legs : ℕ := 4

/-- The number of humans on the walk -/
def num_humans : ℕ := 2

/-- The number of dogs on the walk -/
def num_dogs : ℕ := 2

/-- The total number of legs of all organisms on the walk -/
def total_legs : ℕ := human_legs * num_humans + dog_legs * num_dogs

theorem walk_legs_count : total_legs = 12 := by
  sorry

end NUMINAMATH_CALUDE_walk_legs_count_l1507_150748


namespace NUMINAMATH_CALUDE_study_time_for_desired_average_l1507_150752

/-- Represents the relationship between study time and test score -/
structure StudyRelationship where
  time : ℝ
  score : ℝ
  k : ℝ
  inverse_prop : score * time = k

/-- Represents two tests with their study times and scores -/
structure TwoTests where
  test1 : StudyRelationship
  test2 : StudyRelationship
  avg_score : ℝ
  avg_constraint : (test1.score + test2.score) / 2 = avg_score

/-- The main theorem to prove -/
theorem study_time_for_desired_average (tests : TwoTests) :
  tests.test1.time = 6 ∧
  tests.test1.score = 80 ∧
  tests.avg_score = 85 →
  tests.test2.time = 16 / 3 :=
by sorry

end NUMINAMATH_CALUDE_study_time_for_desired_average_l1507_150752


namespace NUMINAMATH_CALUDE_squirrels_in_tree_l1507_150738

/-- Given a tree with nuts and squirrels, where the number of squirrels is 2 more than the number of nuts,
    prove that if there are 2 nuts, then there are 4 squirrels. -/
theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) 
  (h1 : nuts = 2) 
  (h2 : squirrels = nuts + 2) : 
  squirrels = 4 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_in_tree_l1507_150738


namespace NUMINAMATH_CALUDE_birds_on_fence_l1507_150745

theorem birds_on_fence (initial_birds : ℝ) (birds_flown_away : ℝ) :
  initial_birds = 12.0 →
  birds_flown_away = 8.0 →
  initial_birds - birds_flown_away = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1507_150745


namespace NUMINAMATH_CALUDE_min_investment_amount_l1507_150767

/-- Represents an investment plan with two interest rates -/
structure InvestmentPlan where
  amount_at_7_percent : ℝ
  amount_at_12_percent : ℝ

/-- Calculates the total interest earned from an investment plan -/
def total_interest (plan : InvestmentPlan) : ℝ :=
  0.07 * plan.amount_at_7_percent + 0.12 * plan.amount_at_12_percent

/-- Calculates the total investment amount -/
def total_investment (plan : InvestmentPlan) : ℝ :=
  plan.amount_at_7_percent + plan.amount_at_12_percent

/-- Theorem: The minimum total investment amount is $25,000 -/
theorem min_investment_amount :
  ∀ (plan : InvestmentPlan),
    plan.amount_at_7_percent ≤ 11000 →
    total_interest plan ≥ 2450 →
    total_investment plan ≥ 25000 :=
by sorry

end NUMINAMATH_CALUDE_min_investment_amount_l1507_150767


namespace NUMINAMATH_CALUDE_remainder_97_pow_50_mod_100_l1507_150701

theorem remainder_97_pow_50_mod_100 : 97^50 % 100 = 49 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_pow_50_mod_100_l1507_150701


namespace NUMINAMATH_CALUDE_money_duration_l1507_150779

def lawn_money : ℕ := 9
def weed_eating_money : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_duration : 
  (lawn_money + weed_eating_money) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_duration_l1507_150779


namespace NUMINAMATH_CALUDE_isabel_weekly_run_distance_l1507_150710

/-- Calculates the total distance run in a week given a circuit length, 
    morning runs, afternoon runs, and number of days. -/
def total_distance_run (circuit_length : ℕ) (morning_runs : ℕ) (afternoon_runs : ℕ) (days : ℕ) : ℕ :=
  (circuit_length * (morning_runs + afternoon_runs) * days)

/-- Proves that running a 365-meter circuit 7 times in the morning and 3 times 
    in the afternoon for 7 days results in a total distance of 25550 meters. -/
theorem isabel_weekly_run_distance :
  total_distance_run 365 7 3 7 = 25550 := by
  sorry

#eval total_distance_run 365 7 3 7

end NUMINAMATH_CALUDE_isabel_weekly_run_distance_l1507_150710


namespace NUMINAMATH_CALUDE_positive_y_floor_product_l1507_150771

theorem positive_y_floor_product (y : ℝ) : 
  y > 0 → y * ⌊y⌋ = 90 → y = 10 := by sorry

end NUMINAMATH_CALUDE_positive_y_floor_product_l1507_150771


namespace NUMINAMATH_CALUDE_boyden_family_children_l1507_150750

theorem boyden_family_children (adult_ticket_cost child_ticket_cost total_cost : ℕ) 
  (num_adults : ℕ) (h1 : adult_ticket_cost = child_ticket_cost + 6)
  (h2 : total_cost = 77) (h3 : adult_ticket_cost = 19) (h4 : num_adults = 2) :
  ∃ (num_children : ℕ), 
    num_children * child_ticket_cost + num_adults * adult_ticket_cost = total_cost ∧ 
    num_children = 3 := by
  sorry

end NUMINAMATH_CALUDE_boyden_family_children_l1507_150750


namespace NUMINAMATH_CALUDE_find_M_l1507_150703

theorem find_M (x y z M : ℚ) 
  (sum_eq : x + y + z = 120)
  (x_dec : x - 10 = M)
  (y_inc : y + 10 = M)
  (z_mul : 10 * z = M) :
  M = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1507_150703


namespace NUMINAMATH_CALUDE_eggs_equal_rice_cost_l1507_150758

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℝ := 0.36

/-- The cost of an egg in dollars -/
def egg_cost : ℝ := rice_cost

/-- The cost of half a liter of kerosene in dollars -/
def kerosene_cost : ℝ := 8 * egg_cost

/-- The number of eggs that cost the same as a pound of rice -/
def eggs_per_rice : ℕ := 1

theorem eggs_equal_rice_cost : eggs_per_rice = 1 := by
  sorry

end NUMINAMATH_CALUDE_eggs_equal_rice_cost_l1507_150758


namespace NUMINAMATH_CALUDE_independence_test_conclusions_not_always_correct_l1507_150772

-- Define the concept of independence tests
def IndependenceTest : Type := Unit

-- Define the properties of independence tests
axiom small_probability_principle : IndependenceTest → Prop
axiom conclusions_vary_with_samples : IndependenceTest → Prop
axiom not_only_method : IndependenceTest → Prop

-- Define the statement we want to prove false
def conclusions_always_correct (test : IndependenceTest) : Prop :=
  ∀ (sample : Type), true

-- Theorem statement
theorem independence_test_conclusions_not_always_correct :
  ∃ (test : IndependenceTest),
    small_probability_principle test ∧
    conclusions_vary_with_samples test ∧
    not_only_method test ∧
    ¬(conclusions_always_correct test) :=
by
  sorry

end NUMINAMATH_CALUDE_independence_test_conclusions_not_always_correct_l1507_150772


namespace NUMINAMATH_CALUDE_combined_annual_income_l1507_150735

-- Define the monthly incomes as real numbers
variable (A_income B_income C_income D_income : ℝ)

-- Define the conditions
def income_ratio : Prop :=
  A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ D_income / C_income = 4 / 3

def B_income_relation : Prop :=
  B_income = 1.12 * C_income

def D_income_relation : Prop :=
  D_income = 0.85 * A_income

def C_income_value : Prop :=
  C_income = 15000

-- Define the theorem
theorem combined_annual_income
  (h1 : income_ratio A_income B_income C_income D_income)
  (h2 : B_income_relation B_income C_income)
  (h3 : D_income_relation A_income D_income)
  (h4 : C_income_value C_income) :
  (A_income + B_income + C_income + D_income) * 12 = 936600 :=
by sorry

end NUMINAMATH_CALUDE_combined_annual_income_l1507_150735


namespace NUMINAMATH_CALUDE_complex_product_l1507_150700

theorem complex_product : Complex.I * Complex.I = -1 → (1 + 2 * Complex.I) * (2 - Complex.I) = 4 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_l1507_150700


namespace NUMINAMATH_CALUDE_subtraction_equality_l1507_150773

theorem subtraction_equality : 3.65 - 2.27 - 0.48 = 0.90 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_equality_l1507_150773


namespace NUMINAMATH_CALUDE_department_store_sales_fraction_l1507_150712

theorem department_store_sales_fraction (Q P : ℝ) (h1 : Q > 0) (h2 : P > -100) : 
  let december_sales := Q * (1 + (1 + P/100) + (1 + P/100)^2)
  let total_sales := 3 * 11 + Q + Q * (1 + P/100) + Q * (1 + P/100)^2
  let fraction := december_sales / total_sales
  fraction = (Q * (1 + (1 + P/100) + (1 + P/100)^2)) / (3 * 11 + Q + Q * (1 + P/100) + Q * (1 + P/100)^2) :=
by sorry

end NUMINAMATH_CALUDE_department_store_sales_fraction_l1507_150712


namespace NUMINAMATH_CALUDE_profit_and_pricing_analysis_l1507_150723

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

/-- Represents the profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 50) * (sales_quantity x)

/-- Represents the new profit function after cost price increase -/
def new_profit (x a : ℝ) : ℝ := (x - 50 - a) * (sales_quantity x)

theorem profit_and_pricing_analysis 
  (cost_price : ℝ) 
  (a : ℝ) 
  (h1 : cost_price = 50) 
  (h2 : a > 0) :
  (∃ x₁ x₂, profit x₁ = 800 ∧ profit x₂ = 800 ∧ x₁ ≠ x₂) ∧ 
  (∃ x_max, ∀ x, profit x ≤ profit x_max) ∧
  (∃ x, 50 + a ≤ x ∧ x ≤ 70 ∧ new_profit x a = 960 ∧ a = 4) := by
  sorry


end NUMINAMATH_CALUDE_profit_and_pricing_analysis_l1507_150723


namespace NUMINAMATH_CALUDE_sweets_distribution_l1507_150744

theorem sweets_distribution (num_children : ℕ) (sweets_per_child : ℕ) (remaining_fraction : ℚ) :
  num_children = 48 →
  sweets_per_child = 4 →
  remaining_fraction = 1/3 →
  (num_children * sweets_per_child) / (1 - remaining_fraction) = 288 := by
  sorry

end NUMINAMATH_CALUDE_sweets_distribution_l1507_150744


namespace NUMINAMATH_CALUDE_smallest_student_count_l1507_150734

/-- Represents the number of students in each grade --/
structure StudentCounts where
  grade9 : ℕ
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Checks if the given student counts satisfy the required ratios --/
def satisfiesRatios (counts : StudentCounts) : Prop :=
  3 * counts.grade10 = 2 * counts.grade12 ∧
  7 * counts.grade11 = 4 * counts.grade12 ∧
  5 * counts.grade9 = 3 * counts.grade12

/-- Calculates the total number of students --/
def totalStudents (counts : StudentCounts) : ℕ :=
  counts.grade9 + counts.grade10 + counts.grade11 + counts.grade12

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : StudentCounts),
    satisfiesRatios counts ∧
    totalStudents counts = 298 ∧
    (∀ (other : StudentCounts),
      satisfiesRatios other → totalStudents other ≥ 298) :=
  sorry

end NUMINAMATH_CALUDE_smallest_student_count_l1507_150734


namespace NUMINAMATH_CALUDE_abhay_sameer_speed_difference_l1507_150732

/-- Prove that when Abhay doubles his speed, he takes 1 hour less than Sameer to cover 18 km,
    given that Abhay's original speed is 3 km/h and he initially takes 2 hours more than Sameer. -/
theorem abhay_sameer_speed_difference (distance : ℝ) (abhay_speed : ℝ) (sameer_speed : ℝ) :
  distance = 18 →
  abhay_speed = 3 →
  distance / abhay_speed = distance / sameer_speed + 2 →
  distance / (2 * abhay_speed) = distance / sameer_speed - 1 :=
by sorry

end NUMINAMATH_CALUDE_abhay_sameer_speed_difference_l1507_150732


namespace NUMINAMATH_CALUDE_randy_store_trips_l1507_150787

theorem randy_store_trips (initial_amount : ℕ) (final_amount : ℕ) (amount_per_trip : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  final_amount = 104 →
  amount_per_trip = 2 →
  months_per_year = 12 →
  (initial_amount - final_amount) / amount_per_trip / months_per_year = 4 := by
  sorry

end NUMINAMATH_CALUDE_randy_store_trips_l1507_150787


namespace NUMINAMATH_CALUDE_cube_cross_section_area_l1507_150768

/-- The area of a cross-section in a cube -/
theorem cube_cross_section_area (a : ℝ) (h : a > 0) :
  let cube_edge := a
  let face_diagonal := a * Real.sqrt 2
  let space_diagonal := a * Real.sqrt 3
  let cross_section_area := (face_diagonal * space_diagonal) / 2
  cross_section_area = (a^2 * Real.sqrt 6) / 2 := by
sorry

end NUMINAMATH_CALUDE_cube_cross_section_area_l1507_150768


namespace NUMINAMATH_CALUDE_fraction_powers_equality_l1507_150799

theorem fraction_powers_equality : (0.4 ^ 4) / (0.04 ^ 3) = 400 := by
  sorry

end NUMINAMATH_CALUDE_fraction_powers_equality_l1507_150799


namespace NUMINAMATH_CALUDE_vector_coplanarity_theorem_point_coplanarity_theorem_l1507_150797

/-- A vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of coplanarity for vectors -/
def coplanar_vectors (a b p : Vector3D) : Prop :=
  ∃ (x y : ℝ), p = Vector3D.mk (x * a.x + y * b.x) (x * a.y + y * b.y) (x * a.z + y * b.z)

/-- Definition of coplanarity for points -/
def coplanar_points (M A B P : Point3D) : Prop :=
  ∃ (x y : ℝ), 
    P.x - M.x = x * (A.x - M.x) + y * (B.x - M.x) ∧
    P.y - M.y = x * (A.y - M.y) + y * (B.y - M.y) ∧
    P.z - M.z = x * (A.z - M.z) + y * (B.z - M.z)

theorem vector_coplanarity_theorem (a b p : Vector3D) :
  (∃ (x y : ℝ), p = Vector3D.mk (x * a.x + y * b.x) (x * a.y + y * b.y) (x * a.z + y * b.z)) →
  coplanar_vectors a b p :=
by sorry

theorem point_coplanarity_theorem (M A B P : Point3D) :
  (∃ (x y : ℝ), 
    P.x - M.x = x * (A.x - M.x) + y * (B.x - M.x) ∧
    P.y - M.y = x * (A.y - M.y) + y * (B.y - M.y) ∧
    P.z - M.z = x * (A.z - M.z) + y * (B.z - M.z)) →
  coplanar_points M A B P :=
by sorry

end NUMINAMATH_CALUDE_vector_coplanarity_theorem_point_coplanarity_theorem_l1507_150797


namespace NUMINAMATH_CALUDE_unique_sums_count_l1507_150719

/-- Represents the set of available coins -/
def CoinSet : Finset ℕ := {1, 2, 5, 100, 100, 100, 100, 500, 500}

/-- Generates all possible sums using the given coin set -/
def PossibleSums (coins : Finset ℕ) : Finset ℕ :=
  sorry

/-- The number of unique sums that can be formed using the given coin set -/
theorem unique_sums_count : (PossibleSums CoinSet).card = 119 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l1507_150719


namespace NUMINAMATH_CALUDE_triangles_on_ABC_l1507_150795

/-- The number of triangles that can be formed with marked points on the sides of a triangle -/
def num_triangles (points_AB points_BC points_AC : ℕ) : ℕ :=
  let total_points := points_AB + points_BC + points_AC
  let total_combinations := (total_points.choose 3)
  let invalid_combinations := (points_AB.choose 3) + (points_BC.choose 3) + (points_AC.choose 3)
  total_combinations - invalid_combinations

/-- Theorem stating the number of triangles formed with marked points on triangle ABC -/
theorem triangles_on_ABC : num_triangles 12 9 10 = 4071 := by
  sorry

end NUMINAMATH_CALUDE_triangles_on_ABC_l1507_150795
