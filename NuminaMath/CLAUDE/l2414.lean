import Mathlib

namespace triangle_area_problem_l2414_241482

/-- Given a triangle ABC with area 15 and a point D on AB such that AD:DB = 3:2,
    if there exist points E on BC and F on CA forming triangle ABE and quadrilateral DBEF
    with equal areas, then the area of triangle ABE is 9. -/
theorem triangle_area_problem (A B C D E F : ℝ × ℝ) : 
  let triangle_area (P Q R : ℝ × ℝ) := abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2
  triangle_area A B C = 15 →
  D.1 = (3 * B.1 + 2 * A.1) / 5 ∧ D.2 = (3 * B.2 + 2 * A.2) / 5 →
  E.1 = B.1 ∧ E.2 ≤ B.2 ∧ E.2 ≥ C.2 →
  F.1 ≥ C.1 ∧ F.1 ≤ A.1 ∧ F.2 = C.2 →
  triangle_area A B E = triangle_area D B E + triangle_area D E F →
  triangle_area A B E = 9 :=
by sorry

end triangle_area_problem_l2414_241482


namespace total_stickers_l2414_241453

/-- Given the following conditions:
    - There are 10 stickers on a page originally
    - There are 22 pages of stickers
    - 3 stickers are missing from each page
    Prove that the total number of stickers is 154 -/
theorem total_stickers (original_stickers : ℕ) (pages : ℕ) (missing_stickers : ℕ)
  (h1 : original_stickers = 10)
  (h2 : pages = 22)
  (h3 : missing_stickers = 3) :
  (original_stickers - missing_stickers) * pages = 154 :=
by sorry

end total_stickers_l2414_241453


namespace kantana_chocolates_l2414_241444

/-- The number of chocolates Kantana buys for herself and her sister every Saturday -/
def regular_chocolates : ℕ := 3

/-- The number of additional chocolates Kantana bought for Charlie on the last Saturday -/
def additional_chocolates : ℕ := 10

/-- The number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- The total number of chocolates Kantana bought for the month -/
def total_chocolates : ℕ := (saturdays_in_month - 1) * regular_chocolates + 
                            (regular_chocolates + additional_chocolates)

theorem kantana_chocolates : total_chocolates = 22 := by
  sorry

end kantana_chocolates_l2414_241444


namespace solution_to_equation_l2414_241485

theorem solution_to_equation : ∃ x y : ℤ, x - 3 * y = 1 ∧ x = -2 ∧ y = -1 := by
  sorry

end solution_to_equation_l2414_241485


namespace arithmetic_computation_l2414_241467

theorem arithmetic_computation : 5 + 4 * (2 - 7)^2 = 105 := by sorry

end arithmetic_computation_l2414_241467


namespace sum_of_root_products_l2414_241483

theorem sum_of_root_products (p q r s : ℂ) : 
  (4 * p^4 - 8 * p^3 + 12 * p^2 - 16 * p + 9 = 0) →
  (4 * q^4 - 8 * q^3 + 12 * q^2 - 16 * q + 9 = 0) →
  (4 * r^4 - 8 * r^3 + 12 * r^2 - 16 * r + 9 = 0) →
  (4 * s^4 - 8 * s^3 + 12 * s^2 - 16 * s + 9 = 0) →
  p * q + p * r + p * s + q * r + q * s + r * s = -3 := by
sorry

end sum_of_root_products_l2414_241483


namespace last_episode_length_correct_l2414_241443

/-- Represents the duration of a TV series viewing session -/
structure SeriesViewing where
  episodeLengths : List Nat
  breakLength : Nat
  totalTime : Nat

/-- Calculates the length of the last episode given the viewing details -/
def lastEpisodeLength (s : SeriesViewing) : Nat :=
  s.totalTime
    - (s.episodeLengths.sum + s.breakLength * s.episodeLengths.length)

theorem last_episode_length_correct (s : SeriesViewing) :
  s.episodeLengths = [58, 62, 65, 71, 79] ∧
  s.breakLength = 12 ∧
  s.totalTime = 9 * 60 →
  lastEpisodeLength s = 145 := by
  sorry

#eval lastEpisodeLength {
  episodeLengths := [58, 62, 65, 71, 79],
  breakLength := 12,
  totalTime := 9 * 60
}

end last_episode_length_correct_l2414_241443


namespace function_properties_l2414_241492

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x * (a * x + b) + x^2 + 2 * x

theorem function_properties (a b : ℝ) :
  (f a b 0 = 1 ∧ (deriv (f a b)) 0 = 4) →
  (a = 1 ∧ b = 1) ∧
  (∀ k, (∀ x ∈ Set.Icc (-2) (-1), f 1 1 x ≥ x^2 + 2*(k+1)*x + k) ↔ 
        k ≥ (1/4) * Real.exp (-3/2)) :=
by sorry

end function_properties_l2414_241492


namespace book_cost_l2414_241472

theorem book_cost (total_paid : ℕ) (change : ℕ) (pen_cost : ℕ) (ruler_cost : ℕ) 
  (h1 : total_paid = 50)
  (h2 : change = 20)
  (h3 : pen_cost = 4)
  (h4 : ruler_cost = 1) :
  total_paid - change - (pen_cost + ruler_cost) = 25 := by
  sorry

end book_cost_l2414_241472


namespace dave_candy_pieces_l2414_241437

/-- Calculates the number of candy pieces Dave has left after giving some boxes away. -/
def candyPiecesLeft (initialBoxes : ℕ) (boxesGivenAway : ℕ) (piecesPerBox : ℕ) : ℕ :=
  (initialBoxes - boxesGivenAway) * piecesPerBox

/-- Proves that Dave has 21 pieces of candy left. -/
theorem dave_candy_pieces : 
  candyPiecesLeft 12 5 3 = 21 := by
  sorry

end dave_candy_pieces_l2414_241437


namespace solve_system_with_equal_xy_l2414_241468

theorem solve_system_with_equal_xy (x y n : ℝ) 
  (eq1 : 5 * x - 4 * y = n)
  (eq2 : 3 * x + 5 * y = 8)
  (eq3 : x = y) :
  n = 1 := by
  sorry

end solve_system_with_equal_xy_l2414_241468


namespace specific_solid_surface_area_l2414_241475

/-- A solid with specific dimensions -/
structure Solid where
  front_length : ℝ
  front_width : ℝ
  left_length : ℝ
  left_width : ℝ
  top_radius : ℝ

/-- The surface area of the solid -/
def surface_area (s : Solid) : ℝ := sorry

/-- Theorem stating the surface area of the specific solid -/
theorem specific_solid_surface_area :
  ∀ s : Solid,
    s.front_length = 4 ∧
    s.front_width = 2 ∧
    s.left_length = 4 ∧
    s.left_width = 2 ∧
    s.top_radius = 2 →
    surface_area s = 16 * Real.pi :=
by sorry

end specific_solid_surface_area_l2414_241475


namespace cyclic_sum_inequality_l2414_241498

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^4 / (a^3 + a^2*b + a*b^2 + b^3) +
   b^4 / (b^3 + b^2*c + b*c^2 + c^3) +
   c^4 / (c^3 + c^2*d + c*d^2 + d^3) +
   d^4 / (d^3 + d^2*a + d*a^2 + a^3)) ≥ (a + b + c + d) / 4 := by
  sorry

end cyclic_sum_inequality_l2414_241498


namespace sports_club_members_l2414_241440

/-- A sports club with members who play badminton, tennis, both, or neither. -/
structure SportsClub where
  badminton : ℕ  -- Number of members who play badminton
  tennis : ℕ     -- Number of members who play tennis
  both : ℕ       -- Number of members who play both badminton and tennis
  neither : ℕ    -- Number of members who play neither badminton nor tennis

/-- The total number of members in the sports club -/
def SportsClub.totalMembers (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 35 -/
theorem sports_club_members (club : SportsClub)
    (h1 : club.badminton = 15)
    (h2 : club.tennis = 18)
    (h3 : club.neither = 5)
    (h4 : club.both = 3) :
    club.totalMembers = 35 := by
  sorry

end sports_club_members_l2414_241440


namespace holiday_duration_l2414_241455

theorem holiday_duration (total_rain_days : ℕ) (sunny_mornings : ℕ) (sunny_afternoons : ℕ)
  (h1 : total_rain_days = 7)
  (h2 : sunny_mornings = 5)
  (h3 : sunny_afternoons = 6) :
  ∃ (total_days : ℕ), total_days = 9 ∧ total_days ≥ 9 := by
  sorry

end holiday_duration_l2414_241455


namespace solve_for_s_l2414_241400

theorem solve_for_s (m : ℝ) (s : ℝ) 
  (h1 : 5 = m * (3 ^ s)) 
  (h2 : 45 = m * (9 ^ s)) : 
  s = 2 := by
sorry

end solve_for_s_l2414_241400


namespace smallest_absolute_value_l2414_241494

theorem smallest_absolute_value : ∀ (a b c : ℝ),
  a = 4.1 → b = 13 → c = 3 →
  |(-Real.sqrt 7)| < |a| ∧ |(-Real.sqrt 7)| < Real.sqrt b ∧ |(-Real.sqrt 7)| < |c| :=
by sorry

end smallest_absolute_value_l2414_241494


namespace smallest_chocolate_beverage_volume_l2414_241419

/-- Represents the ratio of milk to syrup in the chocolate beverage -/
def milk_syrup_ratio : ℚ := 5 / 2

/-- Volume of milk in each bottle (in liters) -/
def milk_bottle_volume : ℚ := 2

/-- Volume of syrup in each bottle (in liters) -/
def syrup_bottle_volume : ℚ := 14 / 10

/-- Finds the smallest number of whole bottles of milk and syrup that satisfy the ratio -/
def find_smallest_bottles : ℕ × ℕ := (7, 4)

/-- Calculates the total volume of the chocolate beverage -/
def total_volume (bottles : ℕ × ℕ) : ℚ :=
  milk_bottle_volume * bottles.1 + syrup_bottle_volume * bottles.2

/-- Theorem stating that the smallest volume of chocolate beverage that can be made
    using only whole bottles of milk and syrup is 19.6 L -/
theorem smallest_chocolate_beverage_volume :
  total_volume (find_smallest_bottles) = 196 / 10 := by
  sorry

end smallest_chocolate_beverage_volume_l2414_241419


namespace difference_of_squares_640_360_l2414_241464

theorem difference_of_squares_640_360 : 640^2 - 360^2 = 280000 := by
  sorry

end difference_of_squares_640_360_l2414_241464


namespace total_shells_l2414_241433

theorem total_shells (morning_shells afternoon_shells : ℕ) 
  (h1 : morning_shells = 292)
  (h2 : afternoon_shells = 324) :
  morning_shells + afternoon_shells = 616 := by
  sorry

end total_shells_l2414_241433


namespace cylinder_surface_area_l2414_241474

theorem cylinder_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 1) :
  2 * Real.pi * r * (r + h) = 4 * Real.pi :=
by sorry

end cylinder_surface_area_l2414_241474


namespace salad_vegetables_count_l2414_241486

theorem salad_vegetables_count :
  ∀ (cucumbers tomatoes total : ℕ),
  cucumbers = 70 →
  tomatoes = 3 * cucumbers →
  total = cucumbers + tomatoes →
  total = 280 :=
by
  sorry

end salad_vegetables_count_l2414_241486


namespace oil_volume_in_liters_l2414_241488

def bottle_volume : ℝ := 200
def num_bottles : ℕ := 20
def ml_per_liter : ℝ := 1000

theorem oil_volume_in_liters :
  (bottle_volume * num_bottles) / ml_per_liter = 4 := by
  sorry

end oil_volume_in_liters_l2414_241488


namespace triangle_side_range_l2414_241499

open Real

theorem triangle_side_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- ABC is an acute triangle
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  Real.sqrt 3 * (a * cos B + b * cos A) = 2 * c * sin C ∧  -- Given equation
  b = 1 →  -- Given condition
  sqrt 3 / 2 < c ∧ c < sqrt 3 :=
by sorry

end triangle_side_range_l2414_241499


namespace bob_spending_is_26_l2414_241480

-- Define the prices and quantities
def bread_price : ℚ := 2
def bread_quantity : ℕ := 4
def cheese_price : ℚ := 6
def cheese_quantity : ℕ := 2
def chocolate_price : ℚ := 3
def chocolate_quantity : ℕ := 3
def oil_price : ℚ := 10
def oil_quantity : ℕ := 1

-- Define the discount and coupon
def cheese_discount : ℚ := 0.25
def coupon_value : ℚ := 10
def coupon_threshold : ℚ := 30

-- Define Bob's spending function
def bob_spending : ℚ :=
  let bread_total := bread_price * bread_quantity
  let cheese_total := cheese_price * cheese_quantity * (1 - cheese_discount)
  let chocolate_total := chocolate_price * chocolate_quantity
  let oil_total := oil_price * oil_quantity
  let subtotal := bread_total + cheese_total + chocolate_total + oil_total
  if subtotal ≥ coupon_threshold then subtotal - coupon_value else subtotal

-- Theorem to prove
theorem bob_spending_is_26 : bob_spending = 26 := by sorry

end bob_spending_is_26_l2414_241480


namespace translation_result_l2414_241448

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  ⟨p.x + t.dx, p.y + t.dy⟩

theorem translation_result :
  let A : Point := ⟨-3, 2⟩
  let t : Translation := ⟨3, -2⟩
  applyTranslation A t = ⟨0, 0⟩ := by sorry

end translation_result_l2414_241448


namespace basketball_win_calculation_l2414_241401

/-- Proves the number of games a basketball team needs to win to achieve a specific win percentage -/
theorem basketball_win_calculation (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) 
  (target_percentage : ℚ) (h1 : total_games = first_games + remaining_games) 
  (h2 : total_games = 100) (h3 : first_games = 45) (h4 : first_wins = 30) 
  (h5 : remaining_games = 55) (h6 : target_percentage = 65 / 100) : 
  ∃ (x : ℕ), (first_wins + x : ℚ) / total_games = target_percentage ∧ x = 35 := by
sorry

end basketball_win_calculation_l2414_241401


namespace new_year_fireworks_display_l2414_241430

def fireworks_per_number : ℕ := 6
def fireworks_per_letter : ℕ := 5
def additional_boxes : ℕ := 50
def fireworks_per_box : ℕ := 8

def year_numbers : ℕ := 4
def phrase_letters : ℕ := 12

theorem new_year_fireworks_display :
  let year_fireworks := year_numbers * fireworks_per_number
  let phrase_fireworks := phrase_letters * fireworks_per_letter
  let additional_fireworks := additional_boxes * fireworks_per_box
  year_fireworks + phrase_fireworks + additional_fireworks = 476 := by
sorry

end new_year_fireworks_display_l2414_241430


namespace fraction_less_than_one_l2414_241416

theorem fraction_less_than_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b / a < 1 := by
  sorry

end fraction_less_than_one_l2414_241416


namespace julian_comic_frames_l2414_241489

/-- Calculates the total number of frames in Julian's comic book --/
def total_frames (total_pages : Nat) (avg_frames : Nat) (pages_305 : Nat) (pages_250 : Nat) : Nat :=
  let frames_305 := pages_305 * 305
  let frames_250 := pages_250 * 250
  let remaining_pages := total_pages - pages_305 - pages_250
  let frames_avg := remaining_pages * avg_frames
  frames_305 + frames_250 + frames_avg

/-- Proves that the total number of frames in Julian's comic book is 7040 --/
theorem julian_comic_frames :
  total_frames 25 280 10 7 = 7040 := by
  sorry

end julian_comic_frames_l2414_241489


namespace sum_15_27_in_base4_l2414_241471

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_15_27_in_base4 :
  toBase4 (15 + 27) = [2, 2, 2] :=
sorry

end sum_15_27_in_base4_l2414_241471


namespace profit_maximization_profit_function_correct_sales_at_price_l2414_241429

/-- Represents the daily profit function for a product -/
def profit_function (x : ℝ) : ℝ := (200 - x) * (x - 120)

/-- The cost price of the product -/
def cost_price : ℝ := 120

/-- The reference price point -/
def reference_price : ℝ := 130

/-- The daily sales at the reference price -/
def reference_sales : ℝ := 70

/-- The rate of change in sales with respect to price -/
def sales_price_ratio : ℝ := -1

theorem profit_maximization :
  ∃ (max_price max_profit : ℝ),
    (∀ x, profit_function x ≤ max_profit) ∧
    profit_function max_price = max_profit ∧
    max_price = 160 ∧
    max_profit = 1600 := by sorry

theorem profit_function_correct :
  ∀ x, profit_function x = (200 - x) * (x - cost_price) := by sorry

theorem sales_at_price (x : ℝ) :
  x ≥ reference_price →
  profit_function x = (reference_sales + sales_price_ratio * (x - reference_price)) * (x - cost_price) := by sorry

end profit_maximization_profit_function_correct_sales_at_price_l2414_241429


namespace fourth_intersection_point_l2414_241450

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 4 -/
def curve (p : Point) : Prop := p.x * p.y = 4

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

theorem fourth_intersection_point (c : Circle) 
    (h1 : curve (Point.mk 4 1) ∧ onCircle (Point.mk 4 1) c)
    (h2 : curve (Point.mk (-2) (-2)) ∧ onCircle (Point.mk (-2) (-2)) c)
    (h3 : curve (Point.mk 8 (1/2)) ∧ onCircle (Point.mk 8 (1/2)) c)
    (h4 : ∃ p : Point, curve p ∧ onCircle p c ∧ p ≠ Point.mk 4 1 ∧ p ≠ Point.mk (-2) (-2) ∧ p ≠ Point.mk 8 (1/2)) :
    ∃ p : Point, p = Point.mk (-1/4) (-16) ∧ curve p ∧ onCircle p c := by
  sorry

end fourth_intersection_point_l2414_241450


namespace inscribed_cube_volume_l2414_241420

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by sorry

end inscribed_cube_volume_l2414_241420


namespace potato_bag_weight_l2414_241454

theorem potato_bag_weight (original_weight : ℝ) : 
  (original_weight / (original_weight / 2) = 36) → original_weight = 648 :=
by
  sorry

end potato_bag_weight_l2414_241454


namespace cars_meet_time_l2414_241425

/-- Two cars meet on a highway -/
theorem cars_meet_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 333)
  (h2 : speed1 = 54) (h3 : speed2 = 57) :
  (highway_length / (speed1 + speed2) : ℝ) = 3 := by
  sorry

end cars_meet_time_l2414_241425


namespace total_sums_attempted_l2414_241417

theorem total_sums_attempted (correct : ℕ) (wrong : ℕ) : 
  correct = 25 →
  wrong = 2 * correct →
  correct + wrong = 75 := by
sorry

end total_sums_attempted_l2414_241417


namespace roses_ratio_l2414_241431

/-- Proves that the ratio of roses given to Susan's daughter to the total number of roses in the bouquet is 1:2 -/
theorem roses_ratio (total : ℕ) (vase : ℕ) (daughter : ℕ) : 
  total = 3 * 12 →
  total = vase + daughter →
  vase = 18 →
  12 = (2/3) * vase →
  daughter / total = 1/2 := by
  sorry

end roses_ratio_l2414_241431


namespace dog_count_l2414_241456

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  rollOver : ℕ
  sitStay : ℕ
  stayRollOver : ℕ
  sitRollOver : ℕ
  allThree : ℕ
  none : ℕ
  stayRollOverPlayDead : ℕ

/-- The total number of dogs in the training center -/
def totalDogs (d : DogTricks) : ℕ := sorry

/-- Theorem stating the total number of dogs in the training center -/
theorem dog_count (d : DogTricks) 
  (h1 : d.sit = 60)
  (h2 : d.stay = 35)
  (h3 : d.rollOver = 40)
  (h4 : d.sitStay = 22)
  (h5 : d.stayRollOver = 15)
  (h6 : d.sitRollOver = 20)
  (h7 : d.allThree = 10)
  (h8 : d.none = 10)
  (h9 : d.stayRollOverPlayDead = 5)
  (h10 : d.stayRollOverPlayDead ≤ d.stayRollOver) :
  totalDogs d = 98 := by
  sorry

end dog_count_l2414_241456


namespace prime_sum_10_product_21_l2414_241406

theorem prime_sum_10_product_21 (p q : ℕ) : 
  Prime p → Prime q → p ≠ q → p + q = 10 → p * q = 21 := by sorry

end prime_sum_10_product_21_l2414_241406


namespace sum_of_coefficients_l2414_241463

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₀ + a₁ + a₂ + a₃ = 27 := by
sorry

end sum_of_coefficients_l2414_241463


namespace cuboid_area_example_l2414_241438

/-- The surface area of a cuboid -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 3 cm, length 4 cm, and height 5 cm is 94 cm² -/
theorem cuboid_area_example : cuboid_surface_area 3 4 5 = 94 := by
  sorry

end cuboid_area_example_l2414_241438


namespace first_number_is_seven_l2414_241418

/-- A sequence of 8 numbers where each number starting from the third
    is the sum of the two previous numbers. -/
def FibonacciLikeSequence (a : Fin 8 → ℕ) : Prop :=
  ∀ i : Fin 8, i.val ≥ 2 → a i = a (Fin.sub i 1) + a (Fin.sub i 2)

/-- Theorem stating that if the 5th number is 53 and the 8th number is 225
    in a Fibonacci-like sequence of 8 numbers, then the 1st number is 7. -/
theorem first_number_is_seven
  (a : Fin 8 → ℕ)
  (h_seq : FibonacciLikeSequence a)
  (h_fifth : a 4 = 53)
  (h_eighth : a 7 = 225) :
  a 0 = 7 := by
  sorry


end first_number_is_seven_l2414_241418


namespace work_days_calculation_l2414_241460

/-- Proves that A and B worked together for 10 days given the conditions -/
theorem work_days_calculation (a_rate : ℚ) (b_rate : ℚ) (remaining_work : ℚ) : 
  a_rate = 1 / 30 →
  b_rate = 1 / 40 →
  remaining_work = 5 / 12 →
  ∃ d : ℚ, d = 10 ∧ (a_rate + b_rate) * d = 1 - remaining_work :=
by sorry

end work_days_calculation_l2414_241460


namespace x_minus_y_equals_half_l2414_241458

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {2, 0, x}
def B (x y : ℝ) : Set ℝ := {1/x, |x|, y/x}

-- State the theorem
theorem x_minus_y_equals_half (x y : ℝ) : A x = B x y → x - y = 1/2 := by
  sorry

end x_minus_y_equals_half_l2414_241458


namespace diaries_count_l2414_241445

/-- The number of diaries Natalie's sister has after buying and losing some -/
def final_diaries : ℕ :=
  let initial : ℕ := 23
  let bought : ℕ := 5 * initial
  let total : ℕ := initial + bought
  let lost : ℕ := (7 * total) / 9
  total - lost

theorem diaries_count : final_diaries = 31 := by
  sorry

end diaries_count_l2414_241445


namespace number_of_cows_l2414_241479

/-- The number of cows in a field with a total of 200 animals, 56 sheep, and 104 goats. -/
theorem number_of_cows (total : ℕ) (sheep : ℕ) (goats : ℕ) (h1 : total = 200) (h2 : sheep = 56) (h3 : goats = 104) :
  total - sheep - goats = 40 := by
  sorry

end number_of_cows_l2414_241479


namespace five_by_five_uncoverable_l2414_241452

/-- A checkerboard that can be completely covered by dominoes. -/
structure CoverableCheckerboard where
  rows : ℕ
  cols : ℕ
  even_rows : Even rows
  even_cols : Even cols
  even_total : Even (rows * cols)

/-- A domino covers exactly two squares. -/
def domino_covers : ℕ := 2

/-- Theorem stating that a 5x5 checkerboard cannot be completely covered by dominoes. -/
theorem five_by_five_uncoverable :
  ¬ ∃ (c : CoverableCheckerboard), c.rows = 5 ∧ c.cols = 5 :=
sorry

end five_by_five_uncoverable_l2414_241452


namespace bucket_calculation_reduced_capacity_buckets_l2414_241409

/-- Given a tank that requires a certain number of buckets to fill and a reduction in bucket capacity,
    calculate the new number of buckets required to fill the tank. -/
theorem bucket_calculation (original_buckets : ℕ) (capacity_reduction : ℚ) : 
  original_buckets / capacity_reduction = original_buckets * (1 / capacity_reduction) :=
by sorry

/-- Prove that 105 buckets are required when the original number of buckets is 42
    and the capacity is reduced to two-fifths. -/
theorem reduced_capacity_buckets : 
  let original_buckets : ℕ := 42
  let capacity_reduction : ℚ := 2 / 5
  original_buckets / capacity_reduction = 105 :=
by sorry

end bucket_calculation_reduced_capacity_buckets_l2414_241409


namespace polynomial_simplification_l2414_241404

theorem polynomial_simplification (x : ℝ) :
  (2 * x^13 + 3 * x^12 - 4 * x^9 + 5 * x^7) + 
  (8 * x^11 - 2 * x^9 + 3 * x^7 + 6 * x^4 - 7 * x + 9) + 
  (x^13 + 4 * x^12 + x^11 + 9 * x^9) = 
  3 * x^13 + 7 * x^12 + 9 * x^11 + 3 * x^9 + 8 * x^7 + 6 * x^4 - 7 * x + 9 := by
sorry

end polynomial_simplification_l2414_241404


namespace perfect_rectangle_theorem_l2414_241462

/-- Represents a perfect rectangle divided into squares -/
structure PerfectRectangle where
  squares : List ℕ
  is_perfect : squares.length > 0

/-- The specific perfect rectangle from the problem -/
def given_rectangle : PerfectRectangle where
  squares := [9, 16, 2, 5, 7, 25, 28, 33]
  is_perfect := by simp

/-- Checks if the list is sorted in ascending order -/
def is_sorted (l : List ℕ) : Prop :=
  ∀ i j, i < j → j < l.length → l[i]! ≤ l[j]!

/-- The main theorem to prove -/
theorem perfect_rectangle_theorem (rect : PerfectRectangle) :
  rect = given_rectangle →
  is_sorted (rect.squares.filter (λ x => x ≠ 9 ∧ x ≠ 16)) ∧
  (rect.squares.filter (λ x => x ≠ 9 ∧ x ≠ 16)).length = 6 :=
by sorry

end perfect_rectangle_theorem_l2414_241462


namespace sphere_in_cube_volume_l2414_241441

/-- The volume of a sphere inscribed in a cube of edge length 2 -/
theorem sphere_in_cube_volume :
  let cube_edge : ℝ := 2
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (4 / 3) * Real.pi := by
  sorry

end sphere_in_cube_volume_l2414_241441


namespace toll_formula_correct_l2414_241428

/-- Represents the toll formula for a truck crossing a bridge -/
def toll_formula (x : ℕ) : ℚ := 0.50 + 0.30 * x

/-- Represents an 18-wheel truck with 2 wheels on its front axle and 4 wheels on each other axle -/
def eighteen_wheel_truck : ℕ := 5

theorem toll_formula_correct : 
  toll_formula eighteen_wheel_truck = 2 := by sorry

end toll_formula_correct_l2414_241428


namespace aunt_gift_amount_l2414_241413

def birthday_money_problem (grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money : ℕ) : Prop :=
  grandmother_gift = 20 ∧
  uncle_gift = 30 ∧
  total_money = 125 ∧
  game_cost = 35 ∧
  games_bought = 3 ∧
  remaining_money = 20 ∧
  total_money = grandmother_gift + aunt_gift + uncle_gift ∧
  total_money = game_cost * games_bought + remaining_money

theorem aunt_gift_amount :
  ∀ (grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money : ℕ),
    birthday_money_problem grandmother_gift aunt_gift uncle_gift total_money game_cost games_bought remaining_money →
    aunt_gift = 75 := by
  sorry

end aunt_gift_amount_l2414_241413


namespace fraction_change_l2414_241411

theorem fraction_change (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (2*x + 2*y) / (2*x * 2*y) = (1/2) * ((x + y) / (x * y)) :=
by sorry

end fraction_change_l2414_241411


namespace fuchsia_to_mauve_amount_correct_l2414_241439

/-- Represents the composition of paint in parts --/
structure PaintComposition where
  red : ℚ
  blue : ℚ

/-- The amount of fuchsia paint being changed to mauve paint --/
def fuchsia_amount : ℚ := 106.68

/-- The composition of fuchsia paint --/
def fuchsia : PaintComposition := { red := 5, blue := 3 }

/-- The composition of mauve paint --/
def mauve : PaintComposition := { red := 3, blue := 5 }

/-- The amount of blue paint added to change fuchsia to mauve --/
def blue_added : ℚ := 26.67

/-- Theorem stating that the calculated amount of fuchsia paint is correct --/
theorem fuchsia_to_mauve_amount_correct :
  fuchsia_amount * (fuchsia.blue / (fuchsia.red + fuchsia.blue)) + blue_added =
  fuchsia_amount * (mauve.blue / (mauve.red + mauve.blue)) := by
  sorry

end fuchsia_to_mauve_amount_correct_l2414_241439


namespace variation_relationship_l2414_241470

theorem variation_relationship (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by sorry

end variation_relationship_l2414_241470


namespace remainder_theorem_l2414_241446

theorem remainder_theorem (x y u v : ℤ) 
  (x_pos : 0 < x) (y_pos : 0 < y) 
  (division : x = u * y + v) (rem_bound : 0 ≤ v ∧ v < y) : 
  (x + y * u^2 + 3 * v) % y = (4 * v) % y := by
  sorry

end remainder_theorem_l2414_241446


namespace student_number_problem_l2414_241451

theorem student_number_problem (x : ℝ) : 2 * x - 200 = 110 → x = 155 := by
  sorry

end student_number_problem_l2414_241451


namespace tims_score_is_2352_l2414_241497

-- Define the first 8 prime numbers
def first_8_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of the first 8 prime numbers
def prime_product : Nat := first_8_primes.prod

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Define N as the sum of digits in the product of the first 8 prime numbers
def N : Nat := sum_of_digits prime_product

-- Define Tim's score as the sum of the first N even numbers
def tims_score : Nat := N * (N + 1)

-- The theorem to prove
theorem tims_score_is_2352 : tims_score = 2352 := by sorry

end tims_score_is_2352_l2414_241497


namespace collinear_points_sum_l2414_241477

/-- Three points in 3D space are collinear if they all lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t s : ℝ), p2 = p1 + t • (p3 - p1) ∧ p3 = p1 + s • (p3 - p1)

/-- If the points (2,a,b), (a,3,b), and (a,b,4) are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

#check collinear_points_sum

end collinear_points_sum_l2414_241477


namespace range_a_theorem_l2414_241447

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀ + (a - 1) * x₀ + 1 < 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a > 3 ∨ (a ≥ -1 ∧ a ≤ 1)

-- State the theorem
theorem range_a_theorem (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
by sorry

end range_a_theorem_l2414_241447


namespace accommodation_theorem_l2414_241402

/-- The number of ways to accommodate 6 people in 5 rooms --/
def accommodationWays : ℕ := 39600

/-- The number of ways to accommodate 6 people in 5 rooms with each room having at least one person --/
def waysWithAllRoomsOccupied : ℕ := 3600

/-- The number of ways to accommodate 6 people in 5 rooms with exactly one room left empty --/
def waysWithOneRoomEmpty : ℕ := 36000

/-- The number of people --/
def numPeople : ℕ := 6

/-- The number of rooms --/
def numRooms : ℕ := 5

theorem accommodation_theorem :
  accommodationWays = waysWithAllRoomsOccupied + waysWithOneRoomEmpty ∧
  numPeople = 6 ∧
  numRooms = 5 := by
  sorry

end accommodation_theorem_l2414_241402


namespace car_speed_second_hour_l2414_241427

/-- Theorem: Given a car's speed of 145 km/h in the first hour and an average speed of 102.5 km/h over two hours, the speed in the second hour is 60 km/h. -/
theorem car_speed_second_hour (speed_first_hour : ℝ) (average_speed : ℝ) (speed_second_hour : ℝ) :
  speed_first_hour = 145 →
  average_speed = 102.5 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_second_hour = 60 :=
by
  sorry

end car_speed_second_hour_l2414_241427


namespace right_triangle_area_l2414_241490

theorem right_triangle_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 10^2 → a + b + 10 = 24 → (1/2) * a * b = 24 := by
  sorry

end right_triangle_area_l2414_241490


namespace f_derivative_at_one_l2414_241422

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x

theorem f_derivative_at_one :
  deriv f 1 = 2 * Real.log 2 + 1 / Real.log 2 := by
  sorry

end f_derivative_at_one_l2414_241422


namespace arithmetic_geometric_progression_y_value_l2414_241412

theorem arithmetic_geometric_progression_y_value
  (x y z : ℝ)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (arithmetic_prog : 2 * y = x + z)
  (geometric_prog1 : ∃ r : ℝ, r ≠ 0 ∧ -y = r * (x + 1) ∧ z = r * (-y))
  (geometric_prog2 : ∃ s : ℝ, s ≠ 0 ∧ y = s * x ∧ z + 2 = s * y) :
  y = 12 := by
sorry

end arithmetic_geometric_progression_y_value_l2414_241412


namespace min_value_of_sum_l2414_241465

theorem min_value_of_sum (x y : ℝ) : 
  x > 1 → 
  y > 1 → 
  2 * Real.log 2 = Real.log x + Real.log y → 
  x + y ≥ 200 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ 2 * Real.log 2 = Real.log x + Real.log y ∧ x + y = 200 :=
by sorry

end min_value_of_sum_l2414_241465


namespace profit_per_meter_cloth_l2414_241414

theorem profit_per_meter_cloth (meters_sold : ℕ) (selling_price : ℕ) (cost_price_per_meter : ℕ) 
  (h1 : meters_sold = 66)
  (h2 : selling_price = 660)
  (h3 : cost_price_per_meter = 5) :
  (selling_price - meters_sold * cost_price_per_meter) / meters_sold = 5 :=
by
  sorry

end profit_per_meter_cloth_l2414_241414


namespace shooting_probabilities_l2414_241484

-- Define the probabilities
def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3

-- Define the event of hitting the target exactly twice
def hit_twice : ℚ := prob_A * prob_B

-- Define the event of hitting the target at least once
def hit_at_least_once : ℚ := 1 - (1 - prob_A) * (1 - prob_B)

-- Theorem to prove
theorem shooting_probabilities :
  (hit_twice = 1/6) ∧ (hit_at_least_once = 1 - 1/2 * 2/3) :=
sorry

end shooting_probabilities_l2414_241484


namespace largest_multiple_of_nine_below_negative_seventy_l2414_241491

theorem largest_multiple_of_nine_below_negative_seventy :
  ∀ n : ℤ, n % 9 = 0 ∧ n < -70 → n ≤ -72 :=
by
  sorry

end largest_multiple_of_nine_below_negative_seventy_l2414_241491


namespace min_value_trig_function_l2414_241432

theorem min_value_trig_function (x : ℝ) : 
  Real.sin x ^ 4 + Real.cos x ^ 4 + (1 / Real.cos x) ^ 4 + (1 / Real.sin x) ^ 4 ≥ 17 / 2 :=
by sorry

end min_value_trig_function_l2414_241432


namespace arithmetic_sequence_property_l2414_241423

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SpecificSum (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SpecificSum a) : 
  a 9 - (1/2) * a 10 = 12 := by
  sorry

end arithmetic_sequence_property_l2414_241423


namespace bernoulli_inequality_l2414_241476

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x > 0) (hn : n > 1) :
  (1 + x)^n > 1 + n * x := by
  sorry

end bernoulli_inequality_l2414_241476


namespace flatbread_division_l2414_241487

-- Define a planar region
def PlanarRegion : Type := Set (ℝ × ℝ)

-- Define the area of a planar region
noncomputable def area (R : PlanarRegion) : ℝ := sorry

-- Define a line in 2D space
def Line : Type := Set (ℝ × ℝ)

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the division of a planar region by two lines
def divide (R : PlanarRegion) (l1 l2 : Line) : List PlanarRegion := sorry

-- Theorem statement
theorem flatbread_division (R : PlanarRegion) (P : ℝ) (h : area R = P) :
  ∃ (l1 l2 : Line), perpendicular l1 l2 ∧ 
    ∀ (part : PlanarRegion), part ∈ divide R l1 l2 → area part = P / 4 := by
  sorry

end flatbread_division_l2414_241487


namespace b_minus_c_equals_one_l2414_241403

theorem b_minus_c_equals_one (A B C : ℤ) 
  (h1 : A = 9 - 4)
  (h2 : B = A + 5)
  (h3 : C - 8 = 1)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  B - C = 1 := by
  sorry

end b_minus_c_equals_one_l2414_241403


namespace equationA_is_linear_l2414_241408

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and at least one of a or b is non-zero. --/
def IsLinearInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x/2 + 3y = 2 --/
def EquationA (x y : ℝ) : Prop := x/2 + 3*y = 2

theorem equationA_is_linear : IsLinearInTwoVariables EquationA := by
  sorry


end equationA_is_linear_l2414_241408


namespace quiz_winning_probability_l2414_241435

-- Define the quiz parameters
def num_questions : ℕ := 4
def num_choices : ℕ := 3
def min_correct : ℕ := 3

-- Define the probability of guessing one question correctly
def prob_correct : ℚ := 1 / num_choices

-- Define the probability of guessing one question incorrectly
def prob_incorrect : ℚ := 1 - prob_correct

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of winning
def prob_winning : ℚ :=
  (binomial num_questions num_questions) * (prob_correct ^ num_questions) +
  (binomial num_questions min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct))

-- Theorem statement
theorem quiz_winning_probability :
  prob_winning = 1 / 9 := by sorry

end quiz_winning_probability_l2414_241435


namespace log_problem_trig_problem_l2414_241466

theorem log_problem (k : ℝ) (p : ℝ) 
  (h : Real.log 210 + Real.log k - Real.log 56 + Real.log 40 - Real.log 120 + Real.log 25 = p) : 
  p = 3 := by sorry

theorem trig_problem (A : ℝ) (q : ℝ) 
  (h1 : Real.sin A = 3 / 5) 
  (h2 : Real.cos A / Real.tan A = q / 15) : 
  q = 16 := by sorry

end log_problem_trig_problem_l2414_241466


namespace bus_walk_distance_difference_l2414_241449

/-- Craig's route home from school -/
structure Route where
  busA : ℝ
  walk1 : ℝ
  busB : ℝ
  walk2 : ℝ
  busC : ℝ
  walk3 : ℝ

/-- Calculate the total bus distance -/
def totalBusDistance (r : Route) : ℝ :=
  r.busA + r.busB + r.busC

/-- Calculate the total walking distance -/
def totalWalkDistance (r : Route) : ℝ :=
  r.walk1 + r.walk2 + r.walk3

/-- Craig's actual route -/
def craigsRoute : Route :=
  { busA := 1.25
  , walk1 := 0.35
  , busB := 2.68
  , walk2 := 0.47
  , busC := 3.27
  , walk3 := 0.21 }

/-- Theorem: The difference between total bus distance and total walking distance is 6.17 miles -/
theorem bus_walk_distance_difference :
  totalBusDistance craigsRoute - totalWalkDistance craigsRoute = 6.17 := by
  sorry

end bus_walk_distance_difference_l2414_241449


namespace not_sum_of_three_squares_l2414_241495

theorem not_sum_of_three_squares (n : ℕ) : ¬ ∃ (a b c : ℕ+), (8 * n - 1 : ℤ) = a ^ 2 + b ^ 2 + c ^ 2 := by
  sorry

end not_sum_of_three_squares_l2414_241495


namespace line_through_points_l2414_241410

/-- Theorem: Line passing through specific points with given conditions -/
theorem line_through_points (k x y : ℚ) : 
  (k + 4) / 4 = k →  -- slope condition
  x - y = 2 →        -- condition on x and y
  k - x = 3 →        -- condition on k and x
  k = 4/3 ∧ x = -5/3 ∧ y = -11/3 := by
  sorry

end line_through_points_l2414_241410


namespace calories_in_one_bar_l2414_241426

/-- The number of calories in 11 candy bars -/
def total_calories : ℕ := 341

/-- The number of candy bars -/
def num_bars : ℕ := 11

/-- The number of calories in one candy bar -/
def calories_per_bar : ℕ := total_calories / num_bars

theorem calories_in_one_bar : calories_per_bar = 31 := by
  sorry

end calories_in_one_bar_l2414_241426


namespace mobile_plan_comparison_l2414_241407

/-- Represents the monthly cost in yuan for a mobile phone plan -/
def monthly_cost (rental : ℝ) (rate : ℝ) (duration : ℝ) : ℝ :=
  rental + rate * duration

/-- The monthly rental fee for Global Call in yuan -/
def global_call_rental : ℝ := 50

/-- The per-minute call rate for Global Call in yuan -/
def global_call_rate : ℝ := 0.4

/-- The monthly rental fee for Shenzhouxing in yuan -/
def shenzhouxing_rental : ℝ := 0

/-- The per-minute call rate for Shenzhouxing in yuan -/
def shenzhouxing_rate : ℝ := 0.6

/-- The breakeven point in minutes where both plans cost the same -/
def breakeven_point : ℝ := 250

theorem mobile_plan_comparison :
  ∀ duration : ℝ,
    duration > breakeven_point →
      monthly_cost global_call_rental global_call_rate duration <
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration ∧
    duration < breakeven_point →
      monthly_cost global_call_rental global_call_rate duration >
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration ∧
    duration = breakeven_point →
      monthly_cost global_call_rental global_call_rate duration =
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration :=
by
  sorry

end mobile_plan_comparison_l2414_241407


namespace marathon_first_hour_distance_l2414_241459

/-- Represents a marathon runner's performance -/
structure MarathonRunner where
  initialPace : ℝ  -- Initial pace in miles per hour
  totalDistance : ℝ -- Total marathon distance in miles
  totalTime : ℝ     -- Total race time in hours
  remainingPaceFactor : ℝ -- Factor for remaining pace (e.g., 0.8 for 80%)

/-- Calculates the distance covered in the first hour -/
def distanceInFirstHour (runner : MarathonRunner) : ℝ :=
  runner.initialPace

/-- Calculates the remaining distance after the first hour -/
def remainingDistance (runner : MarathonRunner) : ℝ :=
  runner.totalDistance - distanceInFirstHour runner

/-- Calculates the time spent running the remaining distance -/
def remainingTime (runner : MarathonRunner) : ℝ :=
  runner.totalTime - 1

/-- Theorem: The distance covered in the first hour of a 26-mile marathon is 10 miles -/
theorem marathon_first_hour_distance
  (runner : MarathonRunner)
  (h1 : runner.totalDistance = 26)
  (h2 : runner.totalTime = 3)
  (h3 : runner.remainingPaceFactor = 0.8)
  (h4 : remainingTime runner = 2)
  (h5 : remainingDistance runner / (runner.initialPace * runner.remainingPaceFactor) = remainingTime runner) :
  distanceInFirstHour runner = 10 := by
  sorry


end marathon_first_hour_distance_l2414_241459


namespace smallest_integer_solution_unique_smallest_solution_l2414_241457

theorem smallest_integer_solution (x : ℤ) : (10 - 5 * x < -18) ↔ x ≥ 6 :=
  sorry

theorem unique_smallest_solution : ∃! x : ℤ, (10 - 5 * x < -18) ∧ ∀ y : ℤ, (10 - 5 * y < -18) → x ≤ y :=
  sorry

end smallest_integer_solution_unique_smallest_solution_l2414_241457


namespace cos_alpha_for_point_P_l2414_241481

/-- If the terminal side of angle α passes through point P(a, 2a) where a < 0, then cos(α) = -√5/5 -/
theorem cos_alpha_for_point_P (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : ∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = a ∧ r * (Real.sin α) = 2*a) : 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end cos_alpha_for_point_P_l2414_241481


namespace k_range_for_not_in_second_quadrant_l2414_241469

/-- A linear function that does not pass through the second quadrant -/
structure LinearFunctionNotInSecondQuadrant where
  k : ℝ
  not_in_second_quadrant : ∀ x y : ℝ, y = k * x - k + 3 → ¬(x < 0 ∧ y > 0)

/-- The range of k for a linear function not passing through the second quadrant -/
theorem k_range_for_not_in_second_quadrant (f : LinearFunctionNotInSecondQuadrant) : f.k ≥ 3 := by
  sorry

end k_range_for_not_in_second_quadrant_l2414_241469


namespace bakery_pies_relation_l2414_241493

/-- The number of pies Mcgee's Bakery sold -/
def mcgees_pies : ℕ := 16

/-- The number of pies Smith's Bakery sold -/
def smiths_pies : ℕ := 70

/-- The difference between Smith's pies and the multiple of Mcgee's pies -/
def difference : ℕ := 6

/-- The multiple of Mcgee's pies related to Smith's pies -/
def multiple : ℕ := 4

theorem bakery_pies_relation :
  multiple * mcgees_pies + difference = smiths_pies :=
by sorry

end bakery_pies_relation_l2414_241493


namespace quadratic_equation_solution_l2414_241405

theorem quadratic_equation_solution (a b : ℝ) : 
  (a * 1^2 + b * 1 + 2 = 0) → (2023 - a - b = 2025) := by
  sorry

end quadratic_equation_solution_l2414_241405


namespace factory_wage_problem_l2414_241496

/-- Proves that the hourly rate for the remaining employees is $17 given the problem conditions -/
theorem factory_wage_problem (total_employees : ℕ) (employees_at_12 : ℕ) (employees_at_14 : ℕ)
  (shift_length : ℕ) (total_cost : ℕ) :
  total_employees = 300 →
  employees_at_12 = 200 →
  employees_at_14 = 40 →
  shift_length = 8 →
  total_cost = 31840 →
  let remaining_employees := total_employees - (employees_at_12 + employees_at_14)
  let remaining_cost := total_cost - (employees_at_12 * 12 * shift_length + employees_at_14 * 14 * shift_length)
  remaining_cost / (remaining_employees * shift_length) = 17 := by
  sorry

#check factory_wage_problem

end factory_wage_problem_l2414_241496


namespace min_value_theorem_l2414_241442

theorem min_value_theorem (a b : ℝ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (h3 : ∃ x_0 : ℝ, a * x_0^2 + 2 * x_0 + b = 0) :
  (∀ a b : ℝ, 2 * a^2 + b^2 ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, 2 * a^2 + b^2 = 2 * Real.sqrt 2) := by
sorry

end min_value_theorem_l2414_241442


namespace sparklers_to_crackers_value_comparison_l2414_241478

-- Define the exchange rates
def ornament_to_cracker : ℚ := 2
def sparkler_to_garland : ℚ := 2/5
def ornament_to_garland : ℚ := 1/4

-- Define the conversion function
def convert (item : String) (quantity : ℚ) : ℚ :=
  match item with
  | "sparkler" => quantity * sparkler_to_garland * (1 / ornament_to_garland) * ornament_to_cracker
  | "ornament" => quantity * ornament_to_cracker
  | _ => 0

-- Theorem 1: 10 sparklers are equivalent to 32 crackers
theorem sparklers_to_crackers :
  convert "sparkler" 10 = 32 :=
sorry

-- Theorem 2: 5 Christmas ornaments and 1 cracker are more valuable than 2 sparklers
theorem value_comparison :
  convert "ornament" 5 + 1 > convert "sparkler" 2 :=
sorry

end sparklers_to_crackers_value_comparison_l2414_241478


namespace expression_evaluation_l2414_241473

theorem expression_evaluation : 200 * (200 - 3) + (200^2 - 8^2) = 79336 := by
  sorry

end expression_evaluation_l2414_241473


namespace distribution_methods_eq_240_l2414_241424

/-- The number of ways to distribute 5 volunteers into 4 groups and assign them to intersections -/
def distributionMethods : ℕ := 
  (Nat.choose 5 2) * (Nat.factorial 4)

/-- Theorem stating that the number of distribution methods is 240 -/
theorem distribution_methods_eq_240 : distributionMethods = 240 := by
  sorry

end distribution_methods_eq_240_l2414_241424


namespace geometric_sequence_common_ratio_sum_l2414_241461

theorem geometric_sequence_common_ratio_sum 
  (k p q : ℝ) 
  (h1 : p ≠ q) 
  (h2 : k ≠ 0) 
  (h3 : k * p^2 - k * q^2 = 5 * (k * p - k * q)) : 
  p + q = 5 := by
sorry

end geometric_sequence_common_ratio_sum_l2414_241461


namespace calculation_proof_l2414_241421

theorem calculation_proof : 17 * (17/18) + 35 * (35/36) = 50 + 1/12 := by
  sorry

end calculation_proof_l2414_241421


namespace fraction_not_simplifiable_l2414_241436

theorem fraction_not_simplifiable (n : ℕ) : ¬ ∃ (d : ℤ), d > 1 ∧ d ∣ (21 * n + 4) ∧ d ∣ (14 * n + 3) := by
  sorry

end fraction_not_simplifiable_l2414_241436


namespace unique_solution_factorial_equation_l2414_241434

theorem unique_solution_factorial_equation :
  ∃! (n : ℕ), n > 0 ∧ (n + 2).factorial - (n + 1).factorial - n.factorial = n^2 + n^4 :=
by
  sorry

end unique_solution_factorial_equation_l2414_241434


namespace time_after_elapsed_hours_l2414_241415

def hours_elapsed : ℕ := 2023
def starting_time : ℕ := 3
def clock_hours : ℕ := 12

theorem time_after_elapsed_hours :
  (starting_time + hours_elapsed) % clock_hours = 10 :=
by sorry

end time_after_elapsed_hours_l2414_241415
