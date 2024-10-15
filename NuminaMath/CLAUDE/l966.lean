import Mathlib

namespace NUMINAMATH_CALUDE_complex_condition_l966_96666

theorem complex_condition (a b : ℝ) (hb : b ≠ 0) :
  let z : ℂ := Complex.mk a b
  (z^2 - 4*b*z).im = 0 → a = 2*b := by
  sorry

end NUMINAMATH_CALUDE_complex_condition_l966_96666


namespace NUMINAMATH_CALUDE_actual_distance_scientific_notation_l966_96650

/-- The scale of the map -/
def map_scale : ℚ := 1 / 8000000

/-- The distance between A and B on the map in centimeters -/
def map_distance : ℚ := 3.5

/-- The actual distance between A and B in centimeters -/
def actual_distance : ℕ := 28000000

/-- Theorem stating that the actual distance is equal to 2.8 × 10^7 -/
theorem actual_distance_scientific_notation : 
  (actual_distance : ℝ) = 2.8 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_scientific_notation_l966_96650


namespace NUMINAMATH_CALUDE_four_roots_implies_a_in_open_interval_l966_96627

def f (x : ℝ) : ℝ := |x^2 + x - 2|

theorem four_roots_implies_a_in_open_interval (a : ℝ) :
  (∃ (w x y z : ℝ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f w - a * |w - 2| = 0 ∧
    f x - a * |x - 2| = 0 ∧
    f y - a * |y - 2| = 0 ∧
    f z - a * |z - 2| = 0 ∧
    (∀ t : ℝ, f t - a * |t - 2| = 0 → t = w ∨ t = x ∨ t = y ∨ t = z)) →
  0 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_four_roots_implies_a_in_open_interval_l966_96627


namespace NUMINAMATH_CALUDE_less_number_proof_l966_96664

theorem less_number_proof (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 96) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_less_number_proof_l966_96664


namespace NUMINAMATH_CALUDE_comic_book_collections_equal_l966_96629

/-- Kymbrea's initial comic book collection size -/
def kymbrea_initial : ℕ := 40

/-- Kymbrea's monthly comic book addition rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection size -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate -/
def lashawn_rate : ℕ := 5

/-- The number of months after which LaShawn's collection will be three times Kymbrea's -/
def months : ℕ := 25

theorem comic_book_collections_equal : 
  lashawn_initial + lashawn_rate * months = 3 * (kymbrea_initial + kymbrea_rate * months) := by
  sorry

end NUMINAMATH_CALUDE_comic_book_collections_equal_l966_96629


namespace NUMINAMATH_CALUDE_vehicle_speeds_l966_96632

/-- A structure representing a vehicle with its speed -/
structure Vehicle where
  speed : ℝ
  speed_pos : speed > 0

/-- The problem setup -/
def VehicleProblem (v₁ v₄ : ℝ) : Prop :=
  v₁ > 0 ∧ v₄ > 0 ∧ v₁ > v₄

/-- The theorem statement -/
theorem vehicle_speeds (v₁ v₄ : ℝ) (h : VehicleProblem v₁ v₄) :
  ∃ (v₂ v₃ : ℝ),
    v₂ = 3 * v₁ * v₄ / (2 * v₄ + v₁) ∧
    v₃ = 3 * v₁ * v₄ / (v₄ + 2 * v₁) ∧
    v₁ > v₂ ∧ v₂ > v₃ ∧ v₃ > v₄ :=
  sorry

end NUMINAMATH_CALUDE_vehicle_speeds_l966_96632


namespace NUMINAMATH_CALUDE_range_of_a_for_nonempty_solution_set_l966_96611

theorem range_of_a_for_nonempty_solution_set :
  (∃ (a : ℝ), ∃ (x : ℝ), |x + 2| + |x| ≤ a) →
  (∀ (a : ℝ), (∃ (x : ℝ), |x + 2| + |x| ≤ a) ↔ a ∈ Set.Ici 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_nonempty_solution_set_l966_96611


namespace NUMINAMATH_CALUDE_female_athletes_in_sample_l966_96601

/-- Calculates the number of female athletes in a stratified sample -/
def femaleAthletesSample (totalAthletes maleAthletes femaleAthletes sampleSize : ℕ) : ℕ :=
  (femaleAthletes * sampleSize) / totalAthletes

/-- Theorem stating the number of female athletes in the sample -/
theorem female_athletes_in_sample :
  femaleAthletesSample 84 48 36 21 = 9 := by
  sorry

#eval femaleAthletesSample 84 48 36 21

end NUMINAMATH_CALUDE_female_athletes_in_sample_l966_96601


namespace NUMINAMATH_CALUDE_square_sum_of_special_integers_l966_96679

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 47)
  (h2 : x^2 * y + x * y^2 = 506) : 
  x^2 + y^2 = 101 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_special_integers_l966_96679


namespace NUMINAMATH_CALUDE_men_count_in_alternating_arrangement_l966_96614

/-- Represents the number of arrangements for a given number of men and women -/
def arrangements (men : ℕ) (women : ℕ) : ℕ := sorry

/-- Represents whether men and women are alternating in an arrangement -/
def isAlternating (men : ℕ) (women : ℕ) : Prop := sorry

theorem men_count_in_alternating_arrangement :
  ∀ (men : ℕ),
  (women : ℕ) → women = 2 →
  isAlternating men women →
  arrangements men women = 12 →
  men = 4 := by sorry

end NUMINAMATH_CALUDE_men_count_in_alternating_arrangement_l966_96614


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l966_96693

theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 70 → 
  G = 110 → 
  H = I → 
  J = 2 * H + 25 → 
  F + G + H + I + J = 540 → 
  J = 192.5 ∧ J = max F (max G (max H (max I J))) := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l966_96693


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l966_96674

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 / 9 →
  ((4 / 3) * π * r₁^3) / ((4 / 3) * π * r₂^3) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l966_96674


namespace NUMINAMATH_CALUDE_sqrt_nine_is_rational_l966_96688

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = (p : ℝ) / (q : ℝ)

-- State the theorem
theorem sqrt_nine_is_rational : IsRational (Real.sqrt 9) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_is_rational_l966_96688


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l966_96636

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ 
  m < -2 ∨ m > 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l966_96636


namespace NUMINAMATH_CALUDE_harkamal_fruit_purchase_l966_96699

/-- Calculates the final amount paid after discount for a fruit purchase --/
def final_amount_paid (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
  (oranges_kg : ℝ) (oranges_price : ℝ) (bananas_kg : ℝ) (bananas_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let total_cost := grapes_kg * grapes_price + mangoes_kg * mangoes_price +
                    oranges_kg * oranges_price + bananas_kg * bananas_price
  let discount := discount_rate * total_cost
  total_cost - discount

/-- Theorem stating the final amount paid for Harkamal's fruit purchase --/
theorem harkamal_fruit_purchase :
  final_amount_paid 3 70 9 55 5 40 7 20 0.1 = 940.5 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_fruit_purchase_l966_96699


namespace NUMINAMATH_CALUDE_exists_counterexample_l966_96696

/-- A function is strictly monotonically increasing -/
def StrictlyIncreasing (f : ℚ → ℚ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The range of a function is the entire set of rationals -/
def SurjectiveOnRationals (f : ℚ → ℚ) : Prop :=
  ∀ y, ∃ x, f x = y

/-- The main theorem -/
theorem exists_counterexample : ∃ (f g : ℚ → ℚ),
  StrictlyIncreasing f ∧ StrictlyIncreasing g ∧
  SurjectiveOnRationals f ∧ SurjectiveOnRationals g ∧
  ¬SurjectiveOnRationals (λ x => f x + g x) := by
  sorry

end NUMINAMATH_CALUDE_exists_counterexample_l966_96696


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l966_96643

open Set Real

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 3^x + 1}

-- Define set B
def B : Set ℝ := {x | log x < 0}

-- Statement to prove
theorem complement_A_inter_B : 
  (U \ A) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l966_96643


namespace NUMINAMATH_CALUDE_sqrt_f_squared_2009_l966_96649

-- Define the function f with the given property
axiom f : ℝ → ℝ
axiom f_property : ∀ a b : ℝ, f (a * f b) = a * b

-- State the theorem to be proved
theorem sqrt_f_squared_2009 : Real.sqrt (f 2009 ^ 2) = 2009 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_f_squared_2009_l966_96649


namespace NUMINAMATH_CALUDE_square_arrangement_exists_l966_96690

/-- A square in the plane --/
structure Square where
  sideLength : ℕ
  position : ℝ × ℝ

/-- An arrangement of squares --/
def Arrangement (n : ℕ) := Fin n → Square

/-- Two squares touch if they share a vertex --/
def touches (s1 s2 : Square) : Prop := sorry

/-- An arrangement is valid if no two squares overlap --/
def validArrangement (arr : Arrangement n) : Prop := sorry

/-- An arrangement satisfies the touching condition if each square touches exactly two others --/
def satisfiesTouchingCondition (arr : Arrangement n) : Prop := sorry

/-- Main theorem: For n ≥ 5, there exists a valid arrangement where each square touches exactly two others --/
theorem square_arrangement_exists (n : ℕ) (h : n ≥ 5) :
  ∃ (arr : Arrangement n), validArrangement arr ∧ satisfiesTouchingCondition arr := by
  sorry

end NUMINAMATH_CALUDE_square_arrangement_exists_l966_96690


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_for_x_squared_lt_4_l966_96685

theorem necessary_sufficient_condition_for_x_squared_lt_4 :
  ∀ x : ℝ, x^2 < 4 ↔ -2 ≤ x ∧ x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_for_x_squared_lt_4_l966_96685


namespace NUMINAMATH_CALUDE_expression_value_l966_96652

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = -35 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l966_96652


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l966_96651

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 350) (h_goat : goat_value = 240) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∀ (d : ℕ), d > 0 → (∃ (p g : ℤ), d = pig_value * p + goat_value * g) → d ≥ debt) ∧
  (∃ (p g : ℤ), debt = pig_value * p + goat_value * g) :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l966_96651


namespace NUMINAMATH_CALUDE_chocolate_squares_difference_l966_96640

theorem chocolate_squares_difference (mike_squares jenny_squares : ℕ) 
  (h1 : mike_squares = 20) 
  (h2 : jenny_squares = 65) : 
  jenny_squares - 3 * mike_squares = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_squares_difference_l966_96640


namespace NUMINAMATH_CALUDE_arithmetic_sequences_equal_sum_l966_96624

/-- Sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (a d n : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequences_equal_sum :
  ∃! (n : ℕ), n > 0 ∧ arithmetic_sum 5 4 n = arithmetic_sum 12 3 n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_equal_sum_l966_96624


namespace NUMINAMATH_CALUDE_biking_jogging_swimming_rates_l966_96692

theorem biking_jogging_swimming_rates : 
  ∃! (b j s : ℕ+), 
    (3 * b.val + 2 * j.val + 4 * s.val = 80) ∧ 
    (4 * b.val + 3 * j.val + 2 * s.val = 98) ∧ 
    (b.val^2 + j.val^2 + s.val^2 = 536) := by
  sorry

end NUMINAMATH_CALUDE_biking_jogging_swimming_rates_l966_96692


namespace NUMINAMATH_CALUDE_largest_value_l966_96678

def expr_a : ℝ := 3 - 1 + 4 + 6
def expr_b : ℝ := 3 - 1 * 4 + 6
def expr_c : ℝ := 3 - (1 + 4) * 6
def expr_d : ℝ := 3 - 1 + 4 * 6
def expr_e : ℝ := 3 * (1 - 4) + 6

theorem largest_value :
  expr_d = 26 ∧
  expr_d > expr_a ∧
  expr_d > expr_b ∧
  expr_d > expr_c ∧
  expr_d > expr_e :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l966_96678


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l966_96618

/-- The surface area of a sphere circumscribing a right circular cone -/
theorem circumscribed_sphere_surface_area (h : ℝ) (s : ℝ) (π : ℝ) : 
  h = 3 → s = 2 → π = Real.pi → 
  (4 * π * ((s^2 * 3 / 9) + (h^2 / 4))) = (43 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l966_96618


namespace NUMINAMATH_CALUDE_quadratic_factorization_l966_96616

theorem quadratic_factorization (m : ℝ) : m^2 - 2*m + 1 = (m - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l966_96616


namespace NUMINAMATH_CALUDE_split_cost_12_cupcakes_at_1_50_l966_96609

/-- The amount each person pays when two people buy cupcakes and split the cost evenly -/
def split_cost (num_cupcakes : ℕ) (price_per_cupcake : ℚ) : ℚ :=
  (num_cupcakes : ℚ) * price_per_cupcake / 2

/-- Theorem: When two people buy 12 cupcakes at $1.50 each and split the cost evenly, each person pays $9.00 -/
theorem split_cost_12_cupcakes_at_1_50 :
  split_cost 12 (3/2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_split_cost_12_cupcakes_at_1_50_l966_96609


namespace NUMINAMATH_CALUDE_peters_remaining_money_l966_96660

/-- Calculates Peter's remaining money after shopping at the market. -/
def remaining_money (initial_amount : ℕ) (potato_kg : ℕ) (potato_price : ℕ) 
  (tomato_kg : ℕ) (tomato_price : ℕ) (cucumber_kg : ℕ) (cucumber_price : ℕ) 
  (banana_kg : ℕ) (banana_price : ℕ) : ℕ :=
  initial_amount - (potato_kg * potato_price + tomato_kg * tomato_price + 
    cucumber_kg * cucumber_price + banana_kg * banana_price)

/-- Proves that Peter's remaining money after shopping is $426. -/
theorem peters_remaining_money : 
  remaining_money 500 6 2 9 3 5 4 3 5 = 426 := by
  sorry

end NUMINAMATH_CALUDE_peters_remaining_money_l966_96660


namespace NUMINAMATH_CALUDE_new_person_weight_l966_96687

/-- Given a group of 5 people where one person weighing 40 kg is replaced,
    resulting in an average weight increase of 10 kg, prove that the new
    person weighs 90 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_replaced : Real) 
  (avg_increase : Real) (new_weight : Real) : 
  initial_count = 5 → 
  weight_replaced = 40 → 
  avg_increase = 10 → 
  new_weight = weight_replaced + (initial_count * avg_increase) → 
  new_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l966_96687


namespace NUMINAMATH_CALUDE_intersection_sum_l966_96604

theorem intersection_sum (n c : ℝ) : 
  (∀ x y : ℝ, y = n * x + 3 → y = 4 * x + c → x = 4 ∧ y = 7) → 
  c + n = -8 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l966_96604


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l966_96631

/-- Given a triangle ABC with angle ratio ∠A:∠B:∠C = 3:4:5, it cannot be concluded that ABC is a right triangle. -/
theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / (A + B + C) = 3 / 12 ∧ B / (A + B + C) = 4 / 12 ∧ C / (A + B + C) = 5 / 12) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l966_96631


namespace NUMINAMATH_CALUDE_inequality_proof_l966_96668

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l966_96668


namespace NUMINAMATH_CALUDE_beadshop_profit_l966_96691

theorem beadshop_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ)
  (h_total : total_profit = 1200)
  (h_monday : monday_fraction = 1/3)
  (h_tuesday : tuesday_fraction = 1/4) :
  total_profit - (monday_fraction * total_profit + tuesday_fraction * total_profit) = 500 := by
  sorry

end NUMINAMATH_CALUDE_beadshop_profit_l966_96691


namespace NUMINAMATH_CALUDE_toby_change_is_seven_l966_96670

/-- Represents the cost of a meal for two people -/
structure MealCost where
  cheeseburger_price : ℚ
  milkshake_price : ℚ
  coke_price : ℚ
  fries_price : ℚ
  cookie_price : ℚ
  cookie_quantity : ℕ
  tax : ℚ

/-- Calculates the change Toby brings home after splitting the bill -/
def toby_change (meal : MealCost) (toby_initial_amount : ℚ) : ℚ :=
  let total_cost := 2 * meal.cheeseburger_price + meal.milkshake_price + meal.coke_price +
                    meal.fries_price + meal.cookie_price * meal.cookie_quantity + meal.tax
  let toby_share := total_cost / 2
  toby_initial_amount - toby_share

/-- Theorem stating that Toby's change is $7 given the specific meal costs -/
theorem toby_change_is_seven :
  let meal := MealCost.mk 3.65 2 1 4 0.5 3 0.2
  toby_change meal 15 = 7 := by sorry


end NUMINAMATH_CALUDE_toby_change_is_seven_l966_96670


namespace NUMINAMATH_CALUDE_oysters_with_pearls_percentage_l966_96608

/-- The percentage of oysters with pearls, given the number of oysters collected per dive,
    the number of dives, and the total number of pearls collected. -/
def percentage_oysters_with_pearls (oysters_per_dive : ℕ) (num_dives : ℕ) (total_pearls : ℕ) : ℚ :=
  (total_pearls : ℚ) / ((oysters_per_dive * num_dives) : ℚ) * 100

/-- Theorem stating that the percentage of oysters with pearls is 25%,
    given the specific conditions from the problem. -/
theorem oysters_with_pearls_percentage :
  percentage_oysters_with_pearls 16 14 56 = 25 := by
  sorry


end NUMINAMATH_CALUDE_oysters_with_pearls_percentage_l966_96608


namespace NUMINAMATH_CALUDE_first_robber_guarantee_l966_96689

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : ℕ
  maxBags : ℕ

/-- Represents the outcome of the game for the first robber --/
def FirstRobberOutcome (game : CoinGame) : ℕ := 
  min game.totalCoins (game.totalCoins - (game.maxBags - 1) * (game.totalCoins / (2 * game.maxBags - 1)))

/-- Theorem stating the guaranteed minimum coins for the first robber --/
theorem first_robber_guarantee (game : CoinGame) 
  (h1 : game.totalCoins = 300) 
  (h2 : game.maxBags = 11) : 
  FirstRobberOutcome game ≥ 146 := by
  sorry

#eval FirstRobberOutcome { totalCoins := 300, maxBags := 11 }

end NUMINAMATH_CALUDE_first_robber_guarantee_l966_96689


namespace NUMINAMATH_CALUDE_power_of_five_l966_96626

theorem power_of_five (x : ℕ) : 121 * (5^x) = 75625 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l966_96626


namespace NUMINAMATH_CALUDE_area_of_triangle_FQH_area_of_triangle_FQH_proof_l966_96620

-- Define the rectangle EFGH
structure Rectangle where
  EF : ℝ
  EH : ℝ

-- Define the trapezoid PRHG
structure Trapezoid where
  EP : ℝ
  area : ℝ

-- Define the problem setup
def problem (rect : Rectangle) (trap : Trapezoid) : Prop :=
  rect.EF = 16 ∧ 
  trap.EP = 8 ∧
  trap.area = 160

-- Theorem statement
theorem area_of_triangle_FQH (rect : Rectangle) (trap : Trapezoid) 
  (h : problem rect trap) : ℝ :=
  80

-- Proof
theorem area_of_triangle_FQH_proof (rect : Rectangle) (trap : Trapezoid) 
  (h : problem rect trap) : area_of_triangle_FQH rect trap h = 80 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_FQH_area_of_triangle_FQH_proof_l966_96620


namespace NUMINAMATH_CALUDE_larry_lost_stickers_l966_96667

/-- Given that Larry starts with 93 stickers and ends up with 87 stickers,
    prove that he lost 6 stickers. -/
theorem larry_lost_stickers (initial : ℕ) (final : ℕ) (h1 : initial = 93) (h2 : final = 87) :
  initial - final = 6 := by
  sorry

end NUMINAMATH_CALUDE_larry_lost_stickers_l966_96667


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_range_l966_96605

/-- The function f(x) = x^3 - 3x has a minimum value on the interval (a, 6-a^2) -/
def has_minimum_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a (6 - a^2), ∀ y ∈ Set.Ioo a (6 - a^2), f x ≤ f y

/-- The main theorem -/
theorem minimum_value_implies_a_range (a : ℝ) :
  has_minimum_on_interval (fun x => x^3 - 3*x) a → a ∈ Set.Icc (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_range_l966_96605


namespace NUMINAMATH_CALUDE_long_jump_competition_l966_96694

theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second > first →
  third = second - 2 →
  fourth = third + 3 →
  fourth = 24 →
  second - first = 1 := by
sorry

end NUMINAMATH_CALUDE_long_jump_competition_l966_96694


namespace NUMINAMATH_CALUDE_shorts_savings_l966_96615

/-- Calculates the savings when buying shorts with a discount compared to buying individually -/
def savings (price : ℝ) (quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_cost := price * quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  total_cost - discounted_cost

/-- Proves that the savings when buying 3 pairs of shorts at $10 each with a 10% discount is $3 -/
theorem shorts_savings : savings 10 3 0.1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_shorts_savings_l966_96615


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l966_96676

theorem x_value_from_fraction_equality (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y + 3) / (y^2 + 2*y + 2) →
  x = y^2 + 2*y + 3 := by
sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l966_96676


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l966_96622

/-- The decimal representation of 0.456̄ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 1216 / 333 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l966_96622


namespace NUMINAMATH_CALUDE_max_cable_connections_l966_96645

/-- Represents the number of computers of brand A -/
def brand_a_count : Nat := 28

/-- Represents the number of computers of brand B -/
def brand_b_count : Nat := 12

/-- Represents the minimum number of connections required per computer -/
def min_connections : Nat := 2

/-- Theorem stating the maximum number of distinct cable connections -/
theorem max_cable_connections :
  brand_a_count * brand_b_count = 336 ∧
  brand_a_count * brand_b_count ≥ brand_a_count * min_connections ∧
  brand_a_count * brand_b_count ≥ brand_b_count * min_connections :=
sorry

end NUMINAMATH_CALUDE_max_cable_connections_l966_96645


namespace NUMINAMATH_CALUDE_percent_difference_l966_96613

theorem percent_difference (x y p : ℝ) (h : x = y * (1 + p / 100)) : 
  p = 100 * ((x - y) / y) := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_l966_96613


namespace NUMINAMATH_CALUDE_tangent_lines_to_unit_circle_l966_96683

/-- The equation of a circle with radius 1 centered at the origin -/
def unitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A line is tangent to the unit circle at a given point -/
def isTangent (l : Line) (p : Point) : Prop :=
  unitCircle p.x p.y ∧
  l.a * p.x + l.b * p.y + l.c = 0 ∧
  ∀ (x y : ℝ), unitCircle x y → (l.a * x + l.b * y + l.c = 0 → x = p.x ∧ y = p.y)

theorem tangent_lines_to_unit_circle :
  let p1 : Point := ⟨-1, 0⟩
  let p2 : Point := ⟨-1, 2⟩
  let l1 : Line := ⟨1, 0, 1⟩  -- Represents x = -1
  let l2 : Line := ⟨3, 4, -5⟩  -- Represents 3x + 4y - 5 = 0
  isTangent l1 p1 ∧ isTangent l2 p2 := by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_unit_circle_l966_96683


namespace NUMINAMATH_CALUDE_fraction_multiplication_result_l966_96673

theorem fraction_multiplication_result : 
  (5 / 8 : ℚ) * (7 / 12 : ℚ) * (3 / 7 : ℚ) * 1350 = 210.9375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_result_l966_96673


namespace NUMINAMATH_CALUDE_cricket_team_size_l966_96671

theorem cricket_team_size :
  ∀ (n : ℕ),
  let captain_age : ℕ := 24
  let wicket_keeper_age : ℕ := captain_age + 3
  let team_average_age : ℕ := 21
  let remaining_players_average_age : ℕ := team_average_age - 1
  (n : ℝ) * team_average_age = 
    (n - 2 : ℝ) * remaining_players_average_age + captain_age + wicket_keeper_age →
  n = 11 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_size_l966_96671


namespace NUMINAMATH_CALUDE_oliver_presentation_appropriate_l966_96644

/-- Represents a presentation with a given word count. -/
structure Presentation where
  word_count : ℕ

/-- Checks if a presentation is appropriate given the speaking rate and time constraints. -/
def is_appropriate_presentation (p : Presentation) (speaking_rate : ℕ) (min_time : ℕ) (max_time : ℕ) : Prop :=
  let min_words := speaking_rate * min_time
  let max_words := speaking_rate * max_time
  min_words ≤ p.word_count ∧ p.word_count ≤ max_words

theorem oliver_presentation_appropriate :
  let speaking_rate := 120
  let min_time := 40
  let max_time := 55
  let presentation1 := Presentation.mk 5000
  let presentation2 := Presentation.mk 6200
  is_appropriate_presentation presentation1 speaking_rate min_time max_time ∧
  is_appropriate_presentation presentation2 speaking_rate min_time max_time :=
by sorry

end NUMINAMATH_CALUDE_oliver_presentation_appropriate_l966_96644


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l966_96648

theorem rectangular_garden_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 768 → 
  width = 16 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l966_96648


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l966_96698

theorem cubic_sum_problem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l966_96698


namespace NUMINAMATH_CALUDE_oil_drop_probability_l966_96658

theorem oil_drop_probability (circle_diameter : ℝ) (square_side : ℝ) 
  (h1 : circle_diameter = 3) 
  (h2 : square_side = 1) : 
  (square_side ^ 2) / (π * (circle_diameter / 2) ^ 2) = 4 / (9 * π) :=
sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l966_96658


namespace NUMINAMATH_CALUDE_water_saved_in_june_john_water_savings_l966_96655

/-- Calculates the water saved in June by replacing an inefficient toilet with a more efficient one. -/
theorem water_saved_in_june (old_toilet_usage : ℝ) (flushes_per_day : ℕ) (water_reduction_percentage : ℝ) (days_in_june : ℕ) : ℝ :=
  let new_toilet_usage := old_toilet_usage * (1 - water_reduction_percentage)
  let daily_old_usage := old_toilet_usage * flushes_per_day
  let daily_new_usage := new_toilet_usage * flushes_per_day
  let june_old_usage := daily_old_usage * days_in_june
  let june_new_usage := daily_new_usage * days_in_june
  june_old_usage - june_new_usage

/-- Proves that John saved 1800 gallons of water in June by replacing his old toilet. -/
theorem john_water_savings : water_saved_in_june 5 15 0.8 30 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_water_saved_in_june_john_water_savings_l966_96655


namespace NUMINAMATH_CALUDE_sin_225_degrees_l966_96665

theorem sin_225_degrees :
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l966_96665


namespace NUMINAMATH_CALUDE_text_message_plan_cost_l966_96647

/-- The cost per text message for the first plan -/
def cost_per_text_plan1 : ℚ := 1/4

/-- The monthly fee for the first plan -/
def monthly_fee_plan1 : ℚ := 9

/-- The cost per text message for the second plan -/
def cost_per_text_plan2 : ℚ := 2/5

/-- The number of text messages for which both plans cost the same -/
def equal_cost_messages : ℕ := 60

theorem text_message_plan_cost : 
  monthly_fee_plan1 + equal_cost_messages * cost_per_text_plan1 = 
  equal_cost_messages * cost_per_text_plan2 :=
by sorry

end NUMINAMATH_CALUDE_text_message_plan_cost_l966_96647


namespace NUMINAMATH_CALUDE_apple_students_count_l966_96662

/-- Represents the total number of degrees in a circle -/
def total_degrees : ℕ := 360

/-- Represents the number of degrees in a right angle -/
def right_angle : ℕ := 90

/-- Represents the number of students who chose bananas -/
def banana_students : ℕ := 168

/-- Calculates the number of students who chose apples given the conditions -/
def apple_students : ℕ :=
  (right_angle * (banana_students * 4 / 3)) / total_degrees

theorem apple_students_count : apple_students = 56 := by
  sorry

end NUMINAMATH_CALUDE_apple_students_count_l966_96662


namespace NUMINAMATH_CALUDE_expression_value_l966_96607

theorem expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l966_96607


namespace NUMINAMATH_CALUDE_bridget_apples_bridget_apples_proof_l966_96657

theorem bridget_apples : ℕ → Prop :=
  fun total_apples =>
    ∃ (ann_apples cassie_apples : ℕ),
      -- Bridget gave 4 apples to Tom
      -- She split the remaining apples equally between Ann and Cassie
      ann_apples = cassie_apples ∧
      -- After distribution, she was left with 5 apples
      total_apples = 4 + ann_apples + cassie_apples + 5 ∧
      -- The total number of apples is 13
      total_apples = 13

theorem bridget_apples_proof : bridget_apples 13 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_bridget_apples_proof_l966_96657


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l966_96681

-- Define a structure for the circle arrangement
structure CircleArrangement where
  numbers : List ℕ
  connections : List (ℕ × ℕ)

-- Define the property of valid ratios for connected circles
def validConnectedRatio (a b : ℕ) : Prop :=
  a / b = 3 ∨ a / b = 9 ∨ b / a = 3 ∨ b / a = 9

-- Define the property of invalid ratios for unconnected circles
def invalidUnconnectedRatio (a b : ℕ) : Prop :=
  a / b ≠ 3 ∧ a / b ≠ 9 ∧ b / a ≠ 3 ∧ b / a ≠ 9

-- Define the property of a valid circle arrangement
def validArrangement (arr : CircleArrangement) : Prop :=
  (∀ (a b : ℕ), (a, b) ∈ arr.connections → validConnectedRatio a b) ∧
  (∀ (a b : ℕ), a ∈ arr.numbers ∧ b ∈ arr.numbers ∧ (a, b) ∉ arr.connections → invalidUnconnectedRatio a b)

-- Theorem stating the existence of a valid arrangement
theorem exists_valid_arrangement : ∃ (arr : CircleArrangement), validArrangement arr :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l966_96681


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_three_l966_96653

theorem arithmetic_sqrt_of_three (x : ℝ) : x = Real.sqrt 3 ↔ x ≥ 0 ∧ x ^ 2 = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_three_l966_96653


namespace NUMINAMATH_CALUDE_y_divisibility_l966_96646

def y : ℕ := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem y_divisibility : 
  (∃ k : ℕ, y = 5 * k) ∧ 
  (∃ k : ℕ, y = 10 * k) ∧ 
  (∃ k : ℕ, y = 20 * k) ∧ 
  (∃ k : ℕ, y = 40 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l966_96646


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l966_96610

/-- Given two parallel vectors a and b in R², prove that the magnitude of b is 2√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  b.1 = -2 → 
  (a.1 * b.2 = a.2 * b.1) → 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l966_96610


namespace NUMINAMATH_CALUDE_intersection_A_B_l966_96697

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {2, 3}

theorem intersection_A_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l966_96697


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l966_96638

def I : Set Int := {x | -3 < x ∧ x < 3}
def A : Set Int := {1, 2}
def B : Set Int := {-2, -1, 2}

theorem union_complement_equals_set : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l966_96638


namespace NUMINAMATH_CALUDE_lowest_possible_score_l966_96630

def total_tests : ℕ := 6
def max_score : ℕ := 200
def target_average : ℕ := 170

def first_four_scores : List ℕ := [150, 180, 175, 160]

theorem lowest_possible_score :
  ∃ (score1 score2 : ℕ),
    score1 ≤ max_score ∧ 
    score2 ≤ max_score ∧
    (List.sum first_four_scores + score1 + score2) / total_tests = target_average ∧
    (∀ (s1 s2 : ℕ), 
      s1 ≤ max_score → 
      s2 ≤ max_score → 
      (List.sum first_four_scores + s1 + s2) / total_tests = target_average → 
      min s1 s2 ≥ min score1 score2) ∧
    min score1 score2 = 155 :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l966_96630


namespace NUMINAMATH_CALUDE_table_cost_l966_96625

/-- The cost of furniture items and payment details --/
structure FurniturePurchase where
  couch_cost : ℕ
  lamp_cost : ℕ
  initial_payment : ℕ
  remaining_balance : ℕ

/-- Theorem stating the cost of the table --/
theorem table_cost (purchase : FurniturePurchase)
  (h1 : purchase.couch_cost = 750)
  (h2 : purchase.lamp_cost = 50)
  (h3 : purchase.initial_payment = 500)
  (h4 : purchase.remaining_balance = 400) :
  ∃ (table_cost : ℕ), 
    purchase.couch_cost + table_cost + purchase.lamp_cost - purchase.initial_payment = purchase.remaining_balance ∧
    table_cost = 100 :=
sorry

end NUMINAMATH_CALUDE_table_cost_l966_96625


namespace NUMINAMATH_CALUDE_sams_football_games_l966_96675

/-- Given that Sam went to 14 football games this year and 43 games in total,
    prove that he went to 29 games last year. -/
theorem sams_football_games (games_this_year games_total : ℕ) 
    (h1 : games_this_year = 14)
    (h2 : games_total = 43) :
    games_total - games_this_year = 29 := by
  sorry

end NUMINAMATH_CALUDE_sams_football_games_l966_96675


namespace NUMINAMATH_CALUDE_megan_cupcakes_per_package_l966_96663

/-- Calculates the number of cupcakes per package given the initial number of cupcakes,
    the number of cupcakes eaten, and the number of packages. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Proves that given 68 initial cupcakes, 32 cupcakes eaten, and 6 packages,
    the number of cupcakes in each package is 6. -/
theorem megan_cupcakes_per_package :
  cupcakes_per_package 68 32 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_megan_cupcakes_per_package_l966_96663


namespace NUMINAMATH_CALUDE_oliver_william_money_difference_l966_96656

/-- Calculates the total amount of money given the number of bills of different denominations -/
def calculate_total (twenty_bills ten_bills five_bills : ℕ) : ℕ :=
  20 * twenty_bills + 10 * ten_bills + 5 * five_bills

/-- Represents the problem of comparing Oliver's and William's money -/
theorem oliver_william_money_difference :
  let oliver_total := calculate_total 10 0 3
  let william_total := calculate_total 0 15 4
  oliver_total - william_total = 45 := by sorry

end NUMINAMATH_CALUDE_oliver_william_money_difference_l966_96656


namespace NUMINAMATH_CALUDE_waiter_customers_l966_96639

/-- Calculates the number of customers a waiter has after some tables leave --/
def customers_remaining (initial_tables : Float) (tables_left : Float) (customers_per_table : Float) : Float :=
  (initial_tables - tables_left) * customers_per_table

/-- Theorem: Given the initial conditions, the waiter has 256.0 customers --/
theorem waiter_customers :
  customers_remaining 44.0 12.0 8.0 = 256.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l966_96639


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l966_96677

theorem salary_reduction_percentage (x : ℝ) : 
  (100 - x + (100 - x) * (11.11111111111111 / 100) = 100) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l966_96677


namespace NUMINAMATH_CALUDE_gavin_dreams_total_l966_96617

/-- The number of dreams Gavin has per day this year -/
def dreams_per_day : ℕ := 4

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- The number of dreams Gavin had this year -/
def dreams_this_year : ℕ := dreams_per_day * days_per_year

/-- The number of dreams Gavin had last year -/
def dreams_last_year : ℕ := 2 * dreams_this_year

/-- The total number of dreams Gavin had in two years -/
def total_dreams : ℕ := dreams_this_year + dreams_last_year

theorem gavin_dreams_total : total_dreams = 4380 := by
  sorry

end NUMINAMATH_CALUDE_gavin_dreams_total_l966_96617


namespace NUMINAMATH_CALUDE_light_glow_interval_l966_96628

def seconds_past_midnight (hours minutes seconds : ℕ) : ℕ :=
  hours * 3600 + minutes * 60 + seconds

def start_time : ℕ := seconds_past_midnight 1 57 58
def end_time : ℕ := seconds_past_midnight 3 20 47
def num_glows : ℝ := 354.92857142857144

theorem light_glow_interval :
  let total_time : ℕ := end_time - start_time
  let interval : ℝ := (total_time : ℝ) / num_glows
  ⌊interval⌋ = 14 := by sorry

end NUMINAMATH_CALUDE_light_glow_interval_l966_96628


namespace NUMINAMATH_CALUDE_min_abs_plus_2023_min_value_abs_plus_2023_l966_96654

theorem min_abs_plus_2023 (a : ℚ) : 
  (|a| + 2023 : ℚ) ≥ 2023 := by sorry

theorem min_value_abs_plus_2023 : 
  ∃ (m : ℚ), ∀ (a : ℚ), (|a| + 2023 : ℚ) ≥ m ∧ ∃ (b : ℚ), (|b| + 2023 : ℚ) = m := by
  use 2023
  sorry

end NUMINAMATH_CALUDE_min_abs_plus_2023_min_value_abs_plus_2023_l966_96654


namespace NUMINAMATH_CALUDE_vacant_seats_l966_96619

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 75 / 100) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l966_96619


namespace NUMINAMATH_CALUDE_problem_statement_l966_96695

theorem problem_statement (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3) : 
  (x - z) * (y - w) / ((x - y) * (z - w)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l966_96695


namespace NUMINAMATH_CALUDE_xyz_sum_l966_96633

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 147)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 163) :
  x*y + y*z + x*z = 56 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l966_96633


namespace NUMINAMATH_CALUDE_parabola_intersection_slope_l966_96635

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Theorem: For a parabola y² = 2px with p > 0, if a line through M(-p/2, 0) with slope k
    intersects the parabola at A(x₀, y₀) such that |AM| = 5/4 * |AF|, then k = ±3/4 -/
theorem parabola_intersection_slope (C : Parabola) (A : ParabolaPoint C) (k : ℝ) :
  let M : ℝ × ℝ := (-C.p/2, 0)
  let F : ℝ × ℝ := (C.p/2, 0)
  let AM := Real.sqrt ((A.x + C.p/2)^2 + A.y^2)
  let AF := A.x + C.p/2
  (A.y - 0) / (A.x - (-C.p/2)) = k →
  AM = 5/4 * AF →
  k = 3/4 ∨ k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_slope_l966_96635


namespace NUMINAMATH_CALUDE_exists_solution_for_calendar_equation_l966_96659

theorem exists_solution_for_calendar_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_for_calendar_equation_l966_96659


namespace NUMINAMATH_CALUDE_single_elimination_tournament_matches_l966_96623

/-- Represents a single-elimination tournament. -/
structure Tournament where
  teams : ℕ
  matches_played : ℕ

/-- The number of teams eliminated in a single-elimination tournament. -/
def eliminated_teams (t : Tournament) : ℕ := t.matches_played

/-- A tournament is complete when there is only one team remaining. -/
def is_complete (t : Tournament) : Prop := t.teams - eliminated_teams t = 1

theorem single_elimination_tournament_matches (t : Tournament) :
  t.teams = 128 → is_complete t → t.matches_played = 127 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_matches_l966_96623


namespace NUMINAMATH_CALUDE_compound_proposition_truth_l966_96603

theorem compound_proposition_truth : 
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧ 
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x > x^3) := by
sorry

end NUMINAMATH_CALUDE_compound_proposition_truth_l966_96603


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l966_96612

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → 
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l966_96612


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l966_96682

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l966_96682


namespace NUMINAMATH_CALUDE_differential_of_y_l966_96621

open Real

noncomputable def y (x : ℝ) : ℝ := cos x * log (tan x) - log (tan (x / 2))

theorem differential_of_y (x : ℝ) (h : x ≠ 0) (h' : x ≠ π/2) :
  deriv y x = -sin x * log (tan x) :=
by sorry

end NUMINAMATH_CALUDE_differential_of_y_l966_96621


namespace NUMINAMATH_CALUDE_security_system_probability_l966_96672

theorem security_system_probability (p : ℝ) : 
  (1/8 : ℝ) * (1 - p) + (1 - 1/8) * p = 9/40 → p = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_security_system_probability_l966_96672


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_sum_10_l966_96642

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_first_year_after_2010_with_sum_10 (year : ℕ) : Prop :=
  year > 2010 ∧
  sum_of_digits year = 10 ∧
  ∀ y, 2010 < y ∧ y < year → sum_of_digits y ≠ 10

theorem first_year_after_2010_with_sum_10 :
  is_first_year_after_2010_with_sum_10 2017 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_sum_10_l966_96642


namespace NUMINAMATH_CALUDE_max_distance_product_l966_96606

/-- Fixed point A -/
def A : ℝ × ℝ := (0, 0)

/-- Fixed point B -/
def B : ℝ × ℝ := (1, 3)

/-- Line through A -/
def line_A (m : ℝ) (x y : ℝ) : Prop := x + m * y = 0

/-- Line through B -/
def line_B (m : ℝ) (x y : ℝ) : Prop := m * x - y - m + 3 = 0

/-- Intersection point P -/
def P (m : ℝ) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Product of distances PA and PB -/
def distance_product (m : ℝ) : ℝ := distance (P m) A * distance (P m) B

/-- Theorem: Maximum value of |PA| * |PB| is 5 -/
theorem max_distance_product : 
  ∃ (m : ℝ), ∀ (n : ℝ), distance_product n ≤ distance_product m ∧ distance_product m = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_product_l966_96606


namespace NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l966_96684

theorem max_a_for_quadratic_inequality :
  (∀ x : ℝ, x^2 - a*x + a ≥ 0) → a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l966_96684


namespace NUMINAMATH_CALUDE_ellipse_sum_l966_96637

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

theorem ellipse_sum (e : Ellipse) :
  e.h = 5 ∧ e.k = -3 ∧ e.a = 7 ∧ e.b = 4 →
  e.h + e.k + e.a + e.b = 13 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l966_96637


namespace NUMINAMATH_CALUDE_doughnuts_per_box_l966_96661

theorem doughnuts_per_box (total_doughnuts : ℕ) (num_boxes : ℕ) (doughnuts_per_box : ℕ) : 
  total_doughnuts = 48 → 
  num_boxes = 4 → 
  total_doughnuts = num_boxes * doughnuts_per_box →
  doughnuts_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_per_box_l966_96661


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_condition_l966_96669

theorem not_sufficient_not_necessary_condition (a b : ℝ) :
  ¬(∀ a b : ℝ, a + b > 0 → a * b > 0) ∧ ¬(∀ a b : ℝ, a * b > 0 → a + b > 0) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_condition_l966_96669


namespace NUMINAMATH_CALUDE_range_of_a_l966_96680

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 21 = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | 5*x - a ≥ 3*x + 2}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ≤ -8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l966_96680


namespace NUMINAMATH_CALUDE_three_number_ratio_problem_l966_96600

theorem three_number_ratio_problem (a b c : ℝ) 
  (h_sum : a + b + c = 120)
  (h_ratio1 : a / b = 3 / 4)
  (h_ratio2 : b / c = 3 / 5)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  b = 1440 / 41 := by
sorry

end NUMINAMATH_CALUDE_three_number_ratio_problem_l966_96600


namespace NUMINAMATH_CALUDE_block_stacks_height_difference_main_theorem_l966_96634

/-- Proves that the height difference between the final stack and the second stack is 7 blocks -/
theorem block_stacks_height_difference : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (first_stack : ℕ) (second_stack : ℕ) (final_stack : ℕ) (fallen_blocks : ℕ) (height_diff : ℕ) =>
    first_stack = 7 ∧
    second_stack = first_stack + 5 ∧
    final_stack = second_stack + height_diff ∧
    fallen_blocks = first_stack + (second_stack - 2) + (final_stack - 3) ∧
    fallen_blocks = 33 →
    height_diff = 7

/-- The main theorem stating that the height difference is 7 blocks -/
theorem main_theorem : ∃ (first_stack second_stack final_stack fallen_blocks : ℕ),
  block_stacks_height_difference first_stack second_stack final_stack fallen_blocks 7 :=
sorry

end NUMINAMATH_CALUDE_block_stacks_height_difference_main_theorem_l966_96634


namespace NUMINAMATH_CALUDE_cone_water_volume_ratio_l966_96602

theorem cone_water_volume_ratio (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_volume := (1 / 3) * Real.pi * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by sorry

end NUMINAMATH_CALUDE_cone_water_volume_ratio_l966_96602


namespace NUMINAMATH_CALUDE_rectangle_diagonal_problem_l966_96686

theorem rectangle_diagonal_problem (w l : ℝ) 
  (h1 : w^2 + l^2 = 400) 
  (h2 : 4*w^2 + l^2 = 484) : 
  w^2 = 28 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_problem_l966_96686


namespace NUMINAMATH_CALUDE_valid_K_values_l966_96641

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for K being a valid solution -/
def is_valid_K (K : ℕ) : Prop :=
  ∃ (N : ℕ), N < 50 ∧ triangular_sum K = N^2

theorem valid_K_values :
  {K : ℕ | is_valid_K K} = {1, 8, 49} := by sorry

end NUMINAMATH_CALUDE_valid_K_values_l966_96641
