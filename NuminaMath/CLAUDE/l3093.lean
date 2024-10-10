import Mathlib

namespace max_known_cards_l3093_309362

/-- A strategy for selecting cards and receiving information -/
structure CardStrategy where
  selectCards : Fin 2013 → Finset (Fin 2013)
  receiveNumber : Finset (Fin 2013) → Fin 2013

/-- The set of cards for which we know the numbers after applying a strategy -/
def knownCards (s : CardStrategy) : Finset (Fin 2013) :=
  sorry

/-- The theorem stating that 1986 is the maximum number of cards we can guarantee to know -/
theorem max_known_cards :
  (∃ (s : CardStrategy), (knownCards s).card = 1986) ∧
  (∀ (s : CardStrategy), (knownCards s).card ≤ 1986) :=
sorry

end max_known_cards_l3093_309362


namespace cloth_loss_problem_l3093_309334

/-- Calculates the loss per metre of cloth given the total quantity sold,
    total selling price, and cost price per metre. -/
def loss_per_metre (quantity : ℕ) (selling_price total_cost_price : ℚ) : ℚ :=
  (total_cost_price - selling_price) / quantity

theorem cloth_loss_problem (quantity : ℕ) (selling_price cost_price_per_metre : ℚ) 
  (h1 : quantity = 200)
  (h2 : selling_price = 12000)
  (h3 : cost_price_per_metre = 66) :
  loss_per_metre quantity selling_price (quantity * cost_price_per_metre) = 6 := by
sorry

end cloth_loss_problem_l3093_309334


namespace perfect_square_trinomial_l3093_309323

theorem perfect_square_trinomial (x y k : ℝ) : 
  (∃ a : ℝ, x^2 + k*x*y + 64*y^2 = a^2) → k = 16 ∨ k = -16 := by
  sorry

end perfect_square_trinomial_l3093_309323


namespace crescent_moon_division_l3093_309326

/-- The maximum number of parts a crescent moon can be divided into with n straight cuts -/
def max_parts (n : ℕ) : ℕ := (n^2 + 3*n) / 2 + 1

/-- The number of straight cuts used -/
def num_cuts : ℕ := 5

theorem crescent_moon_division :
  max_parts num_cuts = 21 :=
sorry

end crescent_moon_division_l3093_309326


namespace quadratic_roots_distinct_l3093_309333

theorem quadratic_roots_distinct (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a*x^2 + b*x + c = 0 ∧ discriminant > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end quadratic_roots_distinct_l3093_309333


namespace anns_age_l3093_309325

theorem anns_age (A B t T : ℕ) : 
  A + B = 44 →
  B = A - t →
  B - t = A - T →
  B - T = A / 2 →
  A = 24 :=
by sorry

end anns_age_l3093_309325


namespace complex_equation_solution_l3093_309365

theorem complex_equation_solution (z : ℂ) 
  (h : 18 * Complex.normSq z = 2 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 2) + 48) : 
  z + 12 / z = -3 := by
sorry

end complex_equation_solution_l3093_309365


namespace correct_systematic_sample_l3093_309306

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  totalItems : ℕ
  sampleSize : ℕ
  startingPoint : ℕ
  samplingInterval : ℕ

/-- Generates the sample numbers for a given systematic sampling scheme. -/
def generateSampleNumbers (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.startingPoint + i * s.samplingInterval)

/-- Theorem: The correct sample numbers for systematic sampling of 5 items from 50 products
    are 9, 19, 29, 39, 49. -/
theorem correct_systematic_sample :
  let s : SystematicSampling := {
    totalItems := 50,
    sampleSize := 5,
    startingPoint := 9,
    samplingInterval := 10
  }
  generateSampleNumbers s = [9, 19, 29, 39, 49] := by
  sorry


end correct_systematic_sample_l3093_309306


namespace rem_one_third_neg_three_fourths_l3093_309349

-- Definition of the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- Theorem statement
theorem rem_one_third_neg_three_fourths :
  rem (1/3) (-3/4) = -5/12 := by sorry

end rem_one_third_neg_three_fourths_l3093_309349


namespace commute_time_difference_l3093_309386

/-- Given a set of 5 numbers {x, y, 10, 11, 9} with a mean of 10 and variance of 2, prove that |x-y| = 4 -/
theorem commute_time_difference (x y : ℝ) 
  (mean_eq : (x + y + 10 + 11 + 9) / 5 = 10) 
  (variance_eq : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) : 
  |x - y| = 4 := by
sorry

end commute_time_difference_l3093_309386


namespace pyramidal_stack_logs_example_l3093_309341

/-- Calculates the total number of logs in a pyramidal stack. -/
def pyramidal_stack_logs (bottom_row : ℕ) (top_row : ℕ) (difference : ℕ) : ℕ :=
  let n := (bottom_row - top_row) / difference + 1
  n * (bottom_row + top_row) / 2

/-- Proves that the total number of logs in the given pyramidal stack is 60. -/
theorem pyramidal_stack_logs_example : pyramidal_stack_logs 15 5 2 = 60 := by
  sorry

end pyramidal_stack_logs_example_l3093_309341


namespace odd_square_mod_eight_l3093_309352

theorem odd_square_mod_eight (k : ℤ) : ∃ m : ℤ, (2 * k + 1)^2 = 8 * m + 1 := by
  sorry

end odd_square_mod_eight_l3093_309352


namespace sqrt_product_equality_l3093_309307

theorem sqrt_product_equality (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) = 30) :
  x = 1 / Real.sqrt 20 := by
sorry

end sqrt_product_equality_l3093_309307


namespace tree_height_differences_l3093_309355

def pine_height : ℚ := 15 + 1/4
def birch_height : ℚ := 20 + 1/2
def maple_height : ℚ := 18 + 3/4

theorem tree_height_differences :
  (birch_height - pine_height = 5 + 1/4) ∧
  (birch_height - maple_height = 1 + 3/4) := by
  sorry

end tree_height_differences_l3093_309355


namespace intersection_of_lines_l3093_309324

/-- Parametric equation of a line in 2D space -/
structure ParametricLine2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if a point lies on a parametric line -/
def pointOnLine (p : ℝ × ℝ) (l : ParametricLine2D) : Prop :=
  ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

theorem intersection_of_lines (line1 line2 : ParametricLine2D)
    (h1 : line1 = ParametricLine2D.mk (5, 1) (3, -2))
    (h2 : line2 = ParametricLine2D.mk (2, 8) (5, -3)) :
    ∃! p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 ∧ p = (-73, 53) := by
  sorry

#check intersection_of_lines

end intersection_of_lines_l3093_309324


namespace saree_price_calculation_l3093_309312

/-- Calculate the final price after applying multiple discounts and a tax increase --/
def finalPrice (initialPrice : ℝ) (discounts : List ℝ) (taxRate : ℝ) (finalDiscount : ℝ) : ℝ :=
  let priceAfterDiscounts := discounts.foldl (fun price discount => price * (1 - discount)) initialPrice
  let priceAfterTax := priceAfterDiscounts * (1 + taxRate)
  priceAfterTax * (1 - finalDiscount)

/-- The theorem stating the final price of the sarees --/
theorem saree_price_calculation :
  let initialPrice : ℝ := 495
  let discounts : List ℝ := [0.20, 0.15, 0.10]
  let taxRate : ℝ := 0.05
  let finalDiscount : ℝ := 0.03
  abs (finalPrice initialPrice discounts taxRate finalDiscount - 308.54) < 0.01 := by
  sorry


end saree_price_calculation_l3093_309312


namespace new_variance_after_adding_datapoint_l3093_309369

/-- Given a sample with size 7, average 5, and variance 2, adding a new data point of 5 results in a new variance of 7/4 -/
theorem new_variance_after_adding_datapoint
  (sample_size : ℕ)
  (original_avg : ℝ)
  (original_var : ℝ)
  (new_datapoint : ℝ)
  (h1 : sample_size = 7)
  (h2 : original_avg = 5)
  (h3 : original_var = 2)
  (h4 : new_datapoint = 5) :
  let new_sample_size : ℕ := sample_size + 1
  let new_avg : ℝ := (sample_size * original_avg + new_datapoint) / new_sample_size
  let new_var : ℝ := (sample_size * original_var + sample_size * (new_avg - original_avg)^2) / new_sample_size
  new_var = 7/4 := by sorry

end new_variance_after_adding_datapoint_l3093_309369


namespace pyramid_volume_from_star_figure_l3093_309383

/-- The volume of a pyramid formed by folding a star figure cut from a square --/
theorem pyramid_volume_from_star_figure (outer_side : ℝ) (inner_side : ℝ) 
  (h_outer : outer_side = 40)
  (h_inner : inner_side = 15) :
  let base_area := inner_side ^ 2
  let midpoint_to_center := outer_side / 2
  let center_to_inner_side := inner_side / 2
  let triangle_height := midpoint_to_center - center_to_inner_side
  let pyramid_height := Real.sqrt (triangle_height ^ 2 - (inner_side / 2) ^ 2)
  let volume := (1 / 3) * base_area * pyramid_height
  volume = 750 := by sorry

end pyramid_volume_from_star_figure_l3093_309383


namespace consecutive_squares_theorem_l3093_309360

theorem consecutive_squares_theorem :
  (∀ x : ℤ, ¬∃ y : ℤ, 3 * x^2 + 2 = y^2) ∧
  (∀ x : ℤ, ¬∃ y : ℤ, 6 * x^2 + 6 * x + 19 = y^2) ∧
  (∃ x : ℤ, ∃ y : ℤ, 11 * x^2 + 110 = y^2) ∧
  (∃ y : ℤ, 11 * 23^2 + 110 = y^2) :=
by sorry

end consecutive_squares_theorem_l3093_309360


namespace intersection_point_l3093_309394

/-- The line equation is y = -3x + 3 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 3

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line y = -3x + 3 with the x-axis is (1, 0) -/
theorem intersection_point :
  ∃ (x y : ℝ), line_equation x y ∧ on_x_axis x y ∧ x = 1 ∧ y = 0 :=
sorry

end intersection_point_l3093_309394


namespace line_direction_vector_l3093_309336

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

/-- The point on the line at t = 0 -/
def initial_point (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := 
    (4 + 3 * t / Real.sqrt 34, 2 + 5 * t / Real.sqrt 34)
  let y (x : ℝ) : ℝ := (5 * x - 7) / 3
  ∀ (x : ℝ), x ≥ 4 → 
    let point := (x, y x)
    let dist := Real.sqrt ((x - 4)^2 + (y x - 2)^2)
    point = initial_point line + dist • direction_vector line ∧
    direction_vector line = (3 / Real.sqrt 34, 5 / Real.sqrt 34) :=
by sorry

end line_direction_vector_l3093_309336


namespace jessica_attended_one_game_l3093_309389

/-- The number of soccer games Jessica actually attended -/
def jessica_attended (total games : ℕ) (planned : ℕ) (skipped : ℕ) (rescheduled : ℕ) (additional_missed : ℕ) : ℕ :=
  planned - skipped - additional_missed

/-- Theorem stating that Jessica attended 1 game given the problem conditions -/
theorem jessica_attended_one_game :
  jessica_attended 12 8 3 2 4 = 1 := by
  sorry

end jessica_attended_one_game_l3093_309389


namespace eric_return_time_l3093_309396

def running_time : ℕ := 20
def jogging_time : ℕ := 10
def time_to_park : ℕ := running_time + jogging_time
def return_time_factor : ℕ := 3

theorem eric_return_time : time_to_park * return_time_factor = 90 := by
  sorry

end eric_return_time_l3093_309396


namespace distance_to_y_axis_reflection_distance_specific_point_l3093_309344

/-- The distance between a point and its reflection over the y-axis --/
theorem distance_to_y_axis_reflection (x y : ℝ) : 
  Real.sqrt ((x - (-x))^2 + (y - y)^2) = 2 * |x| :=
sorry

/-- The distance between (2, -4) and its reflection over the y-axis is 4 --/
theorem distance_specific_point : 
  Real.sqrt ((2 - (-2))^2 + (-4 - (-4))^2) = 4 :=
sorry

end distance_to_y_axis_reflection_distance_specific_point_l3093_309344


namespace a_profit_is_25_percent_l3093_309380

/-- Represents the profit percentage as a rational number between 0 and 1 -/
def ProfitPercentage := { x : ℚ // 0 ≤ x ∧ x ≤ 1 }

/-- The bicycle sale scenario -/
structure BicycleSale where
  cost_price_A : ℚ
  selling_price_BC : ℚ
  profit_percentage_B : ProfitPercentage

/-- Calculate the profit percentage of A -/
def profit_percentage_A (sale : BicycleSale) : ProfitPercentage :=
  sorry

/-- Theorem stating that A's profit percentage is 25% given the conditions -/
theorem a_profit_is_25_percent (sale : BicycleSale) 
  (h1 : sale.cost_price_A = 144)
  (h2 : sale.selling_price_BC = 225)
  (h3 : sale.profit_percentage_B = ⟨1/4, by norm_num⟩) :
  profit_percentage_A sale = ⟨1/4, by norm_num⟩ :=
sorry

end a_profit_is_25_percent_l3093_309380


namespace trig_inequality_and_equality_condition_l3093_309351

theorem trig_inequality_and_equality_condition (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) ≥ 9) ∧
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) = 9 ↔ 
    α = Real.arctan (Real.sqrt 2) ∧ β = π/4) :=
by sorry

end trig_inequality_and_equality_condition_l3093_309351


namespace circumradius_range_l3093_309371

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- Points P and Q on side AB, and R on side CD of a unit square -/
structure TrianglePoints (square : UnitSquare) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t, 0)
  Q_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (t, 0)
  R_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (t, 1)

/-- The circumradius of a triangle -/
def circumradius (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the range of possible circumradius values -/
theorem circumradius_range (square : UnitSquare) (points : TrianglePoints square) :
  1/2 < circumradius points.P points.Q points.R ∧ 
  circumradius points.P points.Q points.R ≤ Real.sqrt 2 / 2 := by
  sorry

end circumradius_range_l3093_309371


namespace max_dogs_and_fish_l3093_309340

/-- Represents the count of each animal type in the pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ
  parrots : ℕ
  fish : ℕ

/-- Checks if the given pet shop counts satisfy the ratio constraint -/
def satisfiesRatio (shop : PetShop) : Prop :=
  7 * shop.cats = 7 * shop.dogs ∧
  8 * shop.cats = 7 * shop.bunnies ∧
  3 * shop.cats = 7 * shop.parrots ∧
  5 * shop.cats = 7 * shop.fish

/-- Checks if the total number of dogs and bunnies is 330 -/
def totalDogsAndBunnies330 (shop : PetShop) : Prop :=
  shop.dogs + shop.bunnies = 330

/-- Checks if there are at least twice as many fish as cats -/
def twiceAsManyFishAsCats (shop : PetShop) : Prop :=
  shop.fish ≥ 2 * shop.cats

/-- Theorem stating the maximum number of dogs and corresponding number of fish -/
theorem max_dogs_and_fish (shop : PetShop) 
  (h1 : satisfiesRatio shop) 
  (h2 : totalDogsAndBunnies330 shop) 
  (h3 : twiceAsManyFishAsCats shop) :
  shop.dogs ≤ 154 ∧ (shop.dogs = 154 → shop.fish = 308) :=
sorry

end max_dogs_and_fish_l3093_309340


namespace smallest_bound_inequality_l3093_309384

theorem smallest_bound_inequality (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ ε > 0, ∃ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| > (M - ε)*(a^2 + b^2 + c^2)^2 ∧
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
sorry

end smallest_bound_inequality_l3093_309384


namespace smallest_number_divisible_l3093_309314

theorem smallest_number_divisible (n : ℕ) : n = 746 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m - 18 = 8 * k₁ ∧ 
    m - 18 = 14 * k₂ ∧ 
    m - 18 = 26 * k₃ ∧ 
    m - 18 = 28 * k₄)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n - 18 = 8 * k₁ ∧ 
    n - 18 = 14 * k₂ ∧ 
    n - 18 = 26 * k₃ ∧ 
    n - 18 = 28 * k₄) :=
by sorry

end smallest_number_divisible_l3093_309314


namespace find_number_l3093_309356

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 6) = 72 :=
by
  sorry

end find_number_l3093_309356


namespace certain_number_problem_l3093_309368

theorem certain_number_problem : ∃ x : ℚ, (24 : ℚ) = (4/5) * x + 4 ∧ x = 25 := by
  sorry

end certain_number_problem_l3093_309368


namespace product_of_numbers_l3093_309367

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 9) 
  (h2 : x^2 + y^2 = 157) : 
  x * y = 22 := by sorry

end product_of_numbers_l3093_309367


namespace greatest_integer_radius_l3093_309395

theorem greatest_integer_radius (r : ℝ) (h : r > 0) (area_constraint : π * r^2 < 100 * π) :
  ⌊r⌋ ≤ 9 ∧ ∃ (r' : ℝ), π * r'^2 < 100 * π ∧ ⌊r'⌋ = 9 :=
sorry

end greatest_integer_radius_l3093_309395


namespace decimal_computation_l3093_309377

theorem decimal_computation : (0.25 / 0.005) * 2 = 100 := by
  sorry

end decimal_computation_l3093_309377


namespace flag_distribution_l3093_309379

theorem flag_distribution (F : ℕ) (blue_percent red_percent : ℚ) :
  F % 2 = 0 →
  blue_percent = 60 / 100 →
  red_percent = 65 / 100 →
  blue_percent + red_percent - 1 = 25 / 100 :=
by sorry

end flag_distribution_l3093_309379


namespace money_transfer_problem_l3093_309357

/-- Represents the money transfer problem between Marco and Mary -/
theorem money_transfer_problem (marco_initial : ℕ) (mary_initial : ℕ) (mary_spends : ℕ) :
  marco_initial = 24 →
  mary_initial = 15 →
  mary_spends = 5 →
  let marco_gives := marco_initial / 2
  let mary_final := mary_initial + marco_gives - mary_spends
  let marco_final := marco_initial - marco_gives
  mary_final - marco_final = 10 := by sorry

end money_transfer_problem_l3093_309357


namespace perpendicular_bisector_and_parallel_line_l3093_309303

/-- Given two points A and B in the plane, this theorem proves:
    1. The equation of the perpendicular bisector of AB
    2. The equation of a line passing through P and parallel to AB -/
theorem perpendicular_bisector_and_parallel_line 
  (A B P : ℝ × ℝ) 
  (hA : A = (8, -6)) 
  (hB : B = (2, 2)) 
  (hP : P = (2, -3)) : 
  (∃ (a b c : ℝ), a * 3 = b * 4 ∧ c = 23 ∧ 
    (∀ (x y : ℝ), (a * x + b * y + c = 0) ↔ 
      (x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4)) ∧
  (∃ (d e f : ℝ), d * 4 = -e * 3 ∧ f = 1 ∧
    (∀ (x y : ℝ), (d * x + e * y + f = 0) ↔ 
      (y - P.2) = ((B.2 - A.2) / (B.1 - A.1)) * (x - P.1))) :=
by sorry

end perpendicular_bisector_and_parallel_line_l3093_309303


namespace tablet_down_payment_is_100_l3093_309305

/-- The down payment for a tablet purchase with given conditions. -/
def tablet_down_payment (cash_price installment_total first_4_months next_4_months last_4_months cash_savings : ℕ) : ℕ :=
  installment_total - (4 * first_4_months + 4 * next_4_months + 4 * last_4_months)

/-- Theorem stating that the down payment for the tablet is $100 under given conditions. -/
theorem tablet_down_payment_is_100 :
  tablet_down_payment 450 520 40 35 30 70 = 100 := by
  sorry

end tablet_down_payment_is_100_l3093_309305


namespace probability_second_red_given_first_red_l3093_309338

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4

theorem probability_second_red_given_first_red :
  let p_first_red := red_balls / total_balls
  let p_both_red := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))
  let p_second_red_given_first_red := p_both_red / p_first_red
  p_second_red_given_first_red = 5 / 9 :=
sorry

end probability_second_red_given_first_red_l3093_309338


namespace order_of_exponentials_l3093_309313

theorem order_of_exponentials :
  let a : ℝ := (2 : ℝ) ^ (4/5)
  let b : ℝ := (4 : ℝ) ^ (2/7)
  let c : ℝ := (25 : ℝ) ^ (1/5)
  b < a ∧ a < c :=
by sorry

end order_of_exponentials_l3093_309313


namespace geometric_sequence_first_term_l3093_309302

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_fifth : a 5 = Nat.factorial 7)
  (h_eighth : a 8 = Nat.factorial 8) :
  a 1 = 315 := by
  sorry

end geometric_sequence_first_term_l3093_309302


namespace cheryl_skittles_l3093_309332

/-- 
Given that Cheryl starts with a certain number of Skittles and receives additional Skittles,
this theorem proves the total number of Skittles Cheryl ends up with.
-/
theorem cheryl_skittles (initial : ℕ) (additional : ℕ) :
  initial = 8 → additional = 89 → initial + additional = 97 := by
  sorry

end cheryl_skittles_l3093_309332


namespace min_operations_to_256_l3093_309374

/-- Represents the allowed operations -/
inductive Operation
  | AddOne
  | MultiplyTwo

/-- Defines a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : OperationSequence) : ℕ :=
  ops.foldl (fun n op => match op with
    | Operation.AddOne => n + 1
    | Operation.MultiplyTwo => n * 2) start

/-- Checks if a sequence of operations transforms start into target -/
def isValidSequence (start target : ℕ) (ops : OperationSequence) : Prop :=
  applyOperations start ops = target

/-- The main theorem to be proved -/
theorem min_operations_to_256 :
  ∃ (ops : OperationSequence), isValidSequence 1 256 ops ∧ 
    ops.length = 8 ∧
    (∀ (other_ops : OperationSequence), isValidSequence 1 256 other_ops → 
      ops.length ≤ other_ops.length) :=
  sorry

end min_operations_to_256_l3093_309374


namespace sum_from_interest_and_discount_l3093_309310

/-- Given a sum, rate, and time, if the simple interest is 88 and the true discount is 80, then the sum is 880. -/
theorem sum_from_interest_and_discount (P r t : ℝ) 
  (h1 : P * r * t / 100 = 88)
  (h2 : P * r * t / (100 + r * t) = 80) : 
  P = 880 := by
  sorry

#check sum_from_interest_and_discount

end sum_from_interest_and_discount_l3093_309310


namespace other_polynomial_form_l3093_309328

/-- Given two polynomials with a specified difference, this theorem proves the form of the other polynomial. -/
theorem other_polynomial_form (a b c d : ℝ) 
  (diff : ℝ) -- The difference between the two polynomials
  (poly1 : ℝ) -- One of the polynomials
  (h1 : diff = c^2 * d^2 - a^2 * b^2) -- Condition on the difference
  (h2 : poly1 = a^2 * b^2 + c^2 * d^2 - 2*a*b*c*d) -- Condition on one polynomial
  : ∃ (poly2 : ℝ), (poly2 = 2*c^2*d^2 - 2*a*b*c*d ∨ poly2 = 2*a^2*b^2 - 2*a*b*c*d) ∧ 
    ((poly1 - poly2 = diff) ∨ (poly2 - poly1 = diff)) :=
by
  sorry

end other_polynomial_form_l3093_309328


namespace pigs_joined_l3093_309308

def initial_pigs : ℕ := 64
def final_pigs : ℕ := 86

theorem pigs_joined (initial : ℕ) (final : ℕ) (h1 : initial = initial_pigs) (h2 : final = final_pigs) :
  final - initial = 22 :=
by sorry

end pigs_joined_l3093_309308


namespace circular_garden_radius_l3093_309375

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 3) * π * r^2 → r = 6 := by
  sorry

end circular_garden_radius_l3093_309375


namespace polynomial_division_remainder_l3093_309345

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  X^4 + 3 * X^3 = (X^2 - 3 * X + 2) * q + r ∧ 
  r = 36 * X - 32 ∧ 
  r.degree < (X^2 - 3 * X + 2).degree := by
sorry

end polynomial_division_remainder_l3093_309345


namespace composite_number_division_l3093_309381

def first_seven_composite_product : ℕ := 4 * 6 * 8 * 9 * 10 * 12 * 14
def next_eight_composite_product : ℕ := 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25

theorem composite_number_division :
  (first_seven_composite_product : ℚ) / next_eight_composite_product = 1 / 2475 := by
  sorry

end composite_number_division_l3093_309381


namespace equation_solution_l3093_309376

theorem equation_solution :
  ∃! x : ℚ, 7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + x :=
by
  use -22/3
  sorry

end equation_solution_l3093_309376


namespace multiplication_division_remainder_problem_l3093_309392

theorem multiplication_division_remainder_problem :
  ∃ (x : ℕ), (55 * x) % 8 = 7 ∧ x = 1 := by
  sorry

end multiplication_division_remainder_problem_l3093_309392


namespace tan_roots_problem_l3093_309387

open Real

theorem tan_roots_problem (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : (tan α)^2 - 5*(tan α) + 6 = 0) (h4 : (tan β)^2 - 5*(tan β) + 6 = 0) :
  (α + β = 3*π/4) ∧ ¬∃(x : Real), tan (2*(α + β)) = x := by
  sorry

end tan_roots_problem_l3093_309387


namespace average_growth_rate_is_20_percent_l3093_309329

/-- Represents the monthly revenue growth rate as a real number between 0 and 1 -/
def MonthlyGrowthRate : Type := { r : ℝ // 0 ≤ r ∧ r ≤ 1 }

/-- The revenue in February in millions of yuan -/
def february_revenue : ℝ := 4

/-- The revenue increase rate from February to March -/
def march_increase_rate : ℝ := 0.1

/-- The revenue in May in millions of yuan -/
def may_revenue : ℝ := 633.6

/-- The number of months between March and May -/
def months_between : ℕ := 2

/-- Calculate the average monthly growth rate from March to May -/
def calculate_growth_rate (feb_rev : ℝ) (march_inc : ℝ) (may_rev : ℝ) (months : ℕ) : MonthlyGrowthRate :=
  sorry

theorem average_growth_rate_is_20_percent :
  calculate_growth_rate february_revenue march_increase_rate may_revenue months_between = ⟨0.2, sorry⟩ :=
sorry

end average_growth_rate_is_20_percent_l3093_309329


namespace exists_diagonal_le_two_l3093_309318

-- Define a convex hexagon
structure ConvexHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_convex : sorry -- Add convexity condition

-- Define the property that all sides have length ≤ 1
def all_sides_le_one (h : ConvexHexagon) : Prop :=
  ∀ i : Fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) ≤ 1

-- Define a diagonal of the hexagon
def diagonal (h : ConvexHexagon) (i j : Fin 6) : ℝ :=
  dist (h.vertices i) (h.vertices j)

-- Theorem statement
theorem exists_diagonal_le_two (h : ConvexHexagon) (h_sides : all_sides_le_one h) :
  ∃ (i j : Fin 6), i ≠ j ∧ diagonal h i j ≤ 2 := by
  sorry

end exists_diagonal_le_two_l3093_309318


namespace probability_one_white_two_red_l3093_309361

def white_balls : ℕ := 4
def red_balls : ℕ := 5
def total_balls : ℕ := white_balls + red_balls
def drawn_balls : ℕ := 3

def favorable_outcomes : ℕ := (Nat.choose white_balls 1) * (Nat.choose red_balls 2)
def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_one_white_two_red : 
  (favorable_outcomes : ℚ) / total_outcomes = 10 / 21 := by
  sorry

end probability_one_white_two_red_l3093_309361


namespace ratio_comparison_l3093_309373

theorem ratio_comparison (y : ℕ) (h : y > 4) : (3 : ℚ) / 4 < 3 / y :=
sorry

end ratio_comparison_l3093_309373


namespace total_tickets_is_150_l3093_309317

/-- The number of tickets Alan handed out -/
def alan_tickets : ℕ := 26

/-- The number of tickets Marcy handed out -/
def marcy_tickets : ℕ := 5 * alan_tickets - 6

/-- The total number of tickets handed out by Alan and Marcy -/
def total_tickets : ℕ := alan_tickets + marcy_tickets

/-- Theorem stating that the total number of tickets handed out is 150 -/
theorem total_tickets_is_150 : total_tickets = 150 := by
  sorry

end total_tickets_is_150_l3093_309317


namespace piggy_bank_problem_l3093_309399

theorem piggy_bank_problem (initial_amount : ℝ) : 
  initial_amount = 204 → 
  (initial_amount * (1 - 0.6) * (1 - 0.5) * (1 - 0.35)) = 26.52 :=
by sorry

end piggy_bank_problem_l3093_309399


namespace positive_function_condition_l3093_309309

theorem positive_function_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (2 - a^2) * x + a > 0) ↔ (0 < a ∧ a < 2) := by
  sorry

end positive_function_condition_l3093_309309


namespace gcd_1729_1337_l3093_309385

theorem gcd_1729_1337 : Nat.gcd 1729 1337 = 7 := by
  sorry

end gcd_1729_1337_l3093_309385


namespace hyperbola_equation_l3093_309315

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0) with focal length 2√5,
    and a parabola y = (1/4)x² + 1/4 tangent to its asymptote,
    prove that the equation of the hyperbola C is x²/4 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_focal : 5 = a^2 + b^2)
  (h_tangent : ∃ (x : ℝ), (1/4) * x^2 + (1/4) = (b/a) * x) :
  a = 2 ∧ b = 1 :=
sorry

end hyperbola_equation_l3093_309315


namespace marble_jar_problem_l3093_309350

theorem marble_jar_problem (jar1_blue_ratio : ℚ) (jar1_green_ratio : ℚ)
  (jar2_blue_ratio : ℚ) (jar2_green_ratio : ℚ) (total_green : ℕ) :
  jar1_blue_ratio = 7 / 10 →
  jar1_green_ratio = 3 / 10 →
  jar2_blue_ratio = 6 / 10 →
  jar2_green_ratio = 4 / 10 →
  total_green = 80 →
  ∃ (total_jar1 total_jar2 : ℕ),
    total_jar1 = total_jar2 ∧
    (jar1_green_ratio * total_jar1 + jar2_green_ratio * total_jar2 : ℚ) = total_green ∧
    ⌊jar1_blue_ratio * total_jar1 - jar2_blue_ratio * total_jar2⌋ = 11 :=
by
  sorry

end marble_jar_problem_l3093_309350


namespace cone_radius_l3093_309331

theorem cone_radius (r l : ℝ) : 
  r > 0 → l > 0 →
  π * l = 2 * π * r →
  π * r^2 + π * r * l = 3 * π →
  r = 1 := by
sorry

end cone_radius_l3093_309331


namespace inequality_proof_equality_condition_l3093_309354

theorem inequality_proof (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + b + c + d) / 2 ≥ 1 + Real.sqrt (a * b * c * d) := by
  sorry

theorem equality_condition (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + b + c + d) / 2 = 1 + Real.sqrt (a * b * c * d) ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 := by
  sorry

end inequality_proof_equality_condition_l3093_309354


namespace florist_roses_count_l3093_309393

theorem florist_roses_count (initial : ℕ) (sold : ℕ) (picked : ℕ) : 
  initial = 50 → sold = 15 → picked = 21 → initial - sold + picked = 56 := by
  sorry

end florist_roses_count_l3093_309393


namespace no_real_solutions_quadratic_l3093_309370

theorem no_real_solutions_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) ↔ k < -9/4 := by
  sorry

end no_real_solutions_quadratic_l3093_309370


namespace cylinder_symmetry_properties_l3093_309397

/-- Represents the type of a rotational cylinder -/
inductive CylinderType
  | DoubleSidedBounded
  | SingleSidedBounded
  | DoubleSidedUnbounded

/-- Represents the symmetry properties of a cylinder -/
structure CylinderSymmetry where
  hasAxisSymmetry : Bool
  hasPerpendicularPlaneSymmetry : Bool
  hasBundlePlanesSymmetry : Bool
  hasCenterSymmetry : Bool
  hasInfiniteCentersSymmetry : Bool
  hasTwoSystemsPlanesSymmetry : Bool

/-- Returns the symmetry properties for a given cylinder type -/
def getSymmetryProperties (cType : CylinderType) : CylinderSymmetry :=
  match cType with
  | CylinderType.DoubleSidedBounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := true,
      hasBundlePlanesSymmetry := true,
      hasCenterSymmetry := true,
      hasInfiniteCentersSymmetry := false,
      hasTwoSystemsPlanesSymmetry := false
    }
  | CylinderType.SingleSidedBounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := false,
      hasBundlePlanesSymmetry := true,
      hasCenterSymmetry := false,
      hasInfiniteCentersSymmetry := false,
      hasTwoSystemsPlanesSymmetry := false
    }
  | CylinderType.DoubleSidedUnbounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := false,
      hasBundlePlanesSymmetry := false,
      hasCenterSymmetry := false,
      hasInfiniteCentersSymmetry := true,
      hasTwoSystemsPlanesSymmetry := true
    }

theorem cylinder_symmetry_properties (cType : CylinderType) :
  (getSymmetryProperties cType).hasAxisSymmetry = true ∧
  ((cType = CylinderType.DoubleSidedBounded) → (getSymmetryProperties cType).hasPerpendicularPlaneSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedBounded ∨ cType = CylinderType.SingleSidedBounded) → (getSymmetryProperties cType).hasBundlePlanesSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedBounded) → (getSymmetryProperties cType).hasCenterSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedUnbounded) → (getSymmetryProperties cType).hasInfiniteCentersSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedUnbounded) → (getSymmetryProperties cType).hasTwoSystemsPlanesSymmetry = true) :=
by
  sorry


end cylinder_symmetry_properties_l3093_309397


namespace rectangular_plot_breadth_l3093_309319

/-- Represents a rectangular plot with a given breadth and area -/
structure RectangularPlot where
  breadth : ℝ
  area : ℝ
  length_is_thrice_breadth : length = 3 * breadth
  area_formula : area = length * breadth

/-- The breadth of a rectangular plot with thrice length and 2700 sq m area is 30 m -/
theorem rectangular_plot_breadth (plot : RectangularPlot) 
  (h_area : plot.area = 2700) : plot.breadth = 30 := by
  sorry

end rectangular_plot_breadth_l3093_309319


namespace complex_equation_implies_ratio_l3093_309337

theorem complex_equation_implies_ratio (m n : ℝ) :
  (2 + m * Complex.I) * (n - 2 * Complex.I) = -4 - 3 * Complex.I →
  m / n = 1 := by
sorry

end complex_equation_implies_ratio_l3093_309337


namespace power_zero_is_one_l3093_309339

theorem power_zero_is_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end power_zero_is_one_l3093_309339


namespace fraction_existence_and_nonexistence_l3093_309322

theorem fraction_existence_and_nonexistence :
  (∀ n : ℕ+, ∃ a b : ℕ+, (Real.sqrt n : ℝ) ≤ (a : ℝ) / (b : ℝ) ∧
                         (a : ℝ) / (b : ℝ) ≤ Real.sqrt (n + 1) ∧
                         (b : ℝ) ≤ Real.sqrt n + 1) ∧
  (∃ f : ℕ → ℕ+, ∀ k : ℕ, ∀ a b : ℕ+,
    (Real.sqrt (f k) : ℝ) ≤ (a : ℝ) / (b : ℝ) →
    (a : ℝ) / (b : ℝ) ≤ Real.sqrt (f k + 1) →
    (b : ℝ) > Real.sqrt (f k)) :=
by sorry

end fraction_existence_and_nonexistence_l3093_309322


namespace negative_twenty_one_div_three_l3093_309359

theorem negative_twenty_one_div_three : -21 / 3 = -7 := by
  sorry

end negative_twenty_one_div_three_l3093_309359


namespace max_value_under_constraint_l3093_309330

theorem max_value_under_constraint (x y : ℝ) :
  x^2 + y^2 ≤ 5 →
  3*|x + y| + |4*y + 9| + |7*y - 3*x - 18| ≤ 27 + 6*Real.sqrt 5 :=
by sorry

end max_value_under_constraint_l3093_309330


namespace x_value_proof_l3093_309358

theorem x_value_proof (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 9 = 9*y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by
  sorry

end x_value_proof_l3093_309358


namespace max_books_borrowed_l3093_309363

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 60) (h2 : zero_books = 4) (h3 : one_book = 18) 
  (h4 : two_books = 20) (h5 : avg_books = 5/2) : 
  ∃ (max_books : ℕ), max_books = 41 ∧ 
  ∀ (student_books : ℕ), student_books ≤ max_books :=
by
  sorry

end max_books_borrowed_l3093_309363


namespace largest_multiple_under_1000_l3093_309348

theorem largest_multiple_under_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  n < 1000 ∧
  ∀ m : ℕ, (m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000) → m ≤ n :=
by sorry

end largest_multiple_under_1000_l3093_309348


namespace fraction_problem_l3093_309343

theorem fraction_problem (a b : ℝ) (f : ℝ) : 
  a - b = 8 → 
  a + b = 24 → 
  f * (a + b) = 6 → 
  f = 1/4 := by
sorry

end fraction_problem_l3093_309343


namespace fractional_equation_solution_l3093_309347

theorem fractional_equation_solution :
  ∃ x : ℝ, (((1 - x) / (2 - x)) - 1 = ((2 * x - 5) / (x - 2))) ∧ x = 3 :=
by
  sorry

end fractional_equation_solution_l3093_309347


namespace average_children_in_families_with_children_l3093_309390

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families * total_average) / (total_families - childless_families) = 3.75 := by
  sorry

end average_children_in_families_with_children_l3093_309390


namespace sufficient_condition_l3093_309342

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement that a_4 and a_12 are roots of x^2 + 3x = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 + 3 * a 4 = 0 ∧ a 12 ^ 2 + 3 * a 12 = 0

/-- The theorem stating that the conditions are sufficient for a_8 = ±1 -/
theorem sufficient_condition (a : ℕ → ℝ) :
  geometric_sequence a → roots_condition a → (a 8 = 1 ∨ a 8 = -1) :=
by sorry

end sufficient_condition_l3093_309342


namespace f_not_satisfy_double_property_l3093_309304

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_not_satisfy_double_property : ∃ x : ℝ, f (2 * x) ≠ 2 * f x := by
  sorry

end f_not_satisfy_double_property_l3093_309304


namespace abs_negative_2022_l3093_309321

theorem abs_negative_2022 : |(-2022 : ℤ)| = 2022 := by
  sorry

end abs_negative_2022_l3093_309321


namespace tens_digit_of_8_pow_2023_l3093_309366

/-- The length of the cycle of the last two digits of 8^n -/
def cycle_length : ℕ := 20

/-- The last two digits of 8^3 -/
def last_two_digits_8_cubed : ℕ := 12

/-- The exponent we're interested in -/
def target_exponent : ℕ := 2023

theorem tens_digit_of_8_pow_2023 : 
  (target_exponent % cycle_length = 3) → 
  (last_two_digits_8_cubed / 10 = 1) → 
  (8^target_exponent / 10 % 10 = 1) :=
by sorry

end tens_digit_of_8_pow_2023_l3093_309366


namespace inverse_graph_point_l3093_309301

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the condition that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the condition that the graph of y = x - f(x) passes through (2,5)
axiom graph_condition : 2 - f 2 = 5

-- Theorem to prove
theorem inverse_graph_point :
  (∀ x, f_inv (f x) = x ∧ f (f_inv x) = x) →
  (2 - f 2 = 5) →
  f_inv (-3) + 3 = 5 :=
by sorry

end inverse_graph_point_l3093_309301


namespace simplify_fraction_l3093_309346

theorem simplify_fraction : (36 : ℚ) / 54 = 2 / 3 := by sorry

end simplify_fraction_l3093_309346


namespace line_segments_in_proportion_l3093_309316

theorem line_segments_in_proportion : 
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2 * Real.sqrt 3
  let d : ℝ := Real.sqrt 15
  a * d = b * c := by sorry

end line_segments_in_proportion_l3093_309316


namespace collinear_points_k_value_l3093_309378

/-- Given three distinct collinear points A, B, and C with coordinates relative to point O,
    prove that the value of k is -1/4. -/
theorem collinear_points_k_value
  (k : ℝ)
  (A B C : ℝ × ℝ)
  (hA : A = (k, 2))
  (hB : B = (1, 2*k))
  (hC : C = (1-k, -1))
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hCollinear : ∃ (t : ℝ), C - A = t • (B - A)) :
  k = -1/4 := by
  sorry

end collinear_points_k_value_l3093_309378


namespace min_value_of_x_plus_y_l3093_309353

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z w : ℝ, z > 0 → w > 0 → 2*z + 8*w - z*w = 0 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 8*b - a*b = 0 ∧ a + b = 18 := by
sorry

end min_value_of_x_plus_y_l3093_309353


namespace grape_popsicles_count_l3093_309391

/-- The number of cherry popsicles -/
def cherry_popsicles : ℕ := 13

/-- The number of banana popsicles -/
def banana_popsicles : ℕ := 2

/-- The total number of popsicles -/
def total_popsicles : ℕ := 17

/-- The number of grape popsicles -/
def grape_popsicles : ℕ := total_popsicles - cherry_popsicles - banana_popsicles

theorem grape_popsicles_count : grape_popsicles = 2 := by
  sorry

end grape_popsicles_count_l3093_309391


namespace triangle_properties_l3093_309364

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  -- Given condition
  cos (2*C) - cos (2*A) = 2 * sin (π/3 + C) * sin (π/3 - C) →
  -- Part 1: Prove A = π/3
  A = π/3 ∧
  -- Part 2: Prove range of 2b-c
  (a = sqrt 3 ∧ b ≥ a → 2*b - c ≥ sqrt 3 ∧ 2*b - c < 2 * sqrt 3) :=
by sorry

end triangle_properties_l3093_309364


namespace f_of_3_equals_41_l3093_309382

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

theorem f_of_3_equals_41 : f 3 = 41 := by sorry

end f_of_3_equals_41_l3093_309382


namespace floor_sqrt_80_l3093_309300

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l3093_309300


namespace garden_occupation_fraction_l3093_309311

theorem garden_occupation_fraction :
  ∀ (garden_length garden_width : ℝ)
    (trapezoid_short_side trapezoid_long_side : ℝ)
    (sandbox_side : ℝ),
  garden_length = 40 →
  garden_width = 8 →
  trapezoid_long_side - trapezoid_short_side = 10 →
  trapezoid_short_side + trapezoid_long_side = garden_length →
  sandbox_side = 5 →
  let triangle_leg := (trapezoid_long_side - trapezoid_short_side) / 2
  let triangle_area := triangle_leg ^ 2 / 2
  let total_triangles_area := 2 * triangle_area
  let sandbox_area := sandbox_side ^ 2
  let occupied_area := total_triangles_area + sandbox_area
  let garden_area := garden_length * garden_width
  occupied_area / garden_area = 5 / 32 :=
by sorry

end garden_occupation_fraction_l3093_309311


namespace jimmy_bread_packs_l3093_309335

/-- The number of packs of bread needed for a given number of sandwiches -/
def bread_packs_needed (num_sandwiches : ℕ) (slices_per_sandwich : ℕ) (slices_per_pack : ℕ) (initial_slices : ℕ) : ℕ :=
  ((num_sandwiches * slices_per_sandwich - initial_slices) + slices_per_pack - 1) / slices_per_pack

/-- Theorem: Jimmy needs 4 packs of bread for his picnic -/
theorem jimmy_bread_packs : bread_packs_needed 8 2 4 0 = 4 := by
  sorry

end jimmy_bread_packs_l3093_309335


namespace prime_sequence_ones_digit_l3093_309398

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > 5 →
  q = p + 8 →
  r = q + 8 →
  s = r + 8 →
  ones_digit p = 3 := by
  sorry

end prime_sequence_ones_digit_l3093_309398


namespace race_runners_count_l3093_309372

theorem race_runners_count :
  ∀ (total_runners : ℕ) (ammar_position : ℕ) (julia_position : ℕ),
  ammar_position > 0 →
  julia_position > ammar_position →
  ammar_position - 1 = (total_runners - ammar_position) / 2 →
  julia_position = ammar_position + 10 →
  julia_position - 1 = 2 * (total_runners - julia_position) →
  total_runners = 31 :=
by
  sorry

#check race_runners_count

end race_runners_count_l3093_309372


namespace parallelogram_properties_l3093_309388

/-- Represents a parallelogram with specific properties -/
structure Parallelogram where
  /-- Length of the shorter side -/
  side_short : ℝ
  /-- Length of the longer side -/
  side_long : ℝ
  /-- Length of the first diagonal -/
  diag1 : ℝ
  /-- Length of the second diagonal -/
  diag2 : ℝ
  /-- The difference between the lengths of the sides is 7 -/
  side_diff : side_long - side_short = 7
  /-- A perpendicular from a vertex divides a diagonal into segments of 6 and 15 -/
  diag_segments : diag1 = 6 + 15

/-- Theorem stating the properties of the specific parallelogram -/
theorem parallelogram_properties : 
  ∃ (p : Parallelogram), 
    p.side_short = 10 ∧ 
    p.side_long = 17 ∧ 
    p.diag1 = 21 ∧ 
    p.diag2 = Real.sqrt 337 := by
  sorry

end parallelogram_properties_l3093_309388


namespace valid_committee_count_l3093_309327

/-- Represents the number of male professors in each department -/
def male_profs : Fin 3 → Nat
  | 0 => 3  -- Physics
  | 1 => 2  -- Chemistry
  | 2 => 2  -- Biology

/-- Represents the number of female professors in each department -/
def female_profs : Fin 3 → Nat
  | 0 => 3  -- Physics
  | 1 => 2  -- Chemistry
  | 2 => 3  -- Biology

/-- The total number of departments -/
def num_departments : Nat := 3

/-- The required committee size -/
def committee_size : Nat := 6

/-- The required number of male professors in the committee -/
def required_males : Nat := 3

/-- The required number of female professors in the committee -/
def required_females : Nat := 3

/-- Calculates the number of valid committee formations -/
def count_valid_committees : Nat := sorry

theorem valid_committee_count :
  count_valid_committees = 864 := by sorry

end valid_committee_count_l3093_309327


namespace jeremy_songs_l3093_309320

def songs_problem (songs_yesterday : ℕ) (difference : ℕ) : Prop :=
  let songs_today : ℕ := songs_yesterday + difference
  let total_songs : ℕ := songs_yesterday + songs_today
  total_songs = 23

theorem jeremy_songs :
  songs_problem 9 5 := by
  sorry

end jeremy_songs_l3093_309320
