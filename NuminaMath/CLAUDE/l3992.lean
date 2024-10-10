import Mathlib

namespace i_power_2016_l3992_399258

/-- The complex unit i -/
def i : ℂ := Complex.I

/-- Given properties of i -/
axiom i_power_1 : i^1 = i
axiom i_power_2 : i^2 = -1
axiom i_power_3 : i^3 = -i
axiom i_power_4 : i^4 = 1
axiom i_power_5 : i^5 = i

/-- Theorem: i^2016 = 1 -/
theorem i_power_2016 : i^2016 = 1 := by
  sorry

end i_power_2016_l3992_399258


namespace total_shells_is_83_l3992_399229

/-- The total number of shells in the combined collection of five friends -/
def total_shells (initial_shells : ℕ) 
  (ed_limpet ed_oyster ed_conch ed_scallop : ℕ)
  (jacob_extra : ℕ)
  (marissa_limpet marissa_oyster marissa_conch marissa_scallop : ℕ)
  (priya_clam priya_mussel priya_conch priya_oyster : ℕ)
  (carlos_shells : ℕ) : ℕ :=
  initial_shells + 
  (ed_limpet + ed_oyster + ed_conch + ed_scallop) + 
  (ed_limpet + ed_oyster + ed_conch + ed_scallop + jacob_extra) +
  (marissa_limpet + marissa_oyster + marissa_conch + marissa_scallop) +
  (priya_clam + priya_mussel + priya_conch + priya_oyster) +
  carlos_shells

/-- The theorem stating that the total number of shells is 83 -/
theorem total_shells_is_83 : 
  total_shells 2 7 2 4 3 2 5 6 3 1 8 4 3 2 15 = 83 := by
  sorry

end total_shells_is_83_l3992_399229


namespace subset_intersection_complement_empty_l3992_399216

theorem subset_intersection_complement_empty
  {U : Type} [Nonempty U]
  (M N : Set U)
  (h : M ⊆ N) :
  M ∩ (Set.univ \ N) = ∅ := by
sorry

end subset_intersection_complement_empty_l3992_399216


namespace gummy_bear_manufacturing_time_l3992_399261

/-- The time needed to manufacture gummy bears for a given number of packets -/
def manufacturingTime (bearsPerMinute : ℕ) (bearsPerPacket : ℕ) (numPackets : ℕ) : ℕ :=
  (numPackets * bearsPerPacket) / bearsPerMinute

theorem gummy_bear_manufacturing_time :
  manufacturingTime 300 50 240 = 40 := by
  sorry

end gummy_bear_manufacturing_time_l3992_399261


namespace reciprocal_of_difference_l3992_399259

theorem reciprocal_of_difference : (((1 : ℚ) / 3 - (1 : ℚ) / 4)⁻¹ : ℚ) = 12 := by
  sorry

end reciprocal_of_difference_l3992_399259


namespace fraction_decomposition_l3992_399270

theorem fraction_decomposition (n : ℕ) (h1 : n ≥ 5) (h2 : Odd n) :
  (2 : ℚ) / n = 1 / ((n + 1) / 2) + 1 / (n * (n + 1) / 2) := by
  sorry

end fraction_decomposition_l3992_399270


namespace triangle_side_values_l3992_399220

theorem triangle_side_values (A B C : ℝ) (a b c : ℝ) : 
  A = 30 * π / 180 →  -- Convert 30° to radians
  a = 1 →
  c = Real.sqrt 3 →
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) →  -- Law of Cosines
  (b = 1 ∨ b = 2) := by
  sorry

end triangle_side_values_l3992_399220


namespace tobias_change_l3992_399227

def shoe_cost : ℕ := 95
def saving_months : ℕ := 3
def monthly_allowance : ℕ := 5
def lawn_mowing_charge : ℕ := 15
def driveway_shoveling_charge : ℕ := 7
def lawns_mowed : ℕ := 4
def driveways_shoveled : ℕ := 5

def total_savings : ℕ := 
  saving_months * monthly_allowance + 
  lawns_mowed * lawn_mowing_charge + 
  driveways_shoveled * driveway_shoveling_charge

theorem tobias_change : total_savings - shoe_cost = 15 := by
  sorry

end tobias_change_l3992_399227


namespace parabola_one_x_intercept_l3992_399203

/-- The parabola defined by x = -3y^2 + 2y + 3 has exactly one x-intercept. -/
theorem parabola_one_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 := by
  sorry

end parabola_one_x_intercept_l3992_399203


namespace carousel_horses_count_l3992_399281

theorem carousel_horses_count :
  let blue_horses : ℕ := 3
  let purple_horses : ℕ := 3 * blue_horses
  let green_horses : ℕ := 2 * purple_horses
  let gold_horses : ℕ := green_horses / 6
  blue_horses + purple_horses + green_horses + gold_horses = 33 :=
by sorry

end carousel_horses_count_l3992_399281


namespace system_of_equations_sum_l3992_399200

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end system_of_equations_sum_l3992_399200


namespace mixed_fruit_juice_cost_l3992_399251

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost : ℝ := 1399.45

/-- The cost per litre of the açaí berry juice -/
def acai_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used (in litres) -/
def mixed_fruit_volume : ℝ := 36

/-- The volume of açaí berry juice used (in litres) -/
def acai_volume : ℝ := 24

/-- The cost per litre of the mixed fruit juice -/
def mixed_fruit_cost : ℝ := 265.6166667

theorem mixed_fruit_juice_cost : 
  mixed_fruit_volume * mixed_fruit_cost + acai_volume * acai_cost = 
  cocktail_cost * (mixed_fruit_volume + acai_volume) := by
  sorry

end mixed_fruit_juice_cost_l3992_399251


namespace total_marbles_lost_l3992_399275

def initial_marbles : ℕ := 120

def marbles_lost_outside (total : ℕ) : ℕ :=
  total / 4

def marbles_given_away (remaining : ℕ) : ℕ :=
  remaining / 2

def marbles_lost_bag_tear : ℕ := 10

theorem total_marbles_lost : 
  let remaining_after_outside := initial_marbles - marbles_lost_outside initial_marbles
  let remaining_after_giving := remaining_after_outside - marbles_given_away remaining_after_outside
  let final_remaining := remaining_after_giving - marbles_lost_bag_tear
  initial_marbles - final_remaining = 85 := by
  sorry

end total_marbles_lost_l3992_399275


namespace product_of_solutions_l3992_399252

theorem product_of_solutions : ∃ (x₁ x₂ : ℝ), 
  (|5 * x₁| + 4 = 44) ∧ 
  (|5 * x₂| + 4 = 44) ∧ 
  (x₁ ≠ x₂) ∧
  (x₁ * x₂ = -64) := by
  sorry

end product_of_solutions_l3992_399252


namespace wage_increase_l3992_399299

theorem wage_increase (original_wage new_wage : ℝ) (increase_percentage : ℝ) : 
  new_wage = 90 ∧ 
  increase_percentage = 50 ∧ 
  new_wage = original_wage * (1 + increase_percentage / 100) → 
  original_wage = 60 := by
sorry

end wage_increase_l3992_399299


namespace binomial_20_10_l3992_399289

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 31824)
                       (h2 : Nat.choose 18 9 = 48620)
                       (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 172822 := by
  sorry

end binomial_20_10_l3992_399289


namespace x_makes_2n_plus_x_composite_x_is_correct_l3992_399257

/-- The number added to 2n to make it not prime when n = 4 -/
def x : ℕ := 1

/-- The smallest n for which 2n + x is not prime -/
def smallest_n : ℕ := 4

theorem x_makes_2n_plus_x_composite : 
  ¬ Nat.Prime (2 * smallest_n + x) ∧ 
  ∀ m < smallest_n, Nat.Prime (2 * m + x) := by
  sorry

theorem x_is_correct : x = 1 := by
  sorry

end x_makes_2n_plus_x_composite_x_is_correct_l3992_399257


namespace sufficient_condition_for_inequality_l3992_399205

theorem sufficient_condition_for_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : a^2 + b^2 < 1) : 
  a * b + 1 > a + b := by
  sorry

end sufficient_condition_for_inequality_l3992_399205


namespace house_painting_cost_l3992_399210

/-- Calculates the cost of painting a house given its area and price per square foot. -/
def paintingCost (area : ℝ) (pricePerSqFt : ℝ) : ℝ :=
  area * pricePerSqFt

/-- Proves that the cost of painting a house with an area of 484 sq ft
    at a rate of Rs. 20 per sq ft is equal to Rs. 9,680. -/
theorem house_painting_cost :
  paintingCost 484 20 = 9680 := by
  sorry

end house_painting_cost_l3992_399210


namespace cost_per_pound_of_beef_l3992_399260

/-- Given a grocery bill with chicken, beef, and oil, prove the cost per pound of beef. -/
theorem cost_per_pound_of_beef
  (total_bill : ℝ)
  (chicken_weight : ℝ)
  (beef_weight : ℝ)
  (oil_volume : ℝ)
  (oil_cost : ℝ)
  (chicken_cost : ℝ)
  (h1 : total_bill = 16)
  (h2 : chicken_weight = 2)
  (h3 : beef_weight = 3)
  (h4 : oil_volume = 1)
  (h5 : oil_cost = 1)
  (h6 : chicken_cost = 3) :
  (total_bill - chicken_cost - oil_cost) / beef_weight = 4 := by
sorry

end cost_per_pound_of_beef_l3992_399260


namespace eighth_diagram_shaded_fraction_l3992_399201

/-- The number of shaded triangles in the nth diagram (n ≥ 1) -/
def shaded (n : ℕ) : ℕ := (n - 1) * n / 2

/-- The total number of small triangles in the nth diagram -/
def total (n : ℕ) : ℕ := n ^ 2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded n / total n

theorem eighth_diagram_shaded_fraction :
  shaded_fraction 8 = 7 / 16 := by sorry

end eighth_diagram_shaded_fraction_l3992_399201


namespace tom_peeled_24_potatoes_l3992_399224

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initialPile : ℕ
  maryRate : ℕ
  tomRate : ℕ
  maryAloneTime : ℕ

/-- Calculates the number of potatoes Tom peeled -/
def potatoesPeeledByTom (scenario : PotatoPeeling) : ℕ :=
  let potatoesPeeledByMaryAlone := scenario.maryRate * scenario.maryAloneTime
  let remainingPotatoes := scenario.initialPile - potatoesPeeledByMaryAlone
  let combinedRate := scenario.maryRate + scenario.tomRate
  let timeToFinish := remainingPotatoes / combinedRate
  scenario.tomRate * timeToFinish

/-- Theorem stating that Tom peeled 24 potatoes -/
theorem tom_peeled_24_potatoes :
  let scenario : PotatoPeeling := {
    initialPile := 60,
    maryRate := 4,
    tomRate := 6,
    maryAloneTime := 5
  }
  potatoesPeeledByTom scenario = 24 := by sorry

end tom_peeled_24_potatoes_l3992_399224


namespace function_uniqueness_l3992_399273

theorem function_uniqueness (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) = f n + 1)
  (h2 : ∃ k, f k = 1)
  (h3 : ∀ m, ∃ n, f n ≤ m) :
  ∀ n, f n = n + 1 := by
sorry

end function_uniqueness_l3992_399273


namespace problem_solution_l3992_399209

theorem problem_solution : 
  (100.2 * 99.8 = 9999.96) ∧ (103^2 = 10609) := by sorry

end problem_solution_l3992_399209


namespace exponential_curve_logarithm_relation_l3992_399230

/-- Proves the relationship between u, b, x, and c for an exponential curve -/
theorem exponential_curve_logarithm_relation 
  (a b x : ℝ) 
  (y : ℝ := a * Real.exp (b * x)) 
  (u : ℝ := Real.log y) 
  (c : ℝ := Real.log a) : 
  u = b * x + c := by
  sorry

end exponential_curve_logarithm_relation_l3992_399230


namespace percentage_problem_l3992_399287

theorem percentage_problem (x : ℝ) : 120 = 2.4 * x → x = 50 := by
  sorry

end percentage_problem_l3992_399287


namespace greatest_common_length_l3992_399277

theorem greatest_common_length (a b c : Nat) (ha : a = 39) (hb : b = 52) (hc : c = 65) :
  Nat.gcd a (Nat.gcd b c) = 13 := by
  sorry

end greatest_common_length_l3992_399277


namespace tangent_line_equation_l3992_399249

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the tangent condition
def is_tangent (x y : ℝ) : Prop := ∃ (t : ℝ), circle_M (x + t) (y + t) ∧ 
  ∀ (s : ℝ), s ≠ t → ¬(circle_M (x + s) (y + s))

-- Define the minimization condition
def is_minimized (P M A B : ℝ × ℝ) : Prop := 
  ∀ (Q : ℝ × ℝ), point_P Q.1 Q.2 → 
    (Q.1 - M.1)^2 + (Q.2 - M.2)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥
    (P.1 - M.1)^2 + (P.2 - M.2)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem tangent_line_equation :
  ∀ (P M A B : ℝ × ℝ),
    circle_M M.1 M.2 →
    point_P P.1 P.2 →
    is_tangent (P.1 - A.1) (P.2 - A.2) →
    is_tangent (P.1 - B.1) (P.2 - B.2) →
    is_minimized P M A B →
    2 * A.1 + A.2 + 1 = 0 ∧ 2 * B.1 + B.2 + 1 = 0 := by
  sorry

end tangent_line_equation_l3992_399249


namespace larger_square_side_length_l3992_399271

theorem larger_square_side_length (smaller_side : ℝ) (larger_side : ℝ) : 
  smaller_side = 5 →
  larger_side = smaller_side + 5 →
  larger_side ^ 2 = 4 * smaller_side ^ 2 →
  larger_side = 10 := by
sorry

end larger_square_side_length_l3992_399271


namespace basketball_probability_l3992_399242

theorem basketball_probability (p_at_least_one p_hs3 p_pro3 : ℝ) :
  p_at_least_one = 0.9333333333333333 →
  p_hs3 = 1/2 →
  p_pro3 = 1/3 →
  1 - (1 - p_hs3) * (1 - p_pro3) * (1 - 0.8) = p_at_least_one :=
by sorry

end basketball_probability_l3992_399242


namespace number_exceeding_fraction_l3992_399290

theorem number_exceeding_fraction : ∃ x : ℝ, x = (5 / 9) * x + 150 ∧ x = 337.5 := by
  sorry

end number_exceeding_fraction_l3992_399290


namespace segment_length_unit_circle_l3992_399207

/-- The length of the segment cut by a unit circle from the line y - x = 1 is √2. -/
theorem segment_length_unit_circle : ∃ (L : ℝ), L = Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x^2 + y^2 = 1 ∧ y - x = 1 → 
  ∃ (x' y' : ℝ), x'^2 + y'^2 = 1 ∧ y' - x' = 1 ∧ 
  Real.sqrt ((x - x')^2 + (y - y')^2) = L :=
sorry

end segment_length_unit_circle_l3992_399207


namespace jeans_cost_l3992_399286

theorem jeans_cost (mary_sunglasses : ℕ) (mary_sunglasses_price : ℕ) (rose_shoes : ℕ) (rose_cards : ℕ) (rose_cards_price : ℕ) :
  mary_sunglasses = 2 →
  mary_sunglasses_price = 50 →
  rose_shoes = 150 →
  rose_cards = 2 →
  rose_cards_price = 25 →
  ∃ (jeans_cost : ℕ),
    mary_sunglasses * mary_sunglasses_price + jeans_cost =
    rose_shoes + rose_cards * rose_cards_price ∧
    jeans_cost = 100 :=
by sorry

end jeans_cost_l3992_399286


namespace volume_removed_tetrahedra_2x2x3_l3992_399297

/-- Represents a rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the volume of removed tetrahedra when corners are sliced to form regular hexagons -/
def volume_removed_tetrahedra (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem: The volume of removed tetrahedra for a 2x2x3 rectangular prism is (22 - 46√2) / 3 -/
theorem volume_removed_tetrahedra_2x2x3 :
  volume_removed_tetrahedra ⟨2, 2, 3⟩ = (22 - 46 * Real.sqrt 2) / 3 := by
  sorry

end volume_removed_tetrahedra_2x2x3_l3992_399297


namespace always_odd_expression_l3992_399265

theorem always_odd_expression (o n : ℕ) (ho : Odd o) (hn : n > 0) :
  Odd (o^3 + n^2 * o^2) := by
  sorry

end always_odd_expression_l3992_399265


namespace prime_polynomial_R_value_l3992_399278

theorem prime_polynomial_R_value :
  ∀ (R Q : ℤ),
    R > 0 →
    (∃ p : ℕ+, Nat.Prime p ∧ (R^3 + 4*R^2 + (Q - 93)*R + 14*Q + 10 : ℤ) = p) →
    R = 5 := by
  sorry

end prime_polynomial_R_value_l3992_399278


namespace orchids_planted_tomorrow_l3992_399212

/-- Proves that the number of orchid bushes to be planted tomorrow is 25 --/
theorem orchids_planted_tomorrow
  (initial : ℕ) -- Initial number of orchid bushes
  (planted_today : ℕ) -- Number of orchid bushes planted today
  (final : ℕ) -- Final number of orchid bushes
  (h1 : initial = 47)
  (h2 : planted_today = 37)
  (h3 : final = 109) :
  final - (initial + planted_today) = 25 := by
  sorry

#check orchids_planted_tomorrow

end orchids_planted_tomorrow_l3992_399212


namespace jen_buys_50_candy_bars_l3992_399254

/-- The number of candy bars Jen buys -/
def num_candy_bars : ℕ := 50

/-- The cost of buying each candy bar in cents -/
def buy_price : ℕ := 80

/-- The selling price of each candy bar in cents -/
def sell_price : ℕ := 100

/-- The number of candy bars Jen sells -/
def num_sold : ℕ := 48

/-- Jen's profit in cents -/
def profit : ℕ := 800

/-- Theorem stating that given the conditions, Jen buys 50 candy bars -/
theorem jen_buys_50_candy_bars :
  (sell_price * num_sold) - (buy_price * num_candy_bars) = profit :=
by sorry

end jen_buys_50_candy_bars_l3992_399254


namespace solve_cab_driver_problem_l3992_399218

def cab_driver_problem (day1 day2 day4 day5 average : ℕ) : Prop :=
  let total := 5 * average
  let known_sum := day1 + day2 + day4 + day5
  let day3 := total - known_sum
  (day1 = 300) ∧ (day2 = 150) ∧ (day4 = 400) ∧ (day5 = 500) ∧ (average = 420) → day3 = 750

theorem solve_cab_driver_problem :
  cab_driver_problem 300 150 400 500 420 :=
by
  sorry

end solve_cab_driver_problem_l3992_399218


namespace solve_system_l3992_399256

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 18) : y = 5 := by
  sorry

end solve_system_l3992_399256


namespace rectangle_area_theorem_l3992_399202

-- Define the rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the circle
structure Circle where
  radius : ℝ

-- Define the problem conditions
def tangent_and_midpoint (rect : Rectangle) (circ : Circle) : Prop :=
  -- Circle is tangent to sides EF and EH at their midpoints
  -- and passes through the midpoint of side FG
  True

-- Theorem statement
theorem rectangle_area_theorem (rect : Rectangle) (circ : Circle) :
  tangent_and_midpoint rect circ →
  rect.width * rect.height = 4 * circ.radius ^ 2 := by
  sorry

end rectangle_area_theorem_l3992_399202


namespace expenditure_problem_l3992_399233

theorem expenditure_problem (initial_amount : ℝ) : 
  let remaining_after_clothes := (2/3) * initial_amount
  let remaining_after_food := (4/5) * remaining_after_clothes
  let remaining_after_travel := (3/4) * remaining_after_food
  let remaining_after_entertainment := (5/7) * remaining_after_travel
  let final_remaining := (5/6) * remaining_after_entertainment
  final_remaining = 200 → initial_amount = 840 := by
sorry

end expenditure_problem_l3992_399233


namespace next_number_with_property_l3992_399263

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_number_with_property :
  has_property 1818 ∧
  (∀ m, 1818 < m ∧ m < 1832 → ¬ has_property m) ∧
  has_property 1832 := by sorry

end next_number_with_property_l3992_399263


namespace flower_arrangement_problem_l3992_399266

/-- Represents the flower arrangement problem --/
theorem flower_arrangement_problem 
  (initial_roses : ℕ) 
  (initial_daisies : ℕ) 
  (thrown_roses : ℕ) 
  (thrown_daisies : ℕ) 
  (final_roses : ℕ) 
  (final_daisies : ℕ) 
  (time_constraint : ℕ) :
  initial_roses = 21 →
  initial_daisies = 17 →
  thrown_roses = 34 →
  thrown_daisies = 25 →
  final_roses = 15 →
  final_daisies = 10 →
  time_constraint = 2 →
  (thrown_roses + thrown_daisies) - 
  ((thrown_roses - initial_roses + final_roses) + 
   (thrown_daisies - initial_daisies + final_daisies)) = 13 :=
by sorry


end flower_arrangement_problem_l3992_399266


namespace real_part_of_z_l3992_399236

theorem real_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  Complex.re z = 1 := by sorry

end real_part_of_z_l3992_399236


namespace difference_of_squares_l3992_399223

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 19) : x^2 - y^2 = 190 := by
  sorry

end difference_of_squares_l3992_399223


namespace complex_pure_imaginary_l3992_399245

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I * (a - 2 * Complex.I) + (2 : ℂ) * (a - 2 * Complex.I)).re = 0 → a = -1 :=
by sorry

end complex_pure_imaginary_l3992_399245


namespace average_transformation_l3992_399238

theorem average_transformation (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 8) : 
  ((a₁ + 10) + (a₂ - 10) + (a₃ + 10) + (a₄ - 10) + (a₅ + 10)) / 5 = 10 := by
  sorry

end average_transformation_l3992_399238


namespace cylinder_radius_l3992_399206

/-- 
Theorem: For a cylinder with an original height of 3 inches, 
if increasing either the radius or the height by 7 inches results in the same volume, 
then the original radius must be 7 inches.
-/
theorem cylinder_radius (r : ℝ) : 
  r > 0 →  -- radius must be positive
  3 * π * (r + 7)^2 = 10 * π * r^2 → -- volumes are equal
  r = 7 := by
sorry

end cylinder_radius_l3992_399206


namespace perpendicular_slope_l3992_399239

/-- Given a line with equation 4x - 5y = 10, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 10) → (slope_of_perpendicular_line = -5/4) :=
by
  sorry

end perpendicular_slope_l3992_399239


namespace binomial_coefficient_26_6_l3992_399276

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 3 = 2024)
                                  (h2 : Nat.choose 24 4 = 10626)
                                  (h3 : Nat.choose 24 5 = 42504) :
  Nat.choose 26 6 = 230230 := by
  sorry

end binomial_coefficient_26_6_l3992_399276


namespace equation_solution_set_l3992_399298

theorem equation_solution_set : 
  ∃ (S : Set ℝ), S = {x : ℝ | 16 * Real.sin (Real.pi * x) * Real.cos (Real.pi * x) = 16 * x + 1 / x} ∧ 
  S = {-(1/4), 1/4} := by
  sorry

end equation_solution_set_l3992_399298


namespace item_value_proof_l3992_399234

/-- Proves that the total value of an item is $2,590 given the import tax conditions -/
theorem item_value_proof (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) :
  tax_rate = 0.07 →
  tax_threshold = 1000 →
  tax_paid = 111.30 →
  ∃ (total_value : ℝ), 
    tax_rate * (total_value - tax_threshold) = tax_paid ∧
    total_value = 2590 := by
  sorry

end item_value_proof_l3992_399234


namespace specimen_expiration_time_l3992_399240

def seconds_in_day : ℕ := 24 * 60 * 60

def expiration_time (submission_time : Nat) (expiration_seconds : Nat) : Nat :=
  (submission_time + expiration_seconds) % seconds_in_day

theorem specimen_expiration_time :
  let submission_time : Nat := 15 * 60 * 60  -- 3 PM in seconds
  let expiration_seconds : Nat := 7 * 6 * 5 * 4 * 3 * 2 * 1  -- 7!
  expiration_time submission_time expiration_seconds = 16 * 60 * 60 + 24 * 60  -- 4:24 PM in seconds
  := by sorry

end specimen_expiration_time_l3992_399240


namespace cycle_price_calculation_l3992_399213

/-- Proves that a cycle sold at a 25% loss for 1050 had an original price of 1400 -/
theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1050)
  (h2 : loss_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end cycle_price_calculation_l3992_399213


namespace consumption_wage_ratio_l3992_399247

/-- Linear regression equation parameters -/
def a : ℝ := 0.6
def b : ℝ := 1.5

/-- Average consumption per capita -/
def y : ℝ := 7.5

/-- Theorem stating the ratio of average consumption to average wage -/
theorem consumption_wage_ratio :
  ∃ x : ℝ, y = a * x + b ∧ y / x = 0.75 := by
  sorry

end consumption_wage_ratio_l3992_399247


namespace consecutive_integers_sum_l3992_399268

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 990) → 
  ((a + 2) + (b + 2) + (c + 2) = 36) := by
sorry

end consecutive_integers_sum_l3992_399268


namespace population_growth_l3992_399296

/-- Given an initial population that increases by 10% annually for 2 years
    resulting in 14,520 people, prove that the initial population was 12,000. -/
theorem population_growth (P : ℝ) : 
  (P * (1 + 0.1)^2 = 14520) → P = 12000 := by
  sorry

end population_growth_l3992_399296


namespace perpendicular_planes_perpendicular_lines_parallel_l3992_399295

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (planePerpendicular : Plane → Plane → Prop)

-- Theorem 1: If a line is perpendicular to a plane and contained in another plane,
-- then the two planes are perpendicular
theorem perpendicular_planes
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : contains β a) :
  planePerpendicular α β :=
sorry

-- Theorem 2: If two lines are perpendicular to the same plane,
-- then the lines are parallel
theorem perpendicular_lines_parallel
  (a b : Line) (α : Plane)
  (h1 : perpendicular a α)
  (h2 : perpendicular b α) :
  parallel a b :=
sorry

end perpendicular_planes_perpendicular_lines_parallel_l3992_399295


namespace intersection_perpendicular_tangents_l3992_399284

theorem intersection_perpendicular_tangents (a : ℝ) (h : a > 0) : 
  ∃ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ 
  (2 * Real.sin x = a * Real.cos x) ∧
  (2 * Real.cos x) * (-a * Real.sin x) = -1 
  → a = 2 * Real.sqrt 3 / 3 := by
sorry

end intersection_perpendicular_tangents_l3992_399284


namespace jane_max_tickets_l3992_399232

/-- The cost of a single ticket -/
def ticket_cost : ℕ := 18

/-- Jane's available money -/
def jane_money : ℕ := 150

/-- The number of tickets required for a discount -/
def discount_threshold : ℕ := 5

/-- The discount rate as a fraction -/
def discount_rate : ℚ := 1 / 10

/-- Calculate the cost of n tickets with possible discount -/
def cost_with_discount (n : ℕ) : ℚ :=
  if n ≤ discount_threshold then
    n * ticket_cost
  else
    discount_threshold * ticket_cost + (n - discount_threshold) * ticket_cost * (1 - discount_rate)

/-- The maximum number of tickets Jane can buy -/
def max_tickets : ℕ := 8

/-- Theorem stating the maximum number of tickets Jane can buy -/
theorem jane_max_tickets :
  ∀ n : ℕ, cost_with_discount n ≤ jane_money ↔ n ≤ max_tickets :=
by sorry

end jane_max_tickets_l3992_399232


namespace arithmetic_sequence_first_term_l3992_399293

/-- An arithmetic sequence with a₂ = -5 and common difference d = 3 has a₁ = -8 -/
theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a (n + 1) = a n + d) →  -- Definition of arithmetic sequence
  a 2 = -5 →                    -- Given: a₂ = -5
  d = 3 →                       -- Given: d = 3
  a 1 = -8 :=                   -- Prove: a₁ = -8
by sorry

end arithmetic_sequence_first_term_l3992_399293


namespace pyramid_edges_l3992_399288

/-- Represents a pyramid with a polygonal base. -/
structure Pyramid where
  base_sides : ℕ
  deriving Repr

/-- The number of faces in a pyramid. -/
def num_faces (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of vertices in a pyramid. -/
def num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of edges in a pyramid. -/
def num_edges (p : Pyramid) : ℕ := p.base_sides + p.base_sides

/-- Theorem: A pyramid with 16 faces and vertices combined has 14 edges. -/
theorem pyramid_edges (p : Pyramid) : 
  num_faces p + num_vertices p = 16 → num_edges p = 14 := by
  sorry

end pyramid_edges_l3992_399288


namespace new_alloy_aluminum_bounds_l3992_399221

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  aluminum : ℝ
  copper : ℝ
  magnesium : ℝ

/-- Given three alloys and their compositions, proves that a new alloy with 20% copper
    made from these alloys will have an aluminum percentage between 15% and 40% -/
theorem new_alloy_aluminum_bounds 
  (alloy1 : AlloyComposition)
  (alloy2 : AlloyComposition)
  (alloy3 : AlloyComposition)
  (h1 : alloy1.aluminum = 0.6 ∧ alloy1.copper = 0.15 ∧ alloy1.magnesium = 0.25)
  (h2 : alloy2.aluminum = 0 ∧ alloy2.copper = 0.3 ∧ alloy2.magnesium = 0.7)
  (h3 : alloy3.aluminum = 0.45 ∧ alloy3.copper = 0 ∧ alloy3.magnesium = 0.55)
  : ∃ (x1 x2 x3 : ℝ), 
    x1 + x2 + x3 = 1 ∧
    0.15 * x1 + 0.3 * x2 = 0.2 ∧
    0.15 ≤ 0.6 * x1 + 0.45 * x3 ∧
    0.6 * x1 + 0.45 * x3 ≤ 0.4 :=
by sorry

end new_alloy_aluminum_bounds_l3992_399221


namespace two_consecutive_late_charges_l3992_399285

theorem two_consecutive_late_charges (original_bill : ℝ) (late_charge_rate : ℝ) : 
  original_bill = 400 → 
  late_charge_rate = 0.01 → 
  original_bill * (1 + late_charge_rate)^2 = 408.04 := by
sorry


end two_consecutive_late_charges_l3992_399285


namespace kola_solution_water_percentage_l3992_399250

/-- Proves that the initial water percentage in a kola solution was 64% -/
theorem kola_solution_water_percentage :
  let initial_volume : ℝ := 340
  let initial_kola_percentage : ℝ := 9
  let added_sugar : ℝ := 3.2
  let added_water : ℝ := 8
  let added_kola : ℝ := 6.8
  let final_sugar_percentage : ℝ := 26.536312849162012
  let initial_water_percentage : ℝ := 64
  let initial_sugar_percentage : ℝ := 91 - initial_water_percentage - initial_kola_percentage
  let final_volume : ℝ := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_volume : ℝ := (initial_sugar_percentage / 100) * initial_volume + added_sugar
  final_sugar_volume / final_volume * 100 = final_sugar_percentage :=
by
  sorry


end kola_solution_water_percentage_l3992_399250


namespace binomial_divisibility_l3992_399231

theorem binomial_divisibility (n k : ℕ) (h_k : k > 1) :
  (∀ m : ℕ, 1 ≤ m ∧ m < n → k ∣ Nat.choose n m) ↔
  ∃ (p : ℕ) (t : ℕ+), Nat.Prime p ∧ n = p ^ (t : ℕ) ∧ k = p :=
sorry

end binomial_divisibility_l3992_399231


namespace company_growth_rate_l3992_399214

/-- Represents the yearly capital growth rate as a real number between 0 and 1 -/
def yearly_growth_rate : ℝ := sorry

/-- The initial loan amount in ten thousands of yuan -/
def initial_loan : ℝ := 200

/-- The loan duration in years -/
def loan_duration : ℕ := 2

/-- The annual interest rate as a real number between 0 and 1 -/
def interest_rate : ℝ := 0.08

/-- The surplus after repayment in ten thousands of yuan -/
def surplus : ℝ := 72

theorem company_growth_rate :
  (initial_loan * (1 + yearly_growth_rate) ^ loan_duration) =
  (initial_loan * (1 + interest_rate) ^ loan_duration + surplus) ∧
  yearly_growth_rate = 0.2 := by sorry

end company_growth_rate_l3992_399214


namespace min_value_x_plus_y_l3992_399291

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9*x + y = x*y) :
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 9*x₀ + y₀ = x₀*y₀ ∧ x₀ + y₀ = 16 := by
  sorry

end min_value_x_plus_y_l3992_399291


namespace jackson_earned_thirty_dollars_l3992_399237

-- Define the rate of pay per hour
def pay_rate : ℝ := 5

-- Define the time spent on each chore
def vacuuming_time : ℝ := 2
def dish_washing_time : ℝ := 0.5
def bathroom_cleaning_time : ℝ := 3 * dish_washing_time

-- Define the number of times vacuuming is done
def vacuuming_repetitions : ℕ := 2

-- Calculate total chore time
def total_chore_time : ℝ :=
  vacuuming_time * vacuuming_repetitions + dish_washing_time + bathroom_cleaning_time

-- Calculate earned money
def earned_money : ℝ := total_chore_time * pay_rate

-- Theorem statement
theorem jackson_earned_thirty_dollars :
  earned_money = 30 :=
by sorry

end jackson_earned_thirty_dollars_l3992_399237


namespace shortest_distance_circle_to_line_l3992_399255

/-- The shortest distance from a circle to a line --/
theorem shortest_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + (y + 3)^2 = 9}
  let line := {(x, y) : ℝ × ℝ | y = x}
  (∃ (d : ℝ), d = 3 * (Real.sqrt 2 - 1) ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      d ≤ Real.sqrt ((p.1 - p.2)^2 / 2)) :=
by
  sorry

end shortest_distance_circle_to_line_l3992_399255


namespace intersection_theorem_l3992_399267

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in parametric form -/
structure Line where
  x₀ : ℝ
  α : ℝ

/-- Represents a parabola -/
structure Parabola where
  p : ℝ

/-- Returns true if the given point lies on the line -/
def pointOnLine (point : Point) (line : Line) (t : ℝ) : Prop :=
  point.x = line.x₀ + t * Real.cos line.α ∧ point.y = t * Real.sin line.α

/-- Returns true if the given point lies on the parabola -/
def pointOnParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Main theorem -/
theorem intersection_theorem (line : Line) (parabola : Parabola) 
    (h_p : parabola.p > 0) :
  ∃ (A B : Point) (x₁ x₂ : ℝ),
    (∃ t₁, pointOnLine A line t₁ ∧ pointOnParabola A parabola) ∧
    (∃ t₂, pointOnLine B line t₂ ∧ pointOnParabola B parabola) ∧
    A.x = x₁ ∧ B.x = x₂ →
    (line.x₀^2 = x₁ * x₂) ∧
    (A.x * B.x + A.y * B.y = 0 → line.x₀ = 2 * parabola.p) := by
  sorry

end intersection_theorem_l3992_399267


namespace min_wire_length_for_specific_parallelepiped_l3992_399222

/-- The minimum length of wire needed to construct a rectangular parallelepiped -/
def wire_length (width length height : ℝ) : ℝ :=
  4 * (width + length + height)

/-- Theorem stating the minimum wire length for a specific rectangular parallelepiped -/
theorem min_wire_length_for_specific_parallelepiped :
  wire_length 10 8 5 = 92 := by
  sorry

end min_wire_length_for_specific_parallelepiped_l3992_399222


namespace count_six_digit_permutations_l3992_399235

/-- The number of different positive six-digit integers that can be formed using the digits 2, 2, 5, 5, 9, and 9 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive six-digit integers
    that can be formed using the digits 2, 2, 5, 5, 9, and 9 is equal to 90 -/
theorem count_six_digit_permutations :
  six_digit_permutations = 90 := by
  sorry

end count_six_digit_permutations_l3992_399235


namespace cylinder_ellipse_intersection_l3992_399292

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by a plane intersecting a cylinder -/
structure Ellipse where
  majorAxis : ℝ
  minorAxis : ℝ

/-- The theorem stating the relationship between the cylinder and the ellipse -/
theorem cylinder_ellipse_intersection
  (c : RightCircularCylinder)
  (e : Ellipse)
  (h1 : c.radius = 3)
  (h2 : e.minorAxis = 2 * c.radius)
  (h3 : e.majorAxis = e.minorAxis * 1.6)
  : e.majorAxis = 9.6 := by
  sorry

end cylinder_ellipse_intersection_l3992_399292


namespace tetrahedron_similarity_counterexample_l3992_399283

/-- A tetrahedron with equilateral triangle base and three other sides --/
structure Tetrahedron :=
  (base : ℝ)
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Two triangular faces are similar --/
def similar_faces (t1 t2 : Tetrahedron) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    ((t1.side1 = k * t2.side1 ∧ t1.side2 = k * t2.side2) ∨
     (t1.side1 = k * t2.side2 ∧ t1.side2 = k * t2.side3) ∨
     (t1.side1 = k * t2.side3 ∧ t1.side2 = k * t2.base) ∨
     (t1.side2 = k * t2.side1 ∧ t1.side3 = k * t2.side2) ∨
     (t1.side2 = k * t2.side2 ∧ t1.side3 = k * t2.side3) ∨
     (t1.side2 = k * t2.side3 ∧ t1.side3 = k * t2.base) ∨
     (t1.side3 = k * t2.side1 ∧ t1.base = k * t2.side2) ∨
     (t1.side3 = k * t2.side2 ∧ t1.base = k * t2.side3) ∨
     (t1.side3 = k * t2.side3 ∧ t1.base = k * t2.base))

/-- Two tetrahedrons are similar --/
def similar_tetrahedrons (t1 t2 : Tetrahedron) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    t1.base = k * t2.base ∧
    t1.side1 = k * t2.side1 ∧
    t1.side2 = k * t2.side2 ∧
    t1.side3 = k * t2.side3

/-- The main theorem --/
theorem tetrahedron_similarity_counterexample :
  ∃ (t1 t2 : Tetrahedron),
    (∀ (f1 f2 : Tetrahedron → Tetrahedron → Prop),
      (f1 t1 t1 → f2 t1 t1 → f1 = f2) ∧
      (f1 t2 t2 → f2 t2 t2 → f1 = f2)) ∧
    (∀ (f1 : Tetrahedron → Tetrahedron → Prop),
      (f1 t1 t1 → ∃ (f2 : Tetrahedron → Tetrahedron → Prop), f2 t2 t2 ∧ f1 = f2)) ∧
    ¬(similar_tetrahedrons t1 t2) :=
sorry

end tetrahedron_similarity_counterexample_l3992_399283


namespace negation_of_existence_negation_of_proposition_l3992_399282

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ (∀ x > 1, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 1 > 0) ↔ (∀ x > 1, x^2 - 1 ≤ 0) :=
by sorry

end negation_of_existence_negation_of_proposition_l3992_399282


namespace binary_search_upper_bound_l3992_399226

theorem binary_search_upper_bound (n : ℕ) (h : n ≤ 100) :
  ∃ (k : ℕ), k ≤ 7 ∧ 2^k > n :=
sorry

end binary_search_upper_bound_l3992_399226


namespace least_addition_for_divisibility_l3992_399248

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, (1202 + m) % 4 = 0 → n ≤ m) ∧ (1202 + n) % 4 = 0 → n = 2 := by
  sorry

end least_addition_for_divisibility_l3992_399248


namespace ten_millions_count_hundred_thousands_count_l3992_399269

/-- Represents the progression rate between adjacent counting units -/
def progression_rate : ℕ := 10

/-- The number of ten millions in one hundred million -/
def ten_millions_in_hundred_million : ℕ := progression_rate

/-- The number of hundred thousands in one million -/
def hundred_thousands_in_million : ℕ := progression_rate

/-- Theorem stating the number of ten millions in one hundred million is 10 -/
theorem ten_millions_count : ten_millions_in_hundred_million = 10 := by sorry

/-- Theorem stating the number of hundred thousands in one million is 10 -/
theorem hundred_thousands_count : hundred_thousands_in_million = 10 := by sorry

end ten_millions_count_hundred_thousands_count_l3992_399269


namespace problem_statement_l3992_399244

theorem problem_statement (n : ℝ) : 
  (n - 2009)^2 + (2008 - n)^2 = 1 → (n - 2009) * (2008 - n) = 0 := by
  sorry

end problem_statement_l3992_399244


namespace range_of_m_l3992_399280

-- Define the set of real numbers between 1 and 2
def OpenInterval := {x : ℝ | 1 < x ∧ x < 2}

-- Define the inequality condition
def InequalityCondition (m : ℝ) : Prop :=
  ∀ x ∈ OpenInterval, x^2 + m*x + 2 ≥ 0

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (InequalityCondition m) ↔ m ≥ -2 * Real.sqrt 2 :=
by sorry

end range_of_m_l3992_399280


namespace cost_of_oil_l3992_399262

/-- The cost of oil given the total cost of groceries and the costs of beef and chicken -/
theorem cost_of_oil (total_cost beef_cost chicken_cost : ℝ) : 
  total_cost = 16 → beef_cost = 12 → chicken_cost = 3 → 
  total_cost - (beef_cost + chicken_cost) = 1 := by
sorry

end cost_of_oil_l3992_399262


namespace solution_set_f_leq_x_range_of_a_l3992_399264

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 7| + 1

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | 8/3 ≤ x ∧ x ≤ 6} :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x - 2 * |x - 1| ≤ a) → a ≥ -4 :=
sorry

end solution_set_f_leq_x_range_of_a_l3992_399264


namespace man_walking_speed_l3992_399279

/-- Calculates the speed of a man walking in the same direction as a train,
    given the train's length, speed, and time to cross the man. -/
theorem man_walking_speed (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 600 →
  train_speed_kmh = 64 →
  crossing_time = 35.99712023038157 →
  ∃ (man_speed : ℝ), abs (man_speed - 1.10977777777778) < 0.00000000000001 :=
by sorry

end man_walking_speed_l3992_399279


namespace min_sum_is_two_l3992_399219

/-- Represents a sequence of five digits -/
def DigitSequence := Fin 5 → Nat

/-- Ensures all digits in the sequence are between 1 and 9 -/
def valid_sequence (s : DigitSequence) : Prop :=
  ∀ i, 1 ≤ s i ∧ s i ≤ 9

/-- Computes the sum of the last four digits in the sequence -/
def sum_last_four (s : DigitSequence) : Nat :=
  (s 1) + (s 2) + (s 3) + (s 4)

/-- Represents the evolution rule for the sequence -/
def evolve (s : DigitSequence) : DigitSequence :=
  fun i => match i with
    | 0 => s 1
    | 1 => s 2
    | 2 => s 3
    | 3 => s 4
    | 4 => sum_last_four s % 10

/-- Represents the sum of all digits in the sequence -/
def sequence_sum (s : DigitSequence) : Nat :=
  (s 0) + (s 1) + (s 2) + (s 3) + (s 4)

/-- The main theorem stating that the minimum sum is 2 -/
theorem min_sum_is_two :
  ∃ (s : DigitSequence), valid_sequence s ∧
  ∀ (n : Nat), sequence_sum (Nat.iterate evolve n s) ≥ 2 ∧
  ∃ (m : Nat), sequence_sum (Nat.iterate evolve m s) = 2 :=
sorry

end min_sum_is_two_l3992_399219


namespace kaleb_shirts_l3992_399225

theorem kaleb_shirts (initial_shirts : ℕ) (removed_shirts : ℕ) :
  initial_shirts = 17 →
  removed_shirts = 7 →
  initial_shirts - removed_shirts = 10 :=
by sorry

end kaleb_shirts_l3992_399225


namespace count_odd_rank_subsets_l3992_399208

/-- The number of cards in the deck -/
def total_cards : ℕ := 8056

/-- The number of ranks in the deck -/
def total_ranks : ℕ := 2014

/-- The number of suits per rank -/
def suits_per_rank : ℕ := 4

/-- The number of subsets with cards from an odd number of distinct ranks -/
def odd_rank_subsets : ℕ := (16^total_ranks - 14^total_ranks) / 2

/-- Theorem stating the number of subsets with cards from an odd number of distinct ranks -/
theorem count_odd_rank_subsets :
  total_cards = total_ranks * suits_per_rank →
  odd_rank_subsets = (16^total_ranks - 14^total_ranks) / 2 :=
by
  sorry

end count_odd_rank_subsets_l3992_399208


namespace comic_book_stacking_l3992_399246

theorem comic_book_stacking (batman : ℕ) (xmen : ℕ) (calvin_hobbes : ℕ) :
  batman = 5 →
  xmen = 4 →
  calvin_hobbes = 3 →
  (Nat.factorial batman * Nat.factorial xmen * Nat.factorial calvin_hobbes) *
  Nat.factorial 3 = 103680 :=
by sorry

end comic_book_stacking_l3992_399246


namespace exists_triangular_face_l3992_399294

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- A face of a polyhedron is one of the flat polygonal surfaces that make up its boundary. -/
structure Face (P : ConvexPolyhedron) where
  -- Again, we just declare it exists without full definition
  dummy : Unit

/-- An edge of a polyhedron is a line segment where two faces meet. -/
structure Edge (P : ConvexPolyhedron) where
  dummy : Unit

/-- A vertex of a polyhedron is a point where three or more edges meet. -/
structure Vertex (P : ConvexPolyhedron) where
  dummy : Unit

/-- The number of edges meeting at a vertex. -/
def edgesAtVertex (P : ConvexPolyhedron) (v : Vertex P) : ℕ :=
  sorry -- Definition not provided, but assumed to exist

/-- Predicate to check if a face is triangular. -/
def isTriangular (P : ConvexPolyhedron) (f : Face P) : Prop :=
  sorry -- Definition not provided, but assumed to exist

/-- Theorem stating that if at least four edges meet at each vertex of a convex polyhedron,
    then at least one of its faces is a triangle. -/
theorem exists_triangular_face (P : ConvexPolyhedron)
  (h : ∀ (v : Vertex P), edgesAtVertex P v ≥ 4) :
  ∃ (f : Face P), isTriangular P f := by
  sorry

end exists_triangular_face_l3992_399294


namespace cos_difference_formula_l3992_399217

theorem cos_difference_formula (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5/8 := by
  sorry

end cos_difference_formula_l3992_399217


namespace like_terms_sum_l3992_399272

/-- Given that x^(n+1)y^3 and (1/3)x^3y^(m-1) are like terms, prove that m + n = 6 -/
theorem like_terms_sum (m n : ℤ) : 
  (∃ (x y : ℝ), x^(n+1) * y^3 = (1/3) * x^3 * y^(m-1)) → m + n = 6 := by
  sorry

end like_terms_sum_l3992_399272


namespace water_mixture_percentage_l3992_399241

/-- Given a mixture of wine and water, calculate the new percentage of water after adding more water. -/
theorem water_mixture_percentage
  (total_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (h1 : total_volume = 120)
  (h2 : initial_water_percentage = 20)
  (h3 : added_water = 8) :
  let initial_water := total_volume * (initial_water_percentage / 100)
  let new_water := initial_water + added_water
  let new_total := total_volume + added_water
  new_water / new_total * 100 = 25 := by
  sorry

end water_mixture_percentage_l3992_399241


namespace max_x_in_grid_l3992_399253

/-- Represents a 5x5 grid with X placements -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if three X's are in a row (horizontally, vertically, or diagonally) -/
def has_three_in_row (g : Grid) : Prop := sorry

/-- Checks if each row has at least one X -/
def each_row_has_x (g : Grid) : Prop := sorry

/-- Counts the number of X's in the grid -/
def count_x (g : Grid) : Nat := sorry

/-- Theorem: The maximum number of X's in a 5x5 grid without three in a row and at least one X per row is 10 -/
theorem max_x_in_grid : 
  ∀ g : Grid, 
  ¬has_three_in_row g → 
  each_row_has_x g → 
  count_x g ≤ 10 ∧ 
  ∃ g' : Grid, ¬has_three_in_row g' ∧ each_row_has_x g' ∧ count_x g' = 10 := by
  sorry

end max_x_in_grid_l3992_399253


namespace fraction_addition_l3992_399215

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l3992_399215


namespace magnitude_of_complex_number_l3992_399274

theorem magnitude_of_complex_number (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 1/w = s) : 
  Complex.abs w = 1 := by
sorry

end magnitude_of_complex_number_l3992_399274


namespace magnitude_of_vector_sum_l3992_399204

/-- The magnitude of the sum of vectors (1, √3) and (-2, 0) is 2 -/
theorem magnitude_of_vector_sum : 
  let a : Fin 2 → ℝ := ![1, Real.sqrt 3]
  let b : Fin 2 → ℝ := ![-2, 0]
  Real.sqrt ((a 0 + b 0)^2 + (a 1 + b 1)^2) = 2 := by
  sorry

end magnitude_of_vector_sum_l3992_399204


namespace quadratic_function_value_l3992_399228

/-- A quadratic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

/-- The derivative of f with respect to x -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 4 * x - m

theorem quadratic_function_value (m : ℝ) :
  (∀ x ≥ -2, f_deriv m x ≥ 0) → f m 1 = 15 := by
  sorry

end quadratic_function_value_l3992_399228


namespace base_conversion_equality_l3992_399211

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_equality : 
  toBase7 ((107 + 93) - 47) = [3, 0, 6] :=
sorry

end base_conversion_equality_l3992_399211


namespace min_value_theorem_l3992_399243

theorem min_value_theorem (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq : p + q + r + s + t + u = 11) :
  (3/p + 12/q + 27/r + 48/s + 75/t + 108/u) ≥ 819/11 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ), 
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 11 ∧
    (3/p' + 12/q' + 27/r' + 48/s' + 75/t' + 108/u') = 819/11 :=
by sorry

end min_value_theorem_l3992_399243
