import Mathlib

namespace arson_sentence_calculation_l3791_379190

/-- Calculates the sentence for each arson count given the total sentence and other crime details. -/
theorem arson_sentence_calculation (total_sentence : ℕ) (arson_counts : ℕ) (burglary_charges : ℕ) 
  (burglary_sentence : ℕ) (petty_larceny_ratio : ℕ) (petty_larceny_sentence_fraction : ℚ) :
  total_sentence = 216 →
  arson_counts = 3 →
  burglary_charges = 2 →
  burglary_sentence = 18 →
  petty_larceny_ratio = 6 →
  petty_larceny_sentence_fraction = 1/3 →
  ∃ (arson_sentence : ℕ),
    arson_sentence = 36 ∧
    total_sentence = arson_counts * arson_sentence + 
                     burglary_charges * burglary_sentence +
                     petty_larceny_ratio * burglary_charges * (petty_larceny_sentence_fraction * burglary_sentence) :=
by
  sorry


end arson_sentence_calculation_l3791_379190


namespace company_survey_problem_l3791_379106

/-- The number of employees who do not use social networks -/
def non_social_users : ℕ := 40

/-- The proportion of social network users who use VKontakte -/
def vk_users_ratio : ℚ := 3/4

/-- The proportion of social network users who use both VKontakte and Odnoklassniki -/
def both_users_ratio : ℚ := 13/20

/-- The proportion of total employees who use Odnoklassniki -/
def ok_users_ratio : ℚ := 5/6

/-- The total number of employees in the company -/
def total_employees : ℕ := 540

theorem company_survey_problem :
  ∃ (N : ℕ),
    N = total_employees ∧
    (N - non_social_users : ℚ) * (vk_users_ratio + (1 - vk_users_ratio)) = N * ok_users_ratio :=
sorry

end company_survey_problem_l3791_379106


namespace stone_slab_floor_area_l3791_379154

/-- Calculates the total floor area covered by square stone slabs -/
theorem stone_slab_floor_area 
  (num_slabs : ℕ) 
  (slab_length : ℝ) 
  (h_num_slabs : num_slabs = 30)
  (h_slab_length : slab_length = 140) : 
  (num_slabs * slab_length^2) / 10000 = 58.8 := by
  sorry

end stone_slab_floor_area_l3791_379154


namespace laura_debt_after_one_year_l3791_379181

/-- Calculates the total amount owed after applying simple interest -/
def totalAmountOwed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Laura's debt after one year -/
theorem laura_debt_after_one_year :
  totalAmountOwed 35 0.05 1 = 36.75 := by
  sorry

end laura_debt_after_one_year_l3791_379181


namespace books_count_l3791_379113

/-- The number of books Darryl has -/
def darryl_books : ℕ := 20

/-- The number of books Lamont has -/
def lamont_books : ℕ := 2 * darryl_books

/-- The number of books Loris has -/
def loris_books : ℕ := lamont_books - 3

/-- The total number of books all three have -/
def total_books : ℕ := darryl_books + lamont_books + loris_books

theorem books_count : total_books = 97 := by
  sorry

end books_count_l3791_379113


namespace remainder_theorem_l3791_379158

theorem remainder_theorem (x : ℝ) : 
  let R := fun x => 3^125 * x - 2^125 * x + 2^125 - 2 * 3^125
  let divisor := fun x => x^2 - 5*x + 6
  ∃ Q : ℝ → ℝ, x^125 = Q x * divisor x + R x ∧ 
  (∀ a b : ℝ, R x = a * x + b → (a = 3^125 - 2^125 ∧ b = 2^125 - 2 * 3^125)) :=
by sorry

end remainder_theorem_l3791_379158


namespace weight_swap_l3791_379162

structure Weight :=
  (value : ℝ)

def WeighingScale (W X Y Z : Weight) : Prop :=
  (Z.value > Y.value) ∧
  (X.value > W.value) ∧
  (Y.value + Z.value > W.value + X.value) ∧
  (Z.value > W.value)

theorem weight_swap (W X Y Z : Weight) 
  (h : WeighingScale W X Y Z) : 
  (W.value + X.value > Y.value + Z.value) → 
  (Z.value + X.value > Y.value + W.value) :=
sorry

end weight_swap_l3791_379162


namespace water_removal_for_concentration_l3791_379155

theorem water_removal_for_concentration (initial_volume : ℝ) (final_concentration : ℝ) 
  (water_removed : ℝ) : 
  initial_volume = 21 ∧ 
  final_concentration = 60 ∧ 
  water_removed = 7 → 
  water_removed = initial_volume - (initial_volume * (initial_volume * final_concentration) / 
    (100 * (initial_volume - water_removed))) / 100 :=
by sorry

end water_removal_for_concentration_l3791_379155


namespace one_third_of_eleven_y_plus_three_l3791_379104

theorem one_third_of_eleven_y_plus_three (y : ℝ) : (1 / 3) * (11 * y + 3) = (11 * y / 3) + 1 := by
  sorry

end one_third_of_eleven_y_plus_three_l3791_379104


namespace robert_nickel_difference_l3791_379115

/-- Represents the number of chocolates eaten by each person -/
structure Chocolates where
  sarah : ℕ
  nickel : ℕ
  robert : ℕ

/-- The chocolate eating scenario -/
def chocolate_scenario : Chocolates :=
  { sarah := 15,
    nickel := 15 - 5,
    robert := 2 * (15 - 5) }

/-- Theorem stating the difference between Robert's and Nickel's chocolates -/
theorem robert_nickel_difference :
  chocolate_scenario.robert - chocolate_scenario.nickel = 10 := by
  sorry

end robert_nickel_difference_l3791_379115


namespace discount_effect_l3791_379173

/-- Represents the sales discount as a percentage -/
def discount : ℝ := 10

/-- Represents the increase in the number of items sold as a percentage -/
def items_increase : ℝ := 30

/-- Represents the increase in gross income as a percentage -/
def income_increase : ℝ := 17

theorem discount_effect (P N : ℝ) (h₁ : P > 0) (h₂ : N > 0) : 
  P * (1 - discount / 100) * N * (1 + items_increase / 100) = 
  P * N * (1 + income_increase / 100) := by
  sorry

#check discount_effect

end discount_effect_l3791_379173


namespace average_weight_increase_l3791_379146

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 62 →
  new_weight = 90 →
  (new_weight - old_weight) / initial_count = 3.5 :=
by
  sorry

end average_weight_increase_l3791_379146


namespace playlist_repetitions_l3791_379175

def song1_duration : ℕ := 3
def song2_duration : ℕ := 2
def song3_duration : ℕ := 3
def ride_duration : ℕ := 40

def playlist_duration : ℕ := song1_duration + song2_duration + song3_duration

theorem playlist_repetitions :
  ride_duration / playlist_duration = 5 := by sorry

end playlist_repetitions_l3791_379175


namespace parabola_symmetry_l3791_379143

/-- A parabola with vertex form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on a parabola -/
def IsOnParabola (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * (p.x - para.h)^2 + para.k

theorem parabola_symmetry (para : Parabola) (m : ℝ) :
  let A : Point := { x := -1, y := 4 }
  let B : Point := { x := m, y := 4 }
  (IsOnParabola A para ∧ IsOnParabola B para) → m = 7 := by
  sorry

end parabola_symmetry_l3791_379143


namespace gcd_of_720_120_168_l3791_379149

theorem gcd_of_720_120_168 : Nat.gcd 720 (Nat.gcd 120 168) = 24 := by
  sorry

end gcd_of_720_120_168_l3791_379149


namespace modulus_of_complex_fraction_l3791_379112

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l3791_379112


namespace f_max_value_l3791_379116

/-- The quadratic function f(x) = -2x^2 - 5 -/
def f (x : ℝ) : ℝ := -2 * x^2 - 5

/-- The maximum value of f(x) is -5 -/
theorem f_max_value : ∃ (M : ℝ), M = -5 ∧ ∀ x, f x ≤ M := by sorry

end f_max_value_l3791_379116


namespace units_digit_of_product_l3791_379103

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product :
  units_digit (17 * 28) = 6 := by
  sorry

end units_digit_of_product_l3791_379103


namespace min_of_quadratic_l3791_379180

/-- The quadratic function f(x) = x^2 + px + 2q -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + 2*q

/-- Theorem stating that the minimum of f occurs at x = -p/2 -/
theorem min_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∀ x : ℝ, f p q (-p/2) ≤ f p q x :=
sorry

end min_of_quadratic_l3791_379180


namespace hyperbola_equation_l3791_379161

/-- The standard equation of a hyperbola with given foci and asymptotes -/
theorem hyperbola_equation (c : ℝ) (m : ℝ) :
  c = Real.sqrt 10 →
  m = 1 / 2 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1 ↔
      ((x + c)^2 + y^2)^(1/2) - ((x - c)^2 + y^2)^(1/2) = 2*a)) ∧
    (∀ (x : ℝ), y = m*x ∨ y = -m*x ↔ x^2 / a^2 - y^2 / b^2 = 0) ∧
    a^2 = 8 ∧ b^2 = 2 :=
sorry

end hyperbola_equation_l3791_379161


namespace tangent_intersection_y_coordinate_l3791_379117

noncomputable def curve (x : ℝ) : ℝ := x^3

theorem tangent_intersection_y_coordinate 
  (a b : ℝ) 
  (hA : ∃ y, y = curve a) 
  (hB : ∃ y, y = curve b) 
  (h_perp : (3 * a^2) * (3 * b^2) = -1) :
  ∃ x y, y = -1/3 ∧ 
    y = 3 * a^2 * (x - a) + a^3 ∧ 
    y = 3 * b^2 * (x - b) + b^3 :=
sorry

end tangent_intersection_y_coordinate_l3791_379117


namespace red_boxcars_count_l3791_379198

/-- The number of blue boxcars -/
def num_blue_boxcars : ℕ := 4

/-- The number of black boxcars -/
def num_black_boxcars : ℕ := 7

/-- The capacity of a black boxcar in pounds -/
def black_boxcar_capacity : ℕ := 4000

/-- The capacity of a blue boxcar in pounds -/
def blue_boxcar_capacity : ℕ := 2 * black_boxcar_capacity

/-- The capacity of a red boxcar in pounds -/
def red_boxcar_capacity : ℕ := 3 * blue_boxcar_capacity

/-- The total capacity of all boxcars in pounds -/
def total_capacity : ℕ := 132000

/-- The number of red boxcars -/
def num_red_boxcars : ℕ := 
  (total_capacity - num_black_boxcars * black_boxcar_capacity - num_blue_boxcars * blue_boxcar_capacity) / red_boxcar_capacity

theorem red_boxcars_count : num_red_boxcars = 3 := by
  sorry

end red_boxcars_count_l3791_379198


namespace bicycle_owners_without_scooters_l3791_379179

theorem bicycle_owners_without_scooters (total : ℕ) (bicycle_owners : ℕ) (scooter_owners : ℕ) 
  (h_total : total = 500)
  (h_bicycle : bicycle_owners = 485)
  (h_scooter : scooter_owners = 150)
  (h_subset : bicycle_owners + scooter_owners ≥ total) :
  bicycle_owners - (bicycle_owners + scooter_owners - total) = 350 := by
  sorry

#check bicycle_owners_without_scooters

end bicycle_owners_without_scooters_l3791_379179


namespace complex_roots_theorem_l3791_379171

theorem complex_roots_theorem (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 4 * Complex.I) * (a + 4 * Complex.I) - (10 + 9 * Complex.I) * (a + 4 * Complex.I) + (4 + 46 * Complex.I) = 0 →
  (b + 5 * Complex.I) * (b + 5 * Complex.I) - (10 + 9 * Complex.I) * (b + 5 * Complex.I) + (4 + 46 * Complex.I) = 0 →
  a = 6 ∧ b = 4 := by
sorry

end complex_roots_theorem_l3791_379171


namespace clips_sold_and_average_earning_l3791_379172

/-- Calculates the total number of clips sold and average earning per clip -/
theorem clips_sold_and_average_earning 
  (x : ℝ) -- number of clips sold in April
  (y : ℝ) -- number of clips sold in May
  (z : ℝ) -- number of clips sold in June
  (W : ℝ) -- total earnings
  (h1 : y = x / 2) -- May sales condition
  (h2 : z = y + 0.25 * y) -- June sales condition
  : (x + y + z = 2.125 * x) ∧ (W / (x + y + z) = W / (2.125 * x)) := by
  sorry

end clips_sold_and_average_earning_l3791_379172


namespace simplified_expression_equals_zero_l3791_379102

theorem simplified_expression_equals_zero (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = x + y) : x/y + y/x - 2/(x*y) = 0 := by
  sorry

end simplified_expression_equals_zero_l3791_379102


namespace first_divisor_problem_l3791_379195

theorem first_divisor_problem (y : ℝ) (h : (320 / y) / 3 = 53.33) : y = 2 := by
  sorry

end first_divisor_problem_l3791_379195


namespace product_mod_seventeen_l3791_379165

theorem product_mod_seventeen : (1520 * 1521 * 1522) % 17 = 11 := by
  sorry

end product_mod_seventeen_l3791_379165


namespace fun_math_book_price_l3791_379128

/-- The price of the "Fun Math" book in yuan -/
def book_price : ℝ := 4

/-- The amount Xiaohong is short in yuan -/
def xiaohong_short : ℝ := 2.2

/-- The amount Xiaoming is short in yuan -/
def xiaoming_short : ℝ := 1.8

/-- Theorem stating that the book price is 4 yuan given the conditions -/
theorem fun_math_book_price :
  (book_price - xiaohong_short) + (book_price - xiaoming_short) = book_price :=
by sorry

end fun_math_book_price_l3791_379128


namespace polynomial_expansion_l3791_379186

theorem polynomial_expansion (t : ℝ) : 
  (2*t^2 - 3*t + 2) * (-3*t^2 + t - 5) = -6*t^4 + 11*t^3 - 19*t^2 + 17*t - 10 := by
  sorry

end polynomial_expansion_l3791_379186


namespace square_root_product_squared_l3791_379166

theorem square_root_product_squared : (Real.sqrt 900 * Real.sqrt 784)^2 = 705600 := by
  sorry

end square_root_product_squared_l3791_379166


namespace gas_station_distance_l3791_379185

theorem gas_station_distance (x : ℝ) : 
  (¬ (x ≥ 10)) →   -- Adam's statement is false
  (¬ (x ≤ 7)) →    -- Betty's statement is false
  (¬ (x < 5)) →    -- Carol's statement is false
  (¬ (x ≤ 9)) →    -- Dave's statement is false
  x > 9 :=         -- Conclusion: x is in the interval (9, ∞)
by
  sorry            -- Proof is omitted

#check gas_station_distance

end gas_station_distance_l3791_379185


namespace shortest_reflected_light_path_l3791_379111

/-- The shortest path length for a reflected light ray -/
theorem shortest_reflected_light_path :
  let A : ℝ × ℝ := (-3, 9)
  let C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}
  let reflect_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (p : ℝ × ℝ),
    p.2 = 0 ∧
    p ∉ C ∧
    (∀ (q : ℝ × ℝ), q.2 = 0 ∧ q ∉ C →
      dist A p + dist p (reflect_point (2, 3)) ≤ dist A q + dist q (reflect_point (2, 3))) ∧
    dist A p + dist p (reflect_point (2, 3)) = 12 :=
by sorry

end shortest_reflected_light_path_l3791_379111


namespace quadratic_inequality_l3791_379139

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : 1 < x₁) (h₂ : x₁ < 2) (h₃ : 3 < x₂) (h₄ : x₂ < 4)
  (hy₁ : y₁ = f x₁) (hy₂ : y₂ = f x₂) : y₁ < y₂ := by
  sorry

end quadratic_inequality_l3791_379139


namespace gravel_bags_per_truckload_l3791_379174

/-- Represents the roadwork company's asphalt paving problem -/
def roadwork_problem (road_length : ℕ) (gravel_pitch_ratio : ℕ) (truckloads_per_mile : ℕ)
  (day1_miles : ℕ) (day2_miles : ℕ) (remaining_pitch : ℕ) : Prop :=
  let total_paved := day1_miles + day2_miles
  let remaining_miles := road_length - total_paved
  let remaining_truckloads := remaining_miles * truckloads_per_mile
  let pitch_per_truckload : ℚ := remaining_pitch / remaining_truckloads
  let gravel_per_truckload := gravel_pitch_ratio * pitch_per_truckload
  gravel_per_truckload = 2

/-- The main theorem stating that the number of bags of gravel per truckload is 2 -/
theorem gravel_bags_per_truckload :
  roadwork_problem 16 5 3 4 7 6 :=
sorry

end gravel_bags_per_truckload_l3791_379174


namespace problem_solution_l3791_379183

theorem problem_solution (a b : ℚ) 
  (h1 : 2 * a + 3 = 5 - b) 
  (h2 : 5 + 2 * b = 10 + a) : 
  2 - a = 11 / 5 := by
sorry

end problem_solution_l3791_379183


namespace truck_distance_truck_distance_proof_l3791_379126

/-- The distance traveled by a truck given specific conditions -/
theorem truck_distance (truck_time car_time : ℝ) 
  (speed_difference distance_difference : ℝ) : ℝ :=
  let truck_speed := (car_time * speed_difference + distance_difference) / (car_time - truck_time)
  truck_speed * truck_time

/-- Prove that the truck travels 296 km under the given conditions -/
theorem truck_distance_proof : 
  truck_distance 8 5.5 18 6.5 = 296 := by
  sorry

end truck_distance_truck_distance_proof_l3791_379126


namespace restaurant_gratuities_l3791_379135

/-- Calculates the gratuities charged by a restaurant --/
theorem restaurant_gratuities
  (total_bill : ℝ)
  (sales_tax_rate : ℝ)
  (striploin_cost : ℝ)
  (wine_cost : ℝ)
  (h_total_bill : total_bill = 140)
  (h_sales_tax_rate : sales_tax_rate = 0.1)
  (h_striploin_cost : striploin_cost = 80)
  (h_wine_cost : wine_cost = 10) :
  total_bill - (striploin_cost + wine_cost) * (1 + sales_tax_rate) = 41 := by
  sorry

end restaurant_gratuities_l3791_379135


namespace x_intercept_of_line_l3791_379109

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end x_intercept_of_line_l3791_379109


namespace parabola_chord_length_l3791_379178

/-- The length of a chord passing through the focus of a parabola -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ →  -- Point A satisfies the parabola equation
  y₂^2 = 4*x₂ →  -- Point B satisfies the parabola equation
  x₁ + x₂ = 6 →  -- Given condition
  -- The line passes through the focus (1, 0) of y^2 = 4x
  ∃ (m : ℝ), y₁ = m*(x₁ - 1) ∧ y₂ = m*(x₂ - 1) →
  -- Then the length of chord AB is 8
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 8 :=
by sorry

end parabola_chord_length_l3791_379178


namespace watch_cost_price_l3791_379192

def watch_problem (cost_price : ℝ) : Prop :=
  let loss_percentage : ℝ := 10
  let gain_percentage : ℝ := 4
  let additional_amount : ℝ := 210
  let selling_price_1 : ℝ := cost_price * (1 - loss_percentage / 100)
  let selling_price_2 : ℝ := cost_price * (1 + gain_percentage / 100)
  selling_price_2 = selling_price_1 + additional_amount

theorem watch_cost_price : 
  ∃ (cost_price : ℝ), watch_problem cost_price ∧ cost_price = 1500 :=
sorry

end watch_cost_price_l3791_379192


namespace smallest_product_l3791_379197

def S : Set Int := {-10, -5, 0, 2, 4}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b ∧ x * y = -40 :=
sorry

end smallest_product_l3791_379197


namespace same_heads_probability_l3791_379193

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of ways to get k heads when tossing n pennies -/
def ways_to_get_heads (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of favorable outcomes where Keiko and Ephraim get the same number of heads -/
def favorable_outcomes : ℕ :=
  (ways_to_get_heads keiko_pennies 0 * ways_to_get_heads ephraim_pennies 0) +
  (ways_to_get_heads keiko_pennies 1 * ways_to_get_heads ephraim_pennies 1) +
  (ways_to_get_heads keiko_pennies 2 * ways_to_get_heads ephraim_pennies 2)

/-- The probability of Ephraim getting the same number of heads as Keiko -/
theorem same_heads_probability :
  (favorable_outcomes : ℚ) / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies) = 5/16 := by
  sorry

end same_heads_probability_l3791_379193


namespace cube_root_54880000_l3791_379152

theorem cube_root_54880000 : 
  (Real.rpow 54880000 (1/3 : ℝ)) = 140 * Real.rpow 2 (1/3 : ℝ) := by sorry

end cube_root_54880000_l3791_379152


namespace florist_fertilizer_usage_l3791_379163

/-- A florist's fertilizer usage problem -/
theorem florist_fertilizer_usage 
  (daily_usage : ℝ) 
  (num_days : ℕ) 
  (total_usage : ℝ) 
  (h1 : daily_usage = 2) 
  (h2 : num_days = 9) 
  (h3 : total_usage = 22) : 
  total_usage - (daily_usage * num_days) = 4 := by
sorry

end florist_fertilizer_usage_l3791_379163


namespace even_function_implies_a_equals_one_l3791_379122

/-- A function f is even if f(x) = f(-x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = (a - 2)x^2 + (a - 1)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (a - 2) * x^2 + (a - 1) * x + 3

/-- If f(x) = (a - 2)x^2 + (a - 1)x + 3 is an even function, then a = 1 -/
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
  sorry

end even_function_implies_a_equals_one_l3791_379122


namespace circle_area_20cm_diameter_l3791_379119

/-- The area of a circle with diameter 20 cm is 314 square cm, given π = 3.14 -/
theorem circle_area_20cm_diameter (π : ℝ) (h : π = 3.14) :
  let d : ℝ := 20
  let r : ℝ := d / 2
  π * r^2 = 314 := by sorry

end circle_area_20cm_diameter_l3791_379119


namespace hexagon_side_count_l3791_379129

/-- A convex hexagon with exactly two distinct side lengths -/
structure ConvexHexagon where
  side_length1 : ℝ
  side_length2 : ℝ
  num_sides1 : ℕ
  num_sides2 : ℕ
  distinct_lengths : side_length1 ≠ side_length2
  total_sides : num_sides1 + num_sides2 = 6

/-- The perimeter of a convex hexagon -/
def perimeter (h : ConvexHexagon) : ℝ :=
  h.side_length1 * h.num_sides1 + h.side_length2 * h.num_sides2

theorem hexagon_side_count (h : ConvexHexagon)
  (side1_length : h.side_length1 = 8)
  (side2_length : h.side_length2 = 10)
  (total_perimeter : perimeter h = 56) :
  h.num_sides2 = 4 :=
sorry

end hexagon_side_count_l3791_379129


namespace function_bounds_l3791_379124

theorem function_bounds (x : ℝ) : 
  (1 : ℝ) / 2 ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ (3 : ℝ) / 2 := by
  sorry

end function_bounds_l3791_379124


namespace complex_square_simplification_l3791_379130

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end complex_square_simplification_l3791_379130


namespace dad_steps_l3791_379182

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- The ratio of steps between Dad and Masha -/
def dad_masha_ratio (s : Steps) : Prop :=
  3 * s.masha = 5 * s.dad

/-- The ratio of steps between Masha and Yasha -/
def masha_yasha_ratio (s : Steps) : Prop :=
  3 * s.yasha = 5 * s.masha

/-- The total number of steps taken by Masha and Yasha -/
def masha_yasha_total (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : masha_yasha_total s) :
  s.dad = 90 := by
  sorry

end dad_steps_l3791_379182


namespace fourth_term_of_progression_l3791_379153

-- Define the geometric progression
def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Define our specific progression
def our_progression (n : ℕ) : ℝ := 5^(1 / (5 * 2^(n - 1)))

-- Theorem statement
theorem fourth_term_of_progression :
  our_progression 4 = 5^(1/10) := by
  sorry

end fourth_term_of_progression_l3791_379153


namespace ln_intersection_and_exponential_inequality_l3791_379199

open Real

theorem ln_intersection_and_exponential_inequality (m n : ℝ) (h : m < n) :
  (∃! x : ℝ, log x = x - 1) ∧
  ((exp n - exp m) / (n - m) > exp ((m + n) / 2)) := by
  sorry

end ln_intersection_and_exponential_inequality_l3791_379199


namespace square_area_from_diagonal_l3791_379141

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  (d / Real.sqrt 2) ^ 2 = 64 := by
  sorry

end square_area_from_diagonal_l3791_379141


namespace impossibleGrid_l3791_379160

/-- Represents a 6x6 grid filled with numbers from 1 to 6 -/
def Grid := Fin 6 → Fin 6 → Fin 6

/-- The sum of numbers in a 2x2 subgrid starting at (i, j) -/
def subgridSum (g : Grid) (i j : Fin 5) : ℕ :=
  g i j + g i (j + 1) + g (i + 1) j + g (i + 1) (j + 1)

/-- A predicate that checks if all 2x2 subgrids have different sums -/
def allSubgridSumsDifferent (g : Grid) : Prop :=
  ∀ i j k l : Fin 5, (i, j) ≠ (k, l) → subgridSum g i j ≠ subgridSum g k l

theorem impossibleGrid : ¬ ∃ g : Grid, allSubgridSumsDifferent g := by
  sorry

end impossibleGrid_l3791_379160


namespace complement_of_44_36_l3791_379114

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Calculates the complement of an angle -/
def complement (α : Angle) : Angle :=
  let total_minutes := 90 * 60 - (α.degrees * 60 + α.minutes)
  { degrees := total_minutes / 60, minutes := total_minutes % 60 }

theorem complement_of_44_36 :
  let α : Angle := { degrees := 44, minutes := 36 }
  complement α = { degrees := 45, minutes := 24 } := by
  sorry

end complement_of_44_36_l3791_379114


namespace limsup_subset_l3791_379108

open Set

theorem limsup_subset {α : Type*} (A B : ℕ → Set α) (h : ∀ n, A n ⊆ B n) :
  (⋂ k, ⋃ n ≥ k, A n) ⊆ (⋂ k, ⋃ n ≥ k, B n) := by
  sorry

end limsup_subset_l3791_379108


namespace quadratic_roots_product_l3791_379100

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) → 
  (3 * q^2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = -58 := by
sorry

end quadratic_roots_product_l3791_379100


namespace cylinder_from_constant_rho_l3791_379148

/-- Cylindrical coordinates -/
structure CylindricalCoord where
  ρ : ℝ
  φ : ℝ
  z : ℝ

/-- A set of points in cylindrical coordinates -/
def CylindricalSet (c : ℝ) : Set CylindricalCoord :=
  {p : CylindricalCoord | p.ρ = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalCoord) : Prop :=
  ∃ c > 0, S = CylindricalSet c

/-- Theorem: The set of points satisfying ρ = c forms a cylinder -/
theorem cylinder_from_constant_rho (c : ℝ) (hc : c > 0) :
  IsCylinder (CylindricalSet c) := by
  sorry


end cylinder_from_constant_rho_l3791_379148


namespace total_rent_is_6500_l3791_379150

/-- Represents the grazing data for a milkman -/
structure GrazingData where
  cows : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given grazing data and one milkman's rent -/
def totalRent (a b c d : GrazingData) (aRent : ℕ) : ℕ :=
  let totalCowMonths := a.cows * a.months + b.cows * b.months + c.cows * c.months + d.cows * d.months
  let rentPerCowMonth := aRent / (a.cows * a.months)
  rentPerCowMonth * totalCowMonths

/-- Theorem stating that the total rent is 6500 given the problem conditions -/
theorem total_rent_is_6500 :
  let a : GrazingData := ⟨24, 3⟩
  let b : GrazingData := ⟨10, 5⟩
  let c : GrazingData := ⟨35, 4⟩
  let d : GrazingData := ⟨21, 3⟩
  let aRent : ℕ := 1440
  totalRent a b c d aRent = 6500 := by
  sorry

end total_rent_is_6500_l3791_379150


namespace complex_modulus_product_l3791_379184

theorem complex_modulus_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end complex_modulus_product_l3791_379184


namespace eight_digit_divisible_by_11_l3791_379196

/-- An eight-digit number in the form 8524m637 -/
def eight_digit_number (m : ℕ) : ℕ := 85240000 + m * 1000 + 637

/-- Sum of digits in odd positions -/
def sum_odd_digits (m : ℕ) : ℕ := 8 + 2 + m + 3

/-- Sum of digits in even positions -/
def sum_even_digits : ℕ := 5 + 4 + 6 + 7

/-- A number is divisible by 11 if the difference between the sum of digits in odd and even positions is a multiple of 11 -/
def divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, (sum_odd_digits n - sum_even_digits : ℤ) = 11 * k

theorem eight_digit_divisible_by_11 :
  ∃ m : ℕ, m < 10 ∧ divisible_by_11 (eight_digit_number m) ↔ m = 9 :=
sorry

end eight_digit_divisible_by_11_l3791_379196


namespace factorization_problem1_l3791_379120

theorem factorization_problem1 (x y : ℚ) : x^2 * y - 4 * x * y = x * y * (x - 4) := by sorry

end factorization_problem1_l3791_379120


namespace necessary_but_not_sufficient_l3791_379189

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) := by
  sorry

end necessary_but_not_sufficient_l3791_379189


namespace simplify_and_evaluate_find_m_value_l3791_379157

-- Part 1
theorem simplify_and_evaluate (a b : ℚ) (h1 : a = 1/2) (h2 : b = -2) :
  2 * (3 * a^2 * b - a * b^2) - 3 * (2 * a^2 * b - a * b^2 + a * b) = 5 := by sorry

-- Part 2
theorem find_m_value (a b m : ℚ) :
  (a^2 + 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = -3*b^2 → m = 2 := by sorry

end simplify_and_evaluate_find_m_value_l3791_379157


namespace mary_fruit_expenses_l3791_379105

theorem mary_fruit_expenses :
  let berries_cost : ℚ := 1108 / 100
  let apples_cost : ℚ := 1433 / 100
  let peaches_cost : ℚ := 931 / 100
  berries_cost + apples_cost + peaches_cost = 3472 / 100 := by
  sorry

end mary_fruit_expenses_l3791_379105


namespace sqrt_720_simplification_l3791_379101

theorem sqrt_720_simplification : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end sqrt_720_simplification_l3791_379101


namespace eggs_per_basket_l3791_379144

theorem eggs_per_basket (blue_eggs : Nat) (yellow_eggs : Nat) (min_eggs : Nat)
  (h1 : blue_eggs = 30)
  (h2 : yellow_eggs = 42)
  (h3 : min_eggs = 6) :
  ∃ (x : Nat), x ≥ min_eggs ∧ x ∣ blue_eggs ∧ x ∣ yellow_eggs ∧
    ∀ (y : Nat), y > x → ¬(y ∣ blue_eggs ∧ y ∣ yellow_eggs) :=
by
  sorry

end eggs_per_basket_l3791_379144


namespace race_winner_race_result_l3791_379187

theorem race_winner (race_length : ℝ) (speed_ratio_A : ℝ) (speed_ratio_B : ℝ) (head_start : ℝ) : ℝ :=
  let time_B_finish := race_length / speed_ratio_B
  let distance_A := speed_ratio_A * time_B_finish + head_start
  distance_A - race_length

theorem race_result :
  race_winner 500 3 4 140 = 15 := by sorry

end race_winner_race_result_l3791_379187


namespace total_beverage_amount_l3791_379159

/-- Given 5 bottles, each containing 242.7 ml of beverage, 
    the total amount of beverage is 1213.5 ml. -/
theorem total_beverage_amount :
  let num_bottles : ℕ := 5
  let amount_per_bottle : ℝ := 242.7
  num_bottles * amount_per_bottle = 1213.5 := by
sorry

end total_beverage_amount_l3791_379159


namespace smallest_difference_36k_5m_l3791_379194

theorem smallest_difference_36k_5m :
  (∀ k m : ℕ+, 36^k.val - 5^m.val ≥ 11) ∧
  (∃ k m : ℕ+, 36^k.val - 5^m.val = 11) :=
by sorry

end smallest_difference_36k_5m_l3791_379194


namespace same_solution_k_value_l3791_379188

theorem same_solution_k_value (x : ℝ) (k : ℝ) : 
  (2 * x = 4) ∧ (3 * x + k = -2) → k = -8 := by
  sorry

end same_solution_k_value_l3791_379188


namespace door_diagonal_equation_l3791_379132

theorem door_diagonal_equation (x : ℝ) : x ^ 2 - (x - 2) ^ 2 = (x - 4) ^ 2 := by
  sorry

end door_diagonal_equation_l3791_379132


namespace point_in_second_quadrant_l3791_379177

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -3
  let y : ℝ := 2 * Real.sqrt 2
  second_quadrant x y :=
by
  sorry

end point_in_second_quadrant_l3791_379177


namespace three_digit_number_divisible_by_6_l3791_379151

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem three_digit_number_divisible_by_6 (n : ℕ) (h1 : n ≥ 500 ∧ n < 600) 
  (h2 : n % 10 = 2) (h3 : is_divisible_by_6 n) : 
  n ≥ 100 ∧ n < 1000 :=
sorry

end three_digit_number_divisible_by_6_l3791_379151


namespace hyperbola_parabola_intersection_l3791_379133

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the latus rectum of the parabola
def latus_rectum (x : ℝ) : Prop := x = -1

-- Define the length of the line segment
def line_segment_length (b y : ℝ) : Prop := 2 * y = b

-- Main theorem
theorem hyperbola_parabola_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, hyperbola a b x y ∧ parabola x y ∧ latus_rectum x ∧ line_segment_length b y) →
  a = 2 * Real.sqrt 5 / 5 :=
sorry

end hyperbola_parabola_intersection_l3791_379133


namespace total_profit_calculation_l3791_379121

def total_subscription : ℕ := 50000
def a_extra_over_b : ℕ := 4000
def b_extra_over_c : ℕ := 5000
def a_profit : ℕ := 15120

theorem total_profit_calculation :
  ∃ (c_subscription : ℕ) (total_profit : ℕ),
    let b_subscription := c_subscription + b_extra_over_c
    let a_subscription := b_subscription + a_extra_over_b
    a_subscription + b_subscription + c_subscription = total_subscription ∧
    a_subscription * total_profit = a_profit * total_subscription ∧
    total_profit = 36000 := by
  sorry

end total_profit_calculation_l3791_379121


namespace abs_neg_three_equals_three_l3791_379107

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by
  sorry

end abs_neg_three_equals_three_l3791_379107


namespace circles_internally_tangent_l3791_379167

/-- Two circles are internally tangent if the distance between their centers
    is equal to the absolute difference of their radii -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 - r2)^2

/-- The equation of the first circle: x^2 + y^2 - 2x = 0 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The equation of the second circle: x^2 + y^2 - 2x - 6y - 6 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 6 = 0

theorem circles_internally_tangent :
  ∃ (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ),
    (∀ x y, circle1 x y ↔ (x - c1.1)^2 + (y - c1.2)^2 = r1^2) ∧
    (∀ x y, circle2 x y ↔ (x - c2.1)^2 + (y - c2.2)^2 = r2^2) ∧
    internally_tangent c1 c2 r1 r2 :=
  sorry

end circles_internally_tangent_l3791_379167


namespace train_crossing_time_l3791_379110

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 160 →
  train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4 := by
  sorry

end train_crossing_time_l3791_379110


namespace min_shift_for_trig_transformation_l3791_379176

open Real

/-- The minimum positive shift required to transform sin(2x) + √3cos(2x) into 2sin(2x) -/
theorem min_shift_for_trig_transformation : ∃ (m : ℝ), m > 0 ∧
  (∀ (x : ℝ), sin (2*x) + Real.sqrt 3 * cos (2*x) = 2 * sin (2*(x + m))) ∧
  (∀ (m' : ℝ), m' > 0 → 
    (∀ (x : ℝ), sin (2*x) + Real.sqrt 3 * cos (2*x) = 2 * sin (2*(x + m'))) → 
    m ≤ m') ∧
  m = π / 6 := by
sorry

end min_shift_for_trig_transformation_l3791_379176


namespace masks_donated_to_museum_l3791_379118

/-- Given that Alicia initially had 90 sets of masks and was left with 39 sets after donating to a museum,
    prove that she gave 51 sets to the museum. -/
theorem masks_donated_to_museum (initial_sets : ℕ) (remaining_sets : ℕ) 
    (h1 : initial_sets = 90) 
    (h2 : remaining_sets = 39) : 
  initial_sets - remaining_sets = 51 := by
  sorry

end masks_donated_to_museum_l3791_379118


namespace total_shirts_made_l3791_379145

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 6

/-- The number of minutes the machine worked yesterday -/
def minutes_yesterday : ℕ := 12

/-- The number of minutes the machine worked today -/
def minutes_today : ℕ := 14

/-- Theorem: The total number of shirts made by the machine is 156 -/
theorem total_shirts_made : 
  shirts_per_minute * minutes_yesterday + shirts_per_minute * minutes_today = 156 := by
  sorry

end total_shirts_made_l3791_379145


namespace solve_equation_l3791_379131

theorem solve_equation : 
  ∃ y : ℝ, ((10 - y)^2 = 4 * y^2) ∧ (y = 10/3 ∨ y = -10) :=
by
  sorry

end solve_equation_l3791_379131


namespace largest_n_satisfying_inequality_l3791_379127

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1 : ℚ) / 3 + (n : ℚ) / 7 < 1 ↔ n ≤ 4 :=
by sorry

end largest_n_satisfying_inequality_l3791_379127


namespace cylinder_dimensions_from_sphere_l3791_379147

/-- Given a sphere and a right circular cylinder with equal surface areas,
    prove that the height and diameter of the cylinder are both 14 cm
    when the radius of the sphere is 7 cm. -/
theorem cylinder_dimensions_from_sphere (r : ℝ) (h d : ℝ) : 
  r = 7 →  -- radius of the sphere is 7 cm
  h = d →  -- height and diameter of cylinder are equal
  4 * Real.pi * r^2 = 2 * Real.pi * (d/2) * h →  -- surface areas are equal
  h = 14 ∧ d = 14 := by
  sorry

end cylinder_dimensions_from_sphere_l3791_379147


namespace remaining_macaroons_count_l3791_379125

/-- The number of remaining macaroons after eating some -/
def remaining_macaroons (initial_red initial_green eaten_green : ℕ) : ℕ :=
  let eaten_red := 2 * eaten_green
  let remaining_red := initial_red - eaten_red
  let remaining_green := initial_green - eaten_green
  remaining_red + remaining_green

/-- Theorem stating that the number of remaining macaroons is 45 -/
theorem remaining_macaroons_count :
  remaining_macaroons 50 40 15 = 45 := by
  sorry

end remaining_macaroons_count_l3791_379125


namespace expression_range_l3791_379170

theorem expression_range (a b c : ℝ) 
  (h1 : a - b + c = 0)
  (h2 : c > 0)
  (h3 : 3 * a - 2 * b + c > 0) :
  4/3 < (a + 3*b + 7*c) / (2*a + b) ∧ (a + 3*b + 7*c) / (2*a + b) < 7/2 :=
sorry

end expression_range_l3791_379170


namespace eldest_age_is_fifteen_l3791_379137

/-- The ages of three grandchildren satisfying specific conditions -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ
  age_difference : middle - youngest = 3
  eldest_triple_youngest : eldest = 3 * youngest
  eldest_sum_plus_two : eldest = youngest + middle + 2

/-- The age of the eldest grandchild is 15 -/
theorem eldest_age_is_fifteen (ages : GrandchildrenAges) : ages.eldest = 15 := by
  sorry

end eldest_age_is_fifteen_l3791_379137


namespace dot_product_of_perpendicular_vectors_l3791_379134

/-- Given two planar vectors a and b, where a = (1, √3) and a is perpendicular to (a - b),
    prove that the dot product of a and b is 4. -/
theorem dot_product_of_perpendicular_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0 →
  a.1 * b.1 + a.2 * b.2 = 4 := by
  sorry

end dot_product_of_perpendicular_vectors_l3791_379134


namespace arithmetic_expression_equality_l3791_379191

theorem arithmetic_expression_equality : (3^2 * 5) + (7 * 4) - (42 / 3) = 59 := by
  sorry

end arithmetic_expression_equality_l3791_379191


namespace complex_equation_solution_l3791_379136

theorem complex_equation_solution (i : ℂ) (h : i * i = -1) :
  ∃ z : ℂ, (2 + i) * z = 5 ∧ z = 2 - i :=
by sorry

end complex_equation_solution_l3791_379136


namespace job_completion_time_l3791_379142

/-- Given the work rates of machines A, B, and C, prove that 15 type A machines and 7 type B machines complete the job in 4 hours. -/
theorem job_completion_time 
  (A B C : ℝ) -- Work rates of machines A, B, and C in jobs per hour
  (h1 : 15 * A + 7 * B = 1 / 4) -- 15 type A and 7 type B machines complete the job in x hours
  (h2 : 8 * B + 15 * C = 1 / 11) -- 8 type B and 15 type C machines complete the job in 11 hours
  (h3 : A + B + C = 1 / 44) -- 1 of each machine type completes the job in 44 hours
  : 15 * A + 7 * B = 1 / 4 := by
  sorry

end job_completion_time_l3791_379142


namespace ned_shirts_problem_l3791_379140

theorem ned_shirts_problem (total_shirts : ℕ) (short_sleeve : ℕ) (washed_shirts : ℕ)
  (h1 : total_shirts = 30)
  (h2 : short_sleeve = 9)
  (h3 : washed_shirts = 29) :
  total_shirts - short_sleeve - (total_shirts - washed_shirts) = 19 := by
  sorry

end ned_shirts_problem_l3791_379140


namespace nuts_cost_correct_l3791_379168

/-- The cost of nuts per kilogram -/
def cost_of_nuts : ℝ := 12

/-- The cost of dried fruits per kilogram -/
def cost_of_dried_fruits : ℝ := 8

/-- The amount of nuts bought in kilograms -/
def amount_of_nuts : ℝ := 3

/-- The amount of dried fruits bought in kilograms -/
def amount_of_dried_fruits : ℝ := 2.5

/-- The total cost of the purchase -/
def total_cost : ℝ := 56

theorem nuts_cost_correct : 
  cost_of_nuts * amount_of_nuts + cost_of_dried_fruits * amount_of_dried_fruits = total_cost :=
by sorry

end nuts_cost_correct_l3791_379168


namespace addition_puzzle_solution_l3791_379164

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Function to convert a four-digit number to its decimal representation -/
def toDecimal (a b c d : Digit) : ℕ := 1000 * a.val + 100 * b.val + 10 * c.val + d.val

/-- Predicate to check if three digits are distinct -/
def areDistinct (a b c : Digit) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem addition_puzzle_solution :
  ∃ (possibleD : Finset Digit),
    (∀ a b c d : Digit,
      areDistinct a b c →
      toDecimal a b a c + toDecimal c a b a = toDecimal d c d d →
      d ∈ possibleD) ∧
    possibleD.card = 7 := by sorry

end addition_puzzle_solution_l3791_379164


namespace find_number_l3791_379123

theorem find_number : ∃ x : ℝ, 1.2 * x = 2 * (0.8 * (x - 20)) ∧ x = 80 := by
  sorry

end find_number_l3791_379123


namespace valid_pairs_count_l3791_379138

/-- Represents a person's age --/
structure Age :=
  (value : ℕ)

/-- Represents the current ages of Jane and Dick --/
structure CurrentAges :=
  (jane : Age)
  (dick : Age)

/-- Represents the ages of Jane and Dick after n years --/
structure FutureAges :=
  (jane : Age)
  (dick : Age)

/-- Checks if an age is a two-digit number --/
def is_two_digit (age : Age) : Prop :=
  10 ≤ age.value ∧ age.value ≤ 99

/-- Checks if two ages have interchanged digits --/
def has_interchanged_digits (age1 age2 : Age) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ age1.value = 10 * a + b ∧ age2.value = 10 * b + a

/-- Calculates the number of valid (d, n) pairs --/
def count_valid_pairs (current : CurrentAges) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem valid_pairs_count (current : CurrentAges) :
  current.jane.value = 30 ∧
  current.dick.value > current.jane.value →
  (∀ n : ℕ, n > 0 →
    let future : FutureAges := ⟨⟨current.jane.value + n⟩, ⟨current.dick.value + n⟩⟩
    is_two_digit future.jane ∧
    is_two_digit future.dick ∧
    has_interchanged_digits future.jane future.dick) →
  count_valid_pairs current = 26 :=
by sorry

end valid_pairs_count_l3791_379138


namespace parabola_triangle_problem_l3791_379156

/-- Given three distinct points A, B, C on the parabola y = x^2, where AB is parallel to the x-axis
    and ABC forms a right triangle with area 2016, prove that the y-coordinate of C is 4064255 -/
theorem parabola_triangle_problem (A B C : ℝ × ℝ) : 
  (∃ m n : ℝ, A = (m, m^2) ∧ B = (n, n^2) ∧ C = ((m+n)/2, ((m+n)/2)^2)) →  -- Points on y = x^2
  (A.2 = B.2) →  -- AB parallel to x-axis
  (C.1 = (A.1 + B.1) / 2) →  -- C is above midpoint of AB (right angle)
  (abs (B.1 - A.1) * abs (C.2 - A.2) / 2 = 2016) →  -- Area of triangle ABC
  C.2 = 4064255 := by
  sorry

end parabola_triangle_problem_l3791_379156


namespace divisible_by_eight_l3791_379169

theorem divisible_by_eight (n : ℕ) : ∃ m : ℤ, 3^(4*n+1) + 5^(2*n+1) = 8*m := by
  sorry

end divisible_by_eight_l3791_379169
