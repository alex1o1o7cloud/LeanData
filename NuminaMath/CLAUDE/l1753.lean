import Mathlib

namespace tan_alpha_results_l1753_175371

theorem tan_alpha_results (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.sin (2*α) / (Real.sin α^2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end tan_alpha_results_l1753_175371


namespace pens_bought_proof_l1753_175328

/-- The cost of one pen in rubles -/
def pen_cost : ℕ := 21

/-- The amount Masha spent on pens in rubles -/
def masha_spent : ℕ := 357

/-- The amount Olya spent on pens in rubles -/
def olya_spent : ℕ := 441

/-- The total number of pens bought by Masha and Olya -/
def total_pens : ℕ := 38

theorem pens_bought_proof :
  pen_cost > 10 ∧
  masha_spent % pen_cost = 0 ∧
  olya_spent % pen_cost = 0 ∧
  masha_spent / pen_cost + olya_spent / pen_cost = total_pens :=
by sorry

end pens_bought_proof_l1753_175328


namespace sqrt_equation_solution_l1753_175398

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (2 * x + 9) = 11 → x = 56 := by
  sorry

end sqrt_equation_solution_l1753_175398


namespace absolute_value_equation_solution_l1753_175388

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x - 3| = 5 - 2*x) ↔ (x = 2 ∨ x = 8/3) :=
by sorry

end absolute_value_equation_solution_l1753_175388


namespace gym_class_counts_l1753_175354

/-- Given five gym classes with student counts P1, P2, P3, P4, and P5, prove that
    P2 = 5, P3 = 12.5, P4 = 25/3, and P5 = 25/3 given the following conditions:
    - P1 = 15
    - P1 = P2 + 10
    - P2 = 2 * P3 - 20
    - P3 = (P4 + P5) - 5
    - P4 = (1 / 2) * P5 + 5 -/
theorem gym_class_counts (P1 P2 P3 P4 P5 : ℚ) 
  (h1 : P1 = 15)
  (h2 : P1 = P2 + 10)
  (h3 : P2 = 2 * P3 - 20)
  (h4 : P3 = (P4 + P5) - 5)
  (h5 : P4 = (1 / 2) * P5 + 5) :
  P2 = 5 ∧ P3 = 25/2 ∧ P4 = 25/3 ∧ P5 = 25/3 := by
  sorry


end gym_class_counts_l1753_175354


namespace speed_difference_l1753_175342

/-- The difference in average speeds between two travelers -/
theorem speed_difference (distance : ℝ) (time1 time2 : ℝ) :
  distance > 0 ∧ time1 > 0 ∧ time2 > 0 →
  distance = 15 ∧ time1 = 1/3 ∧ time2 = 1/4 →
  (distance / time2) - (distance / time1) = 15 := by
  sorry

#check speed_difference

end speed_difference_l1753_175342


namespace return_percentage_is_80_percent_l1753_175362

/-- Represents the library book collection and loan statistics --/
structure LibraryStats where
  initial_books : ℕ
  final_books : ℕ
  loaned_books : ℕ

/-- Calculates the percentage of loaned books that were returned --/
def return_percentage (stats : LibraryStats) : ℚ :=
  ((stats.final_books - (stats.initial_books - stats.loaned_books)) / stats.loaned_books) * 100

/-- Theorem stating that the return percentage is 80% for the given statistics --/
theorem return_percentage_is_80_percent (stats : LibraryStats) 
  (h1 : stats.initial_books = 75)
  (h2 : stats.final_books = 64)
  (h3 : stats.loaned_books = 55) : 
  return_percentage stats = 80 := by
  sorry

end return_percentage_is_80_percent_l1753_175362


namespace last_number_in_first_set_l1753_175367

def first_set_mean : ℝ := 90
def second_set_mean : ℝ := 423

def first_set (x y : ℝ) : List ℝ := [28, x, 42, 78, y]
def second_set (x : ℝ) : List ℝ := [128, 255, 511, 1023, x]

theorem last_number_in_first_set (x y : ℝ) : 
  (List.sum (first_set x y) / 5 = first_set_mean) →
  (List.sum (second_set x) / 5 = second_set_mean) →
  y = 104 := by
  sorry

end last_number_in_first_set_l1753_175367


namespace jellybean_theorem_l1753_175311

/-- The number of jellybeans each person has -/
structure JellyBeans where
  arnold : ℕ
  lee : ℕ
  tino : ℕ
  joshua : ℕ

/-- The conditions of the jellybean distribution -/
def jellybean_conditions (j : JellyBeans) : Prop :=
  j.arnold = 5 ∧
  j.lee = 2 * j.arnold ∧
  j.tino = j.lee + 24 ∧
  j.joshua = 3 * j.arnold

/-- The theorem to prove -/
theorem jellybean_theorem (j : JellyBeans) 
  (h : jellybean_conditions j) : 
  j.tino = 34 ∧ j.arnold + j.lee + j.tino + j.joshua = 64 := by
  sorry

end jellybean_theorem_l1753_175311


namespace square_of_two_digit_number_ending_in_five_l1753_175399

theorem square_of_two_digit_number_ending_in_five (d : ℕ) 
  (h : d ≥ 1 ∧ d ≤ 9) : 
  (10 * d + 5)^2 = 100 * d * (d + 1) + 25 := by
  sorry

end square_of_two_digit_number_ending_in_five_l1753_175399


namespace point_in_region_l1753_175369

theorem point_in_region (m : ℝ) : 
  (2 * m + 3 * 1 - 5 > 0) ↔ (m > 1) := by
sorry

end point_in_region_l1753_175369


namespace smith_family_seating_l1753_175393

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smith_family_seating (boys girls : ℕ) (h : boys = 4 ∧ girls = 4) : 
  factorial (boys + girls) - (factorial boys * factorial girls) = 39744 :=
by sorry

end smith_family_seating_l1753_175393


namespace sphere_center_reciprocal_sum_l1753_175332

/-- Given a sphere with center (p,q,r) passing through the origin and three points on the axes,
    prove that the sum of reciprocals of its center coordinates equals 49/72 -/
theorem sphere_center_reciprocal_sum :
  ∀ (p q r : ℝ),
  (p^2 + q^2 + r^2 = p^2 + q^2 + r^2) ∧  -- Distance from center to origin
  (p^2 + q^2 + r^2 = (p-2)^2 + q^2 + r^2) ∧  -- Distance from center to (2,0,0)
  (p^2 + q^2 + r^2 = p^2 + (q-4)^2 + r^2) ∧  -- Distance from center to (0,4,0)
  (p^2 + q^2 + r^2 = p^2 + q^2 + (r-6)^2) →  -- Distance from center to (0,0,6)
  1/p + 1/q + 1/r = 49/72 := by
sorry

end sphere_center_reciprocal_sum_l1753_175332


namespace percentage_defective_meters_l1753_175387

def total_meters : ℕ := 4000
def rejected_meters : ℕ := 2

theorem percentage_defective_meters :
  (rejected_meters : ℝ) / total_meters * 100 = 0.05 := by
  sorry

end percentage_defective_meters_l1753_175387


namespace intersection_A_B_l1753_175312

def f (x : ℝ) : ℝ := x^2 - 12*x + 36

def A : Set ℕ := {a | 1 ≤ a ∧ a ≤ 10}

def B : Set ℕ := {b | ∃ a ∈ A, f a = b}

theorem intersection_A_B : A ∩ B = {1, 4, 9} := by sorry

end intersection_A_B_l1753_175312


namespace tan_theta_equation_l1753_175384

open Real

theorem tan_theta_equation (θ : ℝ) (h1 : π/4 < θ ∧ θ < π/2) 
  (h2 : tan θ + tan (3*θ) + tan (5*θ) = 0) : tan θ = sqrt 3 := by
  sorry

end tan_theta_equation_l1753_175384


namespace sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l1753_175335

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three :
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l1753_175335


namespace toy_organizer_price_correct_l1753_175339

/-- The price of a gaming chair in dollars -/
def gaming_chair_price : ℝ := 83

/-- The number of toy organizer sets ordered -/
def toy_organizer_sets : ℕ := 3

/-- The number of gaming chairs ordered -/
def gaming_chairs : ℕ := 2

/-- The delivery fee percentage -/
def delivery_fee_percent : ℝ := 0.05

/-- The total amount paid by Leon in dollars -/
def total_paid : ℝ := 420

/-- The price per set of toy organizers in dollars -/
def toy_organizer_price : ℝ := 78

theorem toy_organizer_price_correct :
  toy_organizer_price * toy_organizer_sets +
  gaming_chair_price * gaming_chairs +
  delivery_fee_percent * (toy_organizer_price * toy_organizer_sets + gaming_chair_price * gaming_chairs) =
  total_paid := by
  sorry

#check toy_organizer_price_correct

end toy_organizer_price_correct_l1753_175339


namespace length_of_PT_l1753_175382

/-- Given points P, Q, R, S, and T in a coordinate plane where PQ intersects RS at T,
    and the x-coordinate difference between P and Q is 6,
    and the y-coordinate difference between P and Q is 4,
    prove that the length of segment PT is 12√13/11 -/
theorem length_of_PT (P Q R S T : ℝ × ℝ) : 
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    T = (1 - t) • P + t • Q ∧
    T = (1 - t) • R + t • S) →
  Q.1 - P.1 = 6 →
  Q.2 - P.2 = 4 →
  Real.sqrt ((T.1 - P.1)^2 + (T.2 - P.2)^2) = 12 * Real.sqrt 13 / 11 :=
by sorry

end length_of_PT_l1753_175382


namespace earth_habitable_fraction_l1753_175307

theorem earth_habitable_fraction :
  (earth_land_fraction : ℚ) →
  (land_habitable_fraction : ℚ) →
  earth_land_fraction = 1/3 →
  land_habitable_fraction = 1/4 →
  earth_land_fraction * land_habitable_fraction = 1/12 :=
by sorry

end earth_habitable_fraction_l1753_175307


namespace fencing_cost_proof_l1753_175359

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (breadth : ℝ) (cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_proof :
  let length : ℝ := 66
  let breadth : ℝ := 34
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

end fencing_cost_proof_l1753_175359


namespace triangle_area_l1753_175318

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) :
  (1/2) * a * b = 336 := by
  sorry

end triangle_area_l1753_175318


namespace quadratic_function_unique_form_l1753_175348

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique_form
  (f : ℝ → ℝ)
  (hquad : QuadraticFunction f)
  (hf0 : f 0 = 1)
  (hfdiff : ∀ x, f (x + 1) - f x = 4 * x) :
  ∀ x, f x = 2 * x^2 - 2 * x + 1 :=
by sorry

end quadratic_function_unique_form_l1753_175348


namespace train_speed_before_accelerating_l1753_175389

/-- Calculates the average speed of a train before accelerating -/
theorem train_speed_before_accelerating
  (v : ℝ) (s : ℝ) 
  (h1 : v > 0) (h2 : s > 0) :
  ∃ (x : ℝ), x > 0 ∧ s / x = (s + 50) / (x + v) ∧ x = s * v / 50 :=
by sorry

end train_speed_before_accelerating_l1753_175389


namespace product_of_sum_and_sum_of_cubes_l1753_175396

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (sum_eq : a + b = 5)
  (sum_of_cubes_eq : a^3 + b^3 = 125) : 
  a * b = 0 := by
sorry

end product_of_sum_and_sum_of_cubes_l1753_175396


namespace apple_box_weight_l1753_175330

theorem apple_box_weight (total_weight : ℝ) (pies : ℕ) (apples_per_pie : ℝ) : 
  total_weight / 2 = pies * apples_per_pie →
  pies = 15 →
  apples_per_pie = 4 →
  total_weight = 120 := by
sorry

end apple_box_weight_l1753_175330


namespace inequality_implies_lower_bound_l1753_175380

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x : ℝ, |x - 2| - |x + 3| ≤ a) → a ≥ 5 := by
  sorry

end inequality_implies_lower_bound_l1753_175380


namespace no_real_solution_l1753_175394

theorem no_real_solution : ¬∃ (x y : ℝ), x^3 + y^2 = 2 ∧ x^2 + x*y + y^2 - y = 0 := by
  sorry

end no_real_solution_l1753_175394


namespace fraction_inequality_l1753_175331

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end fraction_inequality_l1753_175331


namespace parallel_alternate_interior_false_l1753_175327

-- Define the concept of lines
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define the concept of angles
structure Angle :=
  (measure : ℝ)

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define alternate interior angles
def alternate_interior (a1 a2 : Angle) (l1 l2 : Line) : Prop :=
  -- This definition is simplified for the purpose of this statement
  true

-- The theorem to be proved
theorem parallel_alternate_interior_false :
  ∃ (l1 l2 : Line) (a1 a2 : Angle),
    parallel l1 l2 ∧ ¬(alternate_interior a1 a2 l1 l2 → a1.measure = a2.measure) :=
sorry

end parallel_alternate_interior_false_l1753_175327


namespace friends_total_distance_l1753_175303

/-- Represents the distance walked by each friend -/
structure FriendDistances where
  lionel : ℕ  -- miles
  esther : ℕ  -- yards
  niklaus : ℕ  -- feet

/-- Converts miles to feet -/
def milesToFeet (miles : ℕ) : ℕ := miles * 5280

/-- Converts yards to feet -/
def yardsToFeet (yards : ℕ) : ℕ := yards * 3

/-- Calculates the total distance walked by all friends in feet -/
def totalDistanceInFeet (distances : FriendDistances) : ℕ :=
  milesToFeet distances.lionel + yardsToFeet distances.esther + distances.niklaus

/-- Theorem stating that the total distance walked by the friends is 26332 feet -/
theorem friends_total_distance (distances : FriendDistances) 
  (h1 : distances.lionel = 4)
  (h2 : distances.esther = 975)
  (h3 : distances.niklaus = 1287) :
  totalDistanceInFeet distances = 26332 := by
  sorry


end friends_total_distance_l1753_175303


namespace no_integer_solution_l1753_175386

theorem no_integer_solution : ¬ ∃ (x : ℤ), (x + 12 > 15) ∧ (-3*x > -9) := by
  sorry

end no_integer_solution_l1753_175386


namespace hyperbola_equation_l1753_175360

/-- A hyperbola with given eccentricity and distance from focus to asymptote -/
structure Hyperbola where
  e : ℝ  -- eccentricity
  b : ℝ  -- distance from focus to asymptote

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∃ (x y : ℝ), y^2 / 2 - x^2 / 4 = 1

/-- Theorem: For a hyperbola with eccentricity √3 and distance from focus to asymptote 2,
    the standard equation is y²/2 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_e : h.e = Real.sqrt 3) 
    (h_b : h.b = 2) : 
    standard_equation h := by
  sorry

end hyperbola_equation_l1753_175360


namespace simplify_and_evaluate_l1753_175392

theorem simplify_and_evaluate (x : ℕ) (h1 : x > 0) (h2 : 3 - x ≥ 0) :
  let expr := (1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1))
  x = 3 → expr = 1 := by
  sorry

end simplify_and_evaluate_l1753_175392


namespace article_cost_price_l1753_175364

/-- The cost price of an article satisfying certain profit conditions -/
theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C →  -- Condition 1: Selling price is 105% of cost price
  (1.05 * C - 1) = 1.1 * (0.95 * C) →  -- Condition 2: New selling price equals 110% of new cost price
  C = 200 := by
sorry

end article_cost_price_l1753_175364


namespace digit_repetition_property_l1753_175370

def repeat_digit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

theorem digit_repetition_property (n : ℕ) (h : n > 0) :
  (repeat_digit 6 n)^2 + repeat_digit 8 n = repeat_digit 4 (2*n) :=
sorry

end digit_repetition_property_l1753_175370


namespace taxi_problem_l1753_175324

theorem taxi_problem (fans : ℕ) (company_a company_b : ℕ) : 
  fans = 56 →
  company_b = company_a + 3 →
  5 * company_a < fans →
  6 * company_a > fans →
  4 * company_b < fans →
  5 * company_b > fans →
  company_a = 10 := by
sorry

end taxi_problem_l1753_175324


namespace library_rearrangement_l1753_175385

theorem library_rearrangement (total_books : ℕ) (initial_shelves : ℕ) (books_per_new_shelf : ℕ)
  (h1 : total_books = 1500)
  (h2 : initial_shelves = 50)
  (h3 : books_per_new_shelf = 28)
  (h4 : total_books % initial_shelves = 0) : -- Ensures equally-filled initial shelves
  (total_books % books_per_new_shelf : ℕ) = 14 := by
sorry

end library_rearrangement_l1753_175385


namespace number_of_men_l1753_175319

/-- Proves that the number of men is 15 given the specified conditions -/
theorem number_of_men (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  men = women ∧ women = 8 ∧ 
  total_earnings = 120 ∧
  men_wage = 8 ∧
  total_earnings = men_wage * men →
  men = 15 := by
sorry

end number_of_men_l1753_175319


namespace equation_solution_l1753_175304

theorem equation_solution (y : ℝ) : 
  y = (13/2)^4 ↔ 3 * y^(1/4) - (3 * y^(1/2)) / y^(1/4) = 13 - 2 * y^(1/4) := by
  sorry

end equation_solution_l1753_175304


namespace log_inequality_l1753_175300

theorem log_inequality (x y a b : ℝ) 
  (hx : 0 < x) (hy : x < y) (hy1 : y < 1) 
  (hb : 1 < b) (ha : b < a) : 
  Real.log x / b < Real.log y / a := by
  sorry

end log_inequality_l1753_175300


namespace crayons_lost_or_given_away_l1753_175317

theorem crayons_lost_or_given_away (initial_crayons end_crayons : ℕ) 
  (h1 : initial_crayons = 479)
  (h2 : end_crayons = 134) :
  initial_crayons - end_crayons = 345 :=
by sorry

end crayons_lost_or_given_away_l1753_175317


namespace temperature_difference_l1753_175355

def lowest_temp_beijing : Int := -10
def lowest_temp_hangzhou : Int := -1

theorem temperature_difference :
  lowest_temp_beijing - lowest_temp_hangzhou = 9 := by
  sorry

end temperature_difference_l1753_175355


namespace number_equation_l1753_175383

theorem number_equation : ∃ n : ℝ, (n - 5) * 4 = n * 2 ∧ n = 10 := by
  sorry

end number_equation_l1753_175383


namespace percentage_of_boy_scouts_l1753_175378

theorem percentage_of_boy_scouts (total_scouts : ℝ) (boy_scouts : ℝ) (girl_scouts : ℝ)
  (h1 : boy_scouts + girl_scouts = total_scouts)
  (h2 : 0.60 * total_scouts = 0.50 * boy_scouts + 0.6818 * girl_scouts)
  : boy_scouts / total_scouts = 0.45 := by
  sorry

end percentage_of_boy_scouts_l1753_175378


namespace smallest_sum_reciprocals_l1753_175376

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (a : ℕ) + b ≤ x + y ∧ (a : ℕ) + b = 64 :=
by sorry

end smallest_sum_reciprocals_l1753_175376


namespace max_k_value_l1753_175397

theorem max_k_value (x y k : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_k : k > 0)
  (eq_condition : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
by sorry

end max_k_value_l1753_175397


namespace units_digit_of_199_factorial_l1753_175349

theorem units_digit_of_199_factorial (n : ℕ) : n = 199 → (n.factorial % 10 = 0) := by
  sorry

end units_digit_of_199_factorial_l1753_175349


namespace game_outcome_theorem_l1753_175338

/-- Represents the outcome of the game -/
inductive GameOutcome
| Draw
| BWin

/-- Defines the game rules and determines the outcome for a given n -/
def gameOutcome (n : ℕ+) : GameOutcome :=
  if n ∈ ({1, 2, 4, 6} : Finset ℕ+) then
    GameOutcome.Draw
  else
    GameOutcome.BWin

/-- Theorem stating the game outcome for all positive integers n -/
theorem game_outcome_theorem (n : ℕ+) :
  (gameOutcome n = GameOutcome.Draw ↔ n ∈ ({1, 2, 4, 6} : Finset ℕ+)) ∧
  (gameOutcome n = GameOutcome.BWin ↔ n ∉ ({1, 2, 4, 6} : Finset ℕ+)) :=
by sorry

end game_outcome_theorem_l1753_175338


namespace simpsons_formula_volume_l1753_175329

/-- Simpson's formula for volume calculation -/
theorem simpsons_formula_volume
  (S : ℝ → ℝ) -- Cross-sectional area function
  (x₀ x₁ : ℝ) -- Start and end coordinates
  (h : ℝ) -- Height of the figure
  (hpos : 0 < h) -- Height is positive
  (hdiff : h = x₁ - x₀) -- Height definition
  (hquad : ∃ (a b c : ℝ), ∀ x, S x = a * x^2 + b * x + c) -- S is a quadratic polynomial
  :
  (∫ (x : ℝ) in x₀..x₁, S x) = 
    (h / 6) * (S x₀ + 4 * S ((x₀ + x₁) / 2) + S x₁) :=
by sorry

end simpsons_formula_volume_l1753_175329


namespace physics_class_size_l1753_175373

theorem physics_class_size (total_students : ℕ) 
  (math_only : ℚ) (physics_only : ℚ) (both : ℕ) :
  total_students = 100 →
  both = 10 →
  physics_only + both = 2 * (math_only + both) →
  math_only + physics_only + both = total_students →
  physics_only + both = 220 / 3 := by
  sorry

end physics_class_size_l1753_175373


namespace sum_and_count_theorem_l1753_175347

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_integers 20 40 + count_even_integers 20 40 = 641 := by
  sorry

end sum_and_count_theorem_l1753_175347


namespace parabola_equation_l1753_175308

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_positive : p > 0
  h_equation : ∀ x y, equation x y ↔ x^2 = 2*p*y
  h_focus : focus = (0, p/2)

/-- Theorem: If there exists a point M on parabola C such that |OM| = |MF| = 3,
    then the equation of parabola C is x^2 = 8y -/
theorem parabola_equation (C : Parabola) :
  (∃ M : ℝ × ℝ, C.equation M.1 M.2 ∧ 
    Real.sqrt (M.1^2 + M.2^2) = 3 ∧
    Real.sqrt ((M.1 - C.focus.1)^2 + (M.2 - C.focus.2)^2) = 3) →
  C.p = 4 ∧ ∀ x y, C.equation x y ↔ x^2 = 8*y :=
sorry

end parabola_equation_l1753_175308


namespace prime_equation_unique_solution_l1753_175315

theorem prime_equation_unique_solution (p q : ℕ) :
  Prime p ∧ Prime q ∧ p^3 - q^5 = (p + q)^2 ↔ p = 7 ∧ q = 3 := by
  sorry

end prime_equation_unique_solution_l1753_175315


namespace g_composition_fixed_points_l1753_175301

def g (x : ℝ) : ℝ := x^2 - 4*x

theorem g_composition_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 4 ∨ x = 5 := by
  sorry

end g_composition_fixed_points_l1753_175301


namespace range_of_p_l1753_175357

-- Define the set A
def A (p : ℝ) : Set ℝ := {x : ℝ | |x| * x^2 + (p + 2) * x + 1 = 0}

-- Define the theorem
theorem range_of_p (p : ℝ) : 
  (A p ∩ Set.Ici (0 : ℝ) = ∅) ↔ (-4 < p ∧ p < 0) := by sorry

end range_of_p_l1753_175357


namespace presentation_students_l1753_175343

/-- The number of students in a presentation, given Eunjeong's position -/
def total_students (students_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  students_in_front + 1 + (position_from_back - 1)

/-- Theorem stating the total number of students in the presentation -/
theorem presentation_students : total_students 7 6 = 13 := by
  sorry

end presentation_students_l1753_175343


namespace inverse_fifty_l1753_175337

theorem inverse_fifty (x : ℝ) : (1 / x = 50) → (x = 1 / 50) := by
  sorry

end inverse_fifty_l1753_175337


namespace expression_evaluation_l1753_175374

theorem expression_evaluation : (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end expression_evaluation_l1753_175374


namespace u_less_than_v_l1753_175350

theorem u_less_than_v (u v : ℝ) 
  (hu : (u + u^2 + u^3 + u^4 + u^5 + u^6 + u^7 + u^8) + 10*u^9 = 8)
  (hv : (v + v^2 + v^3 + v^4 + v^5 + v^6 + v^7 + v^8 + v^9 + v^10) + 10*v^11 = 8) :
  u < v := by
sorry

end u_less_than_v_l1753_175350


namespace largest_multiple_of_12_less_than_350_l1753_175353

theorem largest_multiple_of_12_less_than_350 : 
  ∀ n : ℕ, n * 12 < 350 → n * 12 ≤ 348 :=
by
  sorry

end largest_multiple_of_12_less_than_350_l1753_175353


namespace luke_coin_count_l1753_175372

def total_coins (quarter_piles dime_piles coins_per_pile : ℕ) : ℕ :=
  (quarter_piles + dime_piles) * coins_per_pile

theorem luke_coin_count :
  let quarter_piles : ℕ := 5
  let dime_piles : ℕ := 5
  let coins_per_pile : ℕ := 3
  total_coins quarter_piles dime_piles coins_per_pile = 30 := by
  sorry

end luke_coin_count_l1753_175372


namespace line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l1753_175352

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- State the theorem
theorem line_perp_to_plane_and_line_para_to_plane_implies_lines_perp
  (a b : Line) (α : Plane) :
  a ≠ b →  -- a and b are non-coincident
  perpToPlane a α →  -- a is perpendicular to α
  para b α →  -- b is parallel to α
  perp a b :=  -- then a is perpendicular to b
by sorry

end line_perp_to_plane_and_line_para_to_plane_implies_lines_perp_l1753_175352


namespace min_value_2x_plus_y_l1753_175361

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y - 3 = 0) :
  2*x + y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + 2*x₀*y₀ - 3 = 0 ∧ 2*x₀ + y₀ = 3 :=
sorry

end min_value_2x_plus_y_l1753_175361


namespace polynomial_remainder_l1753_175381

/-- Given a polynomial p(x) with the specified division properties, 
    prove that its remainder when divided by (x + 1)(x + 3) is (7/2)x + 13/2 -/
theorem polynomial_remainder (p : Polynomial ℚ) 
  (h1 : (p - 3).eval (-1) = 0)
  (h2 : (p + 4).eval (-3) = 0) :
  ∃ q : Polynomial ℚ, p = q * ((X + 1) * (X + 3)) + (7/2 * X + 13/2) := by
  sorry

end polynomial_remainder_l1753_175381


namespace sandwich_combinations_l1753_175309

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of sandwiches with turkey and mozzarella combinations. -/
def turkey_mozzarella_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and salami combinations. -/
def rye_salami_combos : ℕ := num_cheeses

/-- Represents the number of sandwiches with white bread and chicken combinations. -/
def white_chicken_combos : ℕ := num_cheeses

/-- Theorem stating the number of possible sandwich combinations. -/
theorem sandwich_combinations :
  num_breads * num_meats * num_cheeses - 
  (turkey_mozzarella_combos + rye_salami_combos + white_chicken_combos) = 193 := by
  sorry

end sandwich_combinations_l1753_175309


namespace rectangle_with_equal_adjacent_sides_is_square_l1753_175379

-- Define a rectangle
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_positive : length > 0)
  (width_positive : width > 0)

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Theorem: A rectangle with one pair of adjacent sides equal is a square
theorem rectangle_with_equal_adjacent_sides_is_square (r : Rectangle) 
  (h : r.length = r.width) : ∃ (s : Square), s.side = r.length :=
sorry

end rectangle_with_equal_adjacent_sides_is_square_l1753_175379


namespace tom_running_distance_l1753_175375

/-- Calculates the total distance run in a week given the number of days, hours per day, and speed. -/
def weekly_distance (days : ℕ) (hours_per_day : ℝ) (speed : ℝ) : ℝ :=
  days * hours_per_day * speed

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem tom_running_distance :
  weekly_distance 5 1.5 8 = 60 := by
  sorry

end tom_running_distance_l1753_175375


namespace rower_distance_l1753_175326

/-- Represents the problem of calculating the distance traveled by a rower in a river --/
theorem rower_distance (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : 
  man_speed = 10 →
  river_speed = 1.2 →
  total_time = 1 →
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (upstream_speed * downstream_speed * total_time) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance = 9.856 := by
sorry

#eval (10 - 1.2) * (10 + 1.2) / (2 * ((10 - 1.2) + (10 + 1.2)))

end rower_distance_l1753_175326


namespace product_negative_five_sum_options_l1753_175363

theorem product_negative_five_sum_options (a b c : ℤ) : 
  a * b * c = -5 → (a + b + c = 5 ∨ a + b + c = -3 ∨ a + b + c = -7) := by
sorry

end product_negative_five_sum_options_l1753_175363


namespace complex_equation_solution_l1753_175333

theorem complex_equation_solution (c d : ℂ) (x : ℝ) 
  (h1 : Complex.abs c = 3)
  (h2 : Complex.abs d = 5)
  (h3 : c * d = x - 6 * Complex.I) :
  x = 3 * Real.sqrt 21 := by
sorry

end complex_equation_solution_l1753_175333


namespace students_with_a_l1753_175336

theorem students_with_a (total_students : ℕ) (ratio : ℚ) 
  (h1 : total_students = 30) 
  (h2 : ratio = 2 / 3) : 
  ∃ (a_students : ℕ) (percentage : ℚ), 
    a_students = 20 ∧ 
    percentage = 200 / 3 ∧
    (a_students : ℚ) / total_students = ratio ∧
    percentage = (a_students : ℚ) / total_students * 100 := by
  sorry

end students_with_a_l1753_175336


namespace smallest_positive_period_l1753_175390

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period (f : ℝ → ℝ) 
  (h : ∀ x, f (3 * x) = f (3 * x - 3/2)) :
  ∃ T, T = 1/2 ∧ is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T' :=
sorry

end smallest_positive_period_l1753_175390


namespace rectangle_area_perimeter_sum_l1753_175314

theorem rectangle_area_perimeter_sum (a b : ℕ+) :
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 114 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 116 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 120 ∧
  ∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 122 ∧
  ¬∃ (x y : ℕ+), (x + 2) * (y + 2) - 2 = 118 :=
by sorry


end rectangle_area_perimeter_sum_l1753_175314


namespace square_sum_equals_sixteen_l1753_175302

theorem square_sum_equals_sixteen (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 10) : 
  x^2 + y^2 = 16 := by
sorry

end square_sum_equals_sixteen_l1753_175302


namespace sufficient_not_necessary_condition_l1753_175391

/-- The sequence a_n defined by n^2 - c*n for n ∈ ℕ+ is increasing -/
def is_increasing_sequence (c : ℝ) : Prop :=
  ∀ n : ℕ+, (n + 1)^2 - c*(n + 1) > n^2 - c*n

/-- c ≤ 2 is a sufficient but not necessary condition for the sequence to be increasing -/
theorem sufficient_not_necessary_condition :
  (∀ c : ℝ, c ≤ 2 → is_increasing_sequence c) ∧
  (∃ c : ℝ, c > 2 ∧ is_increasing_sequence c) :=
sorry

end sufficient_not_necessary_condition_l1753_175391


namespace quadratic_equation_solution_l1753_175377

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = a ∨ x = b) : 
  a = -2 ∧ b = 1 := by
  sorry

end quadratic_equation_solution_l1753_175377


namespace largest_r_for_sequence_convergence_l1753_175334

theorem largest_r_for_sequence_convergence (r : ℝ) : r > 2 →
  ∃ (a : ℕ → ℕ+), (∀ n, (a n : ℝ) ≤ a (n + 2) ∧ (a (n + 2) : ℝ) ≤ Real.sqrt ((a n : ℝ)^2 + r * (a (n + 1) : ℝ))) ∧
  ¬∃ M, ∀ n ≥ M, a (n + 2) = a n :=
by sorry

end largest_r_for_sequence_convergence_l1753_175334


namespace gum_cost_theorem_l1753_175316

/-- The price of one piece of gum in cents -/
def price_per_piece : ℕ := 2

/-- The number of pieces of gum being purchased -/
def num_pieces : ℕ := 5000

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 1/20

/-- The minimum number of pieces required for the discount to apply -/
def discount_threshold : ℕ := 4000

/-- Calculates the final cost in dollars after applying the discount if applicable -/
def final_cost : ℚ :=
  let total_cents := price_per_piece * num_pieces
  let discounted_cents := if num_pieces > discount_threshold
                          then total_cents * (1 - discount_rate)
                          else total_cents
  discounted_cents / 100

theorem gum_cost_theorem :
  final_cost = 95 := by sorry

end gum_cost_theorem_l1753_175316


namespace solve_letter_addition_puzzle_l1753_175368

theorem solve_letter_addition_puzzle (E F D : ℕ) 
  (h1 : E + F + D = 15)
  (h2 : F + E + 1 = 12)
  (h3 : E < 10 ∧ F < 10 ∧ D < 10)
  (h4 : E ≠ F ∧ F ≠ D ∧ E ≠ D) : D = 4 := by
  sorry

end solve_letter_addition_puzzle_l1753_175368


namespace quadratic_points_order_l1753_175344

/-- Given a quadratic function f(x) = x² + x - 1, prove that y₂ < y₁ < y₃ 
    where y₁, y₂, and y₃ are the y-coordinates of points on the graph of f 
    with x-coordinates -2, 0, and 2 respectively. -/
theorem quadratic_points_order : 
  let f : ℝ → ℝ := λ x ↦ x^2 + x - 1
  let y₁ : ℝ := f (-2)
  let y₂ : ℝ := f 0
  let y₃ : ℝ := f 2
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end quadratic_points_order_l1753_175344


namespace girls_picked_more_l1753_175395

-- Define the number of mushrooms picked by each person
variable (N I A V : ℕ)

-- Define the conditions
def natasha_most := N > I ∧ N > A ∧ N > V
def ira_not_least := I ≤ N ∧ I ≥ A ∧ I ≥ V
def alexey_more_than_vitya := A > V

-- Theorem to prove
theorem girls_picked_more (h1 : natasha_most N I A V) 
                          (h2 : ira_not_least N I A V) 
                          (h3 : alexey_more_than_vitya A V) : 
  N + I > A + V := by
  sorry

end girls_picked_more_l1753_175395


namespace quadratic_equal_roots_l1753_175323

theorem quadratic_equal_roots (a b c : ℝ) 
  (h1 : b ≠ c) 
  (h2 : ∃ x : ℝ, (b - c) * x^2 + (a - b) * x + (c - a) = 0 ∧ 
       ((b - c) * (2 * x) + (a - b) = 0)) : 
  c = (a + b) / 2 := by
sorry

end quadratic_equal_roots_l1753_175323


namespace cats_remaining_l1753_175340

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 15 → house = 49 → sold = 19 → siamese + house - sold = 45 := by
  sorry

end cats_remaining_l1753_175340


namespace arrangements_count_l1753_175313

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of venues -/
def num_venues : ℕ := 4

/-- Represents the condition that A is assigned to the badminton venue -/
def a_assigned_to_badminton : Prop := true

/-- Represents the condition that each volunteer goes to only one venue -/
def one_venue_per_volunteer : Prop := true

/-- Represents the condition that each venue has at least one volunteer -/
def at_least_one_volunteer_per_venue : Prop := true

/-- The total number of different arrangements -/
def total_arrangements : ℕ := 60

/-- Theorem stating that the number of arrangements is 60 -/
theorem arrangements_count :
  num_volunteers = 5 ∧
  num_venues = 4 ∧
  a_assigned_to_badminton ∧
  one_venue_per_volunteer ∧
  at_least_one_volunteer_per_venue →
  total_arrangements = 60 :=
by sorry

end arrangements_count_l1753_175313


namespace circle_through_three_points_l1753_175366

/-- A circle in a 2D plane --/
structure Circle where
  /-- The coefficient of x^2 (always 1 for a standard form circle equation) --/
  a : ℝ := 1
  /-- The coefficient of y^2 (always 1 for a standard form circle equation) --/
  b : ℝ := 1
  /-- The coefficient of x --/
  d : ℝ
  /-- The coefficient of y --/
  e : ℝ
  /-- The constant term --/
  f : ℝ

/-- Check if a point (x, y) lies on the circle --/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  c.a * x^2 + c.b * y^2 + c.d * x + c.e * y + c.f = 0

/-- The theorem stating that there exists a unique circle passing through three given points --/
theorem circle_through_three_points :
  ∃! c : Circle,
    c.contains 0 0 ∧
    c.contains 1 1 ∧
    c.contains 4 2 ∧
    c.a = 1 ∧
    c.b = 1 ∧
    c.d = -8 ∧
    c.e = 6 ∧
    c.f = 0 := by
  sorry

end circle_through_three_points_l1753_175366


namespace sum_of_multiples_l1753_175346

theorem sum_of_multiples (x y : ℤ) (hx : 6 ∣ x) (hy : 9 ∣ y) : 3 ∣ (x + y) := by
  sorry

end sum_of_multiples_l1753_175346


namespace infinite_geometric_series_first_term_l1753_175322

/-- An infinite geometric series with common ratio 1/4 and sum 40 has a first term of 30. -/
theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1/4)
  (h_S : S = 40)
  (h_sum : S = a / (1 - r))
  : a = 30 := by
  sorry

end infinite_geometric_series_first_term_l1753_175322


namespace unique_permutation_with_difference_one_l1753_175305

theorem unique_permutation_with_difference_one (n : ℕ+) :
  ∃! (x : Fin (2 * n) → Fin (2 * n)), 
    Function.Bijective x ∧ 
    (∀ i : Fin (2 * n), |x i - i.val| = 1) := by
  sorry

end unique_permutation_with_difference_one_l1753_175305


namespace reimu_win_probability_l1753_175320

/-- Represents a coin with two sides that can be colored -/
structure Coin :=
  (side1 : Color)
  (side2 : Color)

/-- Possible colors for a coin side -/
inductive Color
  | White
  | Red
  | Green

/-- The game state -/
structure GameState :=
  (coins : List Coin)
  (currentPlayer : Player)

/-- The players in the game -/
inductive Player
  | Reimu
  | Sanae

/-- The result of the game -/
inductive GameResult
  | ReimuWins
  | SanaeWins
  | Tie

/-- Represents an optimal strategy for playing the game -/
def OptimalStrategy := GameState → Color

/-- The probability of a specific game result given optimal play -/
def resultProbability (strategy : OptimalStrategy) (result : GameResult) : ℚ :=
  sorry

/-- Theorem stating the probability of Reimu winning is 5/16 -/
theorem reimu_win_probability (strategy : OptimalStrategy) :
  resultProbability strategy GameResult.ReimuWins = 5 / 16 :=
sorry

end reimu_win_probability_l1753_175320


namespace factorization_difference_of_squares_l1753_175356

theorem factorization_difference_of_squares (a b : ℝ) : 3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by
  sorry

end factorization_difference_of_squares_l1753_175356


namespace nested_radical_value_l1753_175358

theorem nested_radical_value :
  ∃ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end nested_radical_value_l1753_175358


namespace animal_count_l1753_175306

theorem animal_count (num_cats : ℕ) : 
  (1 : ℕ) +                   -- 1 dog
  num_cats +                  -- cats
  2 * num_cats +              -- rabbits (2 per cat)
  3 * (2 * num_cats) = 37 →   -- hares (3 per rabbit)
  num_cats = 4 := by
sorry

end animal_count_l1753_175306


namespace real_square_properties_l1753_175310

theorem real_square_properties (a b : ℝ) : 
  (a^2 ≠ b^2 → a ≠ b) ∧ (a > |b| → a^2 > b^2) := by
  sorry

end real_square_properties_l1753_175310


namespace basketball_win_rate_l1753_175341

theorem basketball_win_rate (initial_wins : Nat) (initial_games : Nat) 
  (remaining_games : Nat) (target_win_rate : Rat) :
  initial_wins = 35 →
  initial_games = 45 →
  remaining_games = 55 →
  target_win_rate = 3/4 →
  ∃ (remaining_wins : Nat),
    remaining_wins = 40 ∧
    (initial_wins + remaining_wins : Rat) / (initial_games + remaining_games) = target_win_rate :=
by
  sorry

end basketball_win_rate_l1753_175341


namespace product_102_108_l1753_175365

theorem product_102_108 : 102 * 108 = 11016 := by
  sorry

end product_102_108_l1753_175365


namespace no_triangle_with_given_conditions_l1753_175325

theorem no_triangle_with_given_conditions : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧  -- positive sides
  (c = 0.2 * a) ∧            -- shortest side is 20% of longest
  (b = 0.25 * (a + b + c)) ∧ -- third side is 25% of perimeter
  (a + b > c ∧ a + c > b ∧ b + c > a) -- triangle inequality
  := by sorry

end no_triangle_with_given_conditions_l1753_175325


namespace complex_fraction_sum_l1753_175345

theorem complex_fraction_sum : (2 + 2 * Complex.I) / Complex.I + (1 + Complex.I) / (1 - Complex.I) = 2 - Complex.I := by
  sorry

end complex_fraction_sum_l1753_175345


namespace exactly_one_correct_probability_l1753_175351

theorem exactly_one_correct_probability
  (prob_A prob_B : ℝ)
  (h_A : prob_A = 0.8)
  (h_B : prob_B = 0.7)
  (h_independent : True)  -- Representing independence
  : prob_A * (1 - prob_B) + prob_B * (1 - prob_A) = 0.38 := by
  sorry

end exactly_one_correct_probability_l1753_175351


namespace expression_simplification_l1753_175321

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3) :
  (x^2 - 2*x + 1) / (x^2 - x) / (x - 1) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1753_175321
