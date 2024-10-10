import Mathlib

namespace valerie_stamps_l2929_292917

/-- Calculates the total number of stamps needed for mailing various items. -/
def total_stamps (thank_you_cards : ℕ) (bills : ℕ) (extra_rebates : ℕ) : ℕ :=
  let rebates := bills + extra_rebates
  let job_applications := 2 * rebates
  let regular_stamps := thank_you_cards + bills - 1 + rebates + job_applications
  regular_stamps + 1  -- Add 1 for the extra stamp on the electric bill

theorem valerie_stamps :
  total_stamps 3 2 3 = 21 :=
by sorry

end valerie_stamps_l2929_292917


namespace circle_path_in_triangle_l2929_292949

theorem circle_path_in_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : a = 9 ∧ b = 12 ∧ c = 15) (r : ℝ) (h_radius : r = 2) :
  let p := a + b + c
  let s := (c - 2*r) / c
  (s * p) = 26.4 := by sorry

end circle_path_in_triangle_l2929_292949


namespace correct_sum_l2929_292953

theorem correct_sum (a b : ℕ) (h1 : a % 10 = 1) (h2 : b / 10 % 10 = 8) 
  (h3 : (a - 1 + 7) + (b - 80 + 30) = 1946) : a + b = 1990 :=
by
  sorry

end correct_sum_l2929_292953


namespace alyssa_games_last_year_l2929_292984

/-- The number of soccer games Alyssa attended last year -/
def games_last_year : ℕ := sorry

/-- The number of soccer games Alyssa attended this year -/
def games_this_year : ℕ := 11

/-- The number of soccer games Alyssa plans to attend next year -/
def games_next_year : ℕ := 15

/-- The total number of soccer games Alyssa will have attended -/
def total_games : ℕ := 39

/-- Theorem stating that Alyssa attended 13 soccer games last year -/
theorem alyssa_games_last_year : 
  games_last_year + games_this_year + games_next_year = total_games ∧ 
  games_last_year = 13 := by sorry

end alyssa_games_last_year_l2929_292984


namespace lcm_hcf_relation_l2929_292919

theorem lcm_hcf_relation (x : ℕ) :
  Nat.lcm 4 x = 36 ∧ Nat.gcd 4 x = 2 → x = 18 := by
  sorry

end lcm_hcf_relation_l2929_292919


namespace faster_train_speed_is_72_l2929_292983

/-- The speed of the faster train given the conditions of the problem -/
def faster_train_speed (slower_train_speed : ℝ) (speed_difference : ℝ) 
  (crossing_time : ℝ) (train_length : ℝ) : ℝ :=
  slower_train_speed + speed_difference

/-- Theorem stating the speed of the faster train under the given conditions -/
theorem faster_train_speed_is_72 :
  faster_train_speed 36 36 20 200 = 72 := by
  sorry

end faster_train_speed_is_72_l2929_292983


namespace animal_ages_l2929_292967

theorem animal_ages (x : ℝ) 
  (h1 : 7 * (x - 3) = 2.5 * x - 3) : x + 2.5 * x = 14 := by
  sorry

end animal_ages_l2929_292967


namespace rectangle_area_l2929_292959

theorem rectangle_area (a b : ℕ) : 
  (2 * (a + b) = 16) →
  (a^2 + b^2 - 2*a*b - 4 = 0) →
  (a * b = 15) := by
sorry

end rectangle_area_l2929_292959


namespace thabo_owns_160_books_l2929_292955

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcoverNonfiction : ℕ
  paperbackNonfiction : ℕ
  paperbackFiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabosBooks : BookCollection where
  hardcoverNonfiction := 25
  paperbackNonfiction := 25 + 20
  paperbackFiction := 2 * (25 + 20)

/-- The total number of books in a collection -/
def totalBooks (books : BookCollection) : ℕ :=
  books.hardcoverNonfiction + books.paperbackNonfiction + books.paperbackFiction

/-- Theorem stating that Thabo owns 160 books in total -/
theorem thabo_owns_160_books : totalBooks thabosBooks = 160 := by
  sorry


end thabo_owns_160_books_l2929_292955


namespace exist_three_equal_digit_sums_l2929_292963

-- Define the sum of decimal digits function
def S (n : ℕ) : ℕ := sorry

-- State the theorem
theorem exist_three_equal_digit_sums :
  ∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 25 ∧
  S (a^6 + 2014) = S (b^6 + 2014) ∧ S (b^6 + 2014) = S (c^6 + 2014) := by sorry

end exist_three_equal_digit_sums_l2929_292963


namespace ship_speed_l2929_292993

/-- The speed of a ship in still water, given specific conditions of its journey on a river -/
theorem ship_speed (total_time : ℝ) (total_distance : ℝ) (current_speed : ℝ) : 
  total_time = 6 ∧ 
  total_distance = 36 ∧ 
  current_speed = 3 → 
  ∃ (ship_speed : ℝ), 
    ship_speed = 3 + 3 * Real.sqrt 2 ∧
    total_time * (ship_speed^2 - current_speed^2) = 2 * total_distance * ship_speed :=
by sorry

end ship_speed_l2929_292993


namespace amy_required_school_hours_per_week_l2929_292939

/-- Amy's work schedule and earnings --/
structure WorkSchedule where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_weeks : ℕ
  school_target_earnings : ℕ

/-- Calculate the required hours per week during school --/
def required_school_hours_per_week (schedule : WorkSchedule) : ℚ :=
  let hourly_rate : ℚ := schedule.summer_earnings / (schedule.summer_weeks * schedule.summer_hours_per_week)
  let total_school_hours : ℚ := schedule.school_target_earnings / hourly_rate
  total_school_hours / schedule.school_weeks

/-- Amy's specific work schedule --/
def amy_schedule : WorkSchedule := {
  summer_weeks := 8
  summer_hours_per_week := 40
  summer_earnings := 3200
  school_weeks := 32
  school_target_earnings := 4000
}

theorem amy_required_school_hours_per_week :
  required_school_hours_per_week amy_schedule = 12.5 := by
  sorry

end amy_required_school_hours_per_week_l2929_292939


namespace carpool_gas_expense_l2929_292937

/-- Calculates the monthly gas expense per person in a carpool scenario -/
theorem carpool_gas_expense
  (one_way_commute : ℝ)
  (gas_cost_per_gallon : ℝ)
  (car_efficiency : ℝ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (num_people : ℕ)
  (h1 : one_way_commute = 21)
  (h2 : gas_cost_per_gallon = 2.5)
  (h3 : car_efficiency = 30)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : num_people = 5) :
  (2 * one_way_commute * days_per_week * weeks_per_month / car_efficiency * gas_cost_per_gallon) / num_people = 14 := by
  sorry


end carpool_gas_expense_l2929_292937


namespace larger_number_proof_l2929_292982

theorem larger_number_proof (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 15 → L = 1635 := by
sorry

end larger_number_proof_l2929_292982


namespace min_value_quadratic_l2929_292968

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 2007 ∧ ∀ x, 3 * x^2 - 12 * x + 2023 ≥ m := by
  sorry

end min_value_quadratic_l2929_292968


namespace quadratic_inequality_solution_set_l2929_292925

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} :=
by sorry

end quadratic_inequality_solution_set_l2929_292925


namespace candy_bar_cost_l2929_292932

theorem candy_bar_cost (soft_drink_cost candy_bar_count total_spent : ℕ) :
  soft_drink_cost = 2 →
  candy_bar_count = 5 →
  total_spent = 27 →
  ∃ (candy_bar_cost : ℕ), candy_bar_cost * candy_bar_count + soft_drink_cost = total_spent ∧ candy_bar_cost = 5 :=
by sorry

end candy_bar_cost_l2929_292932


namespace slope_of_line_l2929_292914

theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end slope_of_line_l2929_292914


namespace travis_cereal_weeks_l2929_292921

/-- Proves the number of weeks Travis eats cereal given his consumption and spending habits -/
theorem travis_cereal_weeks (boxes_per_week : ℕ) (cost_per_box : ℚ) (total_spent : ℚ) :
  boxes_per_week = 2 →
  cost_per_box = 3 →
  total_spent = 312 →
  (total_spent / (boxes_per_week * cost_per_box) : ℚ) = 52 := by
  sorry

end travis_cereal_weeks_l2929_292921


namespace stating_acquaintance_group_relation_l2929_292987

/-- 
A group of people with specific acquaintance relationships.
-/
structure AcquaintanceGroup where
  n : ℕ  -- Total number of people
  k : ℕ  -- Number of acquaintances per person
  l : ℕ  -- Number of common acquaintances for acquainted pairs
  m : ℕ  -- Number of common acquaintances for non-acquainted pairs
  k_lt_n : k < n  -- Each person is acquainted with fewer than the total number of people

/-- 
Theorem stating the relationship between the parameters of an AcquaintanceGroup.
-/
theorem acquaintance_group_relation (g : AcquaintanceGroup) : 
  g.m * (g.n - g.k - 1) = g.k * (g.k - g.l - 1) := by
  sorry

end stating_acquaintance_group_relation_l2929_292987


namespace ellipse_eccentricity_l2929_292941

/-- Definition of an ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The maximum distance from a point on the ellipse to F₁ -/
def max_distance (E : Ellipse) : ℝ := 7

/-- The minimum distance from a point on the ellipse to F₁ -/
def min_distance (E : Ellipse) : ℝ := 1

/-- The eccentricity of an ellipse -/
def eccentricity (E : Ellipse) : ℝ := sorry

/-- Theorem: The square root of the eccentricity of the ellipse E is √3/2 -/
theorem ellipse_eccentricity (E : Ellipse) :
  Real.sqrt (eccentricity E) = Real.sqrt 3 / 2 := by sorry

end ellipse_eccentricity_l2929_292941


namespace cuboid_height_l2929_292945

/-- Proves that a rectangular parallelepiped with given dimensions has a specific height -/
theorem cuboid_height (width length sum_of_edges : ℝ) (h : ℝ) : 
  width = 30 →
  length = 22 →
  sum_of_edges = 224 →
  4 * length + 4 * width + 4 * h = sum_of_edges →
  h = 4 := by
sorry

end cuboid_height_l2929_292945


namespace bamboo_pole_sections_l2929_292979

theorem bamboo_pole_sections (n : ℕ) (a : ℕ → ℝ) : 
  (∀ i j, i < j → j ≤ n → a j - a i = (j - i) * (a 2 - a 1)) →  -- arithmetic sequence
  (a 1 = 10) →  -- top section length
  (a n + a (n-1) + a (n-2) = 114) →  -- last three sections total
  (a 6 ^ 2 = a 1 * a n) →  -- 6th section is geometric mean of first and last
  (n > 6) →
  n = 16 :=
by sorry

end bamboo_pole_sections_l2929_292979


namespace ferry_tourist_count_l2929_292999

/-- Represents the ferry schedule and passenger count --/
structure FerrySchedule where
  start_time : Nat -- 10 AM represented as 0
  end_time : Nat   -- 3 PM represented as 10
  initial_passengers : Nat
  passenger_decrease : Nat

/-- Calculates the total number of tourists transported by the ferry --/
def total_tourists (schedule : FerrySchedule) : Nat :=
  let num_trips := schedule.end_time - schedule.start_time + 1
  let arithmetic_sum := num_trips * (2 * schedule.initial_passengers - (num_trips - 1) * schedule.passenger_decrease)
  arithmetic_sum / 2

/-- Theorem stating that the total number of tourists is 990 --/
theorem ferry_tourist_count :
  ∀ (schedule : FerrySchedule),
    schedule.start_time = 0 ∧
    schedule.end_time = 10 ∧
    schedule.initial_passengers = 100 ∧
    schedule.passenger_decrease = 2 →
    total_tourists schedule = 990 := by
  sorry

end ferry_tourist_count_l2929_292999


namespace rectangular_field_area_l2929_292902

theorem rectangular_field_area (L W : ℝ) : 
  L = 40 →                 -- One side (length) is 40 feet
  2 * W + L = 74 →         -- Total fencing is 74 feet (two widths plus one length)
  L * W = 680 :=           -- The area of the field is 680 square feet
by sorry

end rectangular_field_area_l2929_292902


namespace parents_contribution_half_l2929_292948

/-- Represents the financial details for Nancy's university tuition --/
structure TuitionFinances where
  tuition : ℕ
  scholarship : ℕ
  workHours : ℕ
  hourlyWage : ℕ

/-- Calculates the ratio of parents' contribution to total tuition --/
def parentsContributionRatio (finances : TuitionFinances) : Rat :=
  let studentLoan := 2 * finances.scholarship
  let totalAid := finances.scholarship + studentLoan
  let workEarnings := finances.workHours * finances.hourlyWage
  let nancyContribution := totalAid + workEarnings
  let parentsContribution := finances.tuition - nancyContribution
  parentsContribution / finances.tuition

/-- Theorem stating that the parents' contribution ratio is 1/2 --/
theorem parents_contribution_half (finances : TuitionFinances) 
  (h1 : finances.tuition = 22000)
  (h2 : finances.scholarship = 3000)
  (h3 : finances.workHours = 200)
  (h4 : finances.hourlyWage = 10) :
  parentsContributionRatio finances = 1/2 := by
  sorry

end parents_contribution_half_l2929_292948


namespace absolute_value_w_l2929_292969

theorem absolute_value_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
sorry

end absolute_value_w_l2929_292969


namespace cos_transformation_l2929_292909

theorem cos_transformation (x : ℝ) : 
  Real.sqrt 2 * Real.cos (3 * x) = Real.sqrt 2 * Real.cos ((3 / 2) * (2 * x)) := by
  sorry

end cos_transformation_l2929_292909


namespace five_digit_divisible_by_nine_l2929_292954

theorem five_digit_divisible_by_nine :
  ∀ B : ℕ,
  (0 ≤ B ∧ B ≤ 9) →
  (40000 + 10000*B + 1000*B + 100 + 10 + 3) % 9 = 0 →
  B = 5 := by
sorry

end five_digit_divisible_by_nine_l2929_292954


namespace sheila_weekly_earnings_l2929_292950

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hourlyWage : ℝ
  hoursMonWedFri : ℝ
  hoursTueThu : ℝ
  daysWithLongHours : ℕ
  daysWithShortHours : ℕ

/-- Calculates the weekly earnings based on the work schedule --/
def weeklyEarnings (schedule : WorkSchedule) : ℝ :=
  (schedule.hourlyWage * schedule.hoursMonWedFri * schedule.daysWithLongHours) +
  (schedule.hourlyWage * schedule.hoursTueThu * schedule.daysWithShortHours)

/-- Theorem stating that Sheila's weekly earnings are $288 --/
theorem sheila_weekly_earnings :
  let schedule : WorkSchedule := {
    hourlyWage := 8,
    hoursMonWedFri := 8,
    hoursTueThu := 6,
    daysWithLongHours := 3,
    daysWithShortHours := 2
  }
  weeklyEarnings schedule = 288 := by
  sorry

end sheila_weekly_earnings_l2929_292950


namespace cannot_form_triangle_l2929_292978

/-- Represents the lengths of three line segments -/
structure Triangle :=
  (a b c : ℝ)

/-- Checks if three line segments can form a triangle -/
def canFormTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- Theorem: The set of line segments 2cm, 3cm, 6cm cannot form a triangle -/
theorem cannot_form_triangle :
  ¬ canFormTriangle ⟨2, 3, 6⟩ :=
sorry

end cannot_form_triangle_l2929_292978


namespace perfect_square_sum_l2929_292913

theorem perfect_square_sum : ∃ k : ℕ, 2^8 + 2^11 + 2^12 = k^2 := by
  sorry

end perfect_square_sum_l2929_292913


namespace total_snake_owners_is_75_l2929_292911

/-- The number of people in the neighborhood who own pets -/
def total_population : ℕ := 200

/-- The number of people who own only dogs -/
def only_dogs : ℕ := 30

/-- The number of people who own only cats -/
def only_cats : ℕ := 25

/-- The number of people who own only birds -/
def only_birds : ℕ := 10

/-- The number of people who own only snakes -/
def only_snakes : ℕ := 7

/-- The number of people who own only fish -/
def only_fish : ℕ := 12

/-- The number of people who own both cats and dogs -/
def cats_and_dogs : ℕ := 15

/-- The number of people who own both birds and dogs -/
def birds_and_dogs : ℕ := 12

/-- The number of people who own both birds and cats -/
def birds_and_cats : ℕ := 8

/-- The number of people who own both snakes and dogs -/
def snakes_and_dogs : ℕ := 3

/-- The number of people who own both snakes and cats -/
def snakes_and_cats : ℕ := 4

/-- The number of people who own both snakes and birds -/
def snakes_and_birds : ℕ := 2

/-- The number of people who own both fish and dogs -/
def fish_and_dogs : ℕ := 9

/-- The number of people who own both fish and cats -/
def fish_and_cats : ℕ := 6

/-- The number of people who own both fish and birds -/
def fish_and_birds : ℕ := 14

/-- The number of people who own both fish and snakes -/
def fish_and_snakes : ℕ := 11

/-- The number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : ℕ := 5

/-- The number of people who own cats, dogs, and birds -/
def cats_dogs_birds : ℕ := 4

/-- The number of people who own cats, birds, and snakes -/
def cats_birds_snakes : ℕ := 6

/-- The number of people who own dogs, birds, and snakes -/
def dogs_birds_snakes : ℕ := 9

/-- The number of people who own cats, fish, and dogs -/
def cats_fish_dogs : ℕ := 7

/-- The number of people who own birds, fish, and dogs -/
def birds_fish_dogs : ℕ := 5

/-- The number of people who own birds, fish, and cats -/
def birds_fish_cats : ℕ := 3

/-- The number of people who own snakes, fish, and dogs -/
def snakes_fish_dogs : ℕ := 8

/-- The number of people who own snakes, fish, and cats -/
def snakes_fish_cats : ℕ := 4

/-- The number of people who own snakes, fish, and birds -/
def snakes_fish_birds : ℕ := 6

/-- The number of people who own all five pets -/
def all_five_pets : ℕ := 10

/-- The total number of snake owners in the neighborhood -/
def total_snake_owners : ℕ := 
  only_snakes + snakes_and_dogs + snakes_and_cats + snakes_and_birds + 
  fish_and_snakes + cats_dogs_snakes + cats_birds_snakes + dogs_birds_snakes + 
  snakes_fish_dogs + snakes_fish_cats + snakes_fish_birds + all_five_pets

theorem total_snake_owners_is_75 : total_snake_owners = 75 := by
  sorry

end total_snake_owners_is_75_l2929_292911


namespace sally_reading_time_l2929_292960

/-- The number of pages Sally reads on a weekday -/
def weekday_pages : ℕ := 10

/-- The number of pages Sally reads on a weekend day -/
def weekend_pages : ℕ := 20

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of pages in Sally's book -/
def book_pages : ℕ := 180

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- Theorem stating that it takes Sally 2 weeks to finish the book -/
theorem sally_reading_time :
  weekday_pages * weekdays + weekend_pages * weekend_days = book_pages / weeks_to_finish :=
by sorry

end sally_reading_time_l2929_292960


namespace gigi_mushrooms_l2929_292970

/-- The number of pieces each mushroom is cut into -/
def pieces_per_mushroom : ℕ := 4

/-- The number of mushroom pieces Kenny used -/
def kenny_pieces : ℕ := 38

/-- The number of mushroom pieces Karla used -/
def karla_pieces : ℕ := 42

/-- The number of mushroom pieces left on the cutting board -/
def remaining_pieces : ℕ := 8

/-- Theorem stating that the total number of whole mushrooms GiGi cut up is 22 -/
theorem gigi_mushrooms :
  (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 := by
  sorry

end gigi_mushrooms_l2929_292970


namespace west_side_denial_percentage_l2929_292964

theorem west_side_denial_percentage :
  let total_kids := 260
  let riverside_kids := 120
  let west_side_kids := 90
  let mountaintop_kids := 50
  let riverside_denied_percentage := 20
  let mountaintop_denied_percentage := 50
  let kids_admitted := 148
  
  let riverside_denied := riverside_kids * riverside_denied_percentage / 100
  let mountaintop_denied := mountaintop_kids * mountaintop_denied_percentage / 100
  let total_denied := total_kids - kids_admitted
  let west_side_denied := total_denied - riverside_denied - mountaintop_denied
  let west_side_denied_percentage := west_side_denied / west_side_kids * 100

  west_side_denied_percentage = 70 := by sorry

end west_side_denial_percentage_l2929_292964


namespace ellipse_properties_l2929_292935

-- Define the ellipse (E)
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 16 * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 2

-- Define the focus of a parabola
def parabola_focus (x y : ℝ) : Prop :=
  parabola x y ∧ y = 0

-- Define a point on the ellipse
def point_on_ellipse (a b x y : ℝ) : Prop :=
  ellipse a b x y

-- Define a point on the major axis
def point_on_major_axis (m : ℝ) : Prop :=
  ∃ a b : ℝ, ellipse a b m 0

-- State the theorem
theorem ellipse_properties :
  ∃ a b : ℝ,
    (∀ x y : ℝ, ellipse a b x y →
      (∃ xf yf : ℝ, parabola_focus xf yf ∧ ellipse a b xf yf) ∧
      (∀ xh yh : ℝ, hyperbola xh yh → 
        ∃ c : ℝ, c^2 = a^2 - b^2 ∧ c^2 = xh^2 - yh^2 / 2)) →
    (a^2 = 16 ∧ b^2 = 12) ∧
    (∀ m : ℝ, point_on_major_axis m →
      (∀ x y : ℝ, point_on_ellipse a b x y →
        (x = 4 → (∀ x' y' : ℝ, point_on_ellipse a b x' y' →
          (x' - m)^2 + y'^2 ≥ (x - m)^2 + y^2))) →
      1 ≤ m ∧ m ≤ 4) :=
sorry

end ellipse_properties_l2929_292935


namespace martha_ellen_age_ratio_l2929_292920

/-- The ratio of Martha's age to Ellen's age in six years -/
def age_ratio (martha_current_age ellen_current_age : ℕ) : ℚ :=
  (martha_current_age + 6) / (ellen_current_age + 6)

/-- Theorem stating the ratio of Martha's age to Ellen's age in six years -/
theorem martha_ellen_age_ratio :
  age_ratio 32 10 = 19 / 8 := by
  sorry

end martha_ellen_age_ratio_l2929_292920


namespace even_function_property_l2929_292928

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h1 : EvenFunction f) 
  (h2 : ∀ x < 0, f x = x * (x + 1)) : 
  ∀ x > 0, f x = x * (x - 1) := by
  sorry

end even_function_property_l2929_292928


namespace sum_of_common_divisors_l2929_292940

def number_list : List Int := [48, 96, -16, 144, 192]

def is_common_divisor (d : Nat) : Bool :=
  number_list.all (fun n => n % d = 0)

def common_divisors : List Nat :=
  (List.range 193).filter is_common_divisor

theorem sum_of_common_divisors : (common_divisors.sum) = 15 := by
  sorry

end sum_of_common_divisors_l2929_292940


namespace math_sequences_count_l2929_292933

theorem math_sequences_count : 
  let letters := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S']
  let n := letters.length
  let first_letter := 'M'
  let last_letter_options := (letters.filter (· ≠ 'A')).filter (· ≠ first_letter)
  let middle_letters_count := 2
  (n - 1 - middle_letters_count).factorial * 
  last_letter_options.length * 
  (Nat.choose (n - 2) middle_letters_count) = 392 := by
sorry

end math_sequences_count_l2929_292933


namespace quadratic_solution_difference_squared_l2929_292986

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (2 * a^2 - 7 * a + 6 = 0) →
  (2 * b^2 - 7 * b + 6 = 0) →
  (a ≠ b) →
  (a - b)^2 = (1 : ℝ) / 4 := by
  sorry

end quadratic_solution_difference_squared_l2929_292986


namespace safe_elixir_preparations_l2929_292929

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- Represents the number of forbidden herb-crystal combinations. -/
def num_forbidden : ℕ := 3

/-- Calculates the number of safe elixir preparations. -/
def safe_preparations : ℕ := num_herbs * num_crystals - num_forbidden

/-- Theorem stating that the number of safe elixir preparations is 21. -/
theorem safe_elixir_preparations :
  safe_preparations = 21 := by sorry

end safe_elixir_preparations_l2929_292929


namespace simplify_radical_expression_l2929_292956

theorem simplify_radical_expression :
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end simplify_radical_expression_l2929_292956


namespace abigail_savings_l2929_292994

/-- Calculates the monthly savings given the total savings and number of months. -/
def monthly_savings (total_savings : ℕ) (num_months : ℕ) : ℕ :=
  total_savings / num_months

/-- Theorem stating that given a total savings of 48000 over 12 months, 
    the monthly savings is 4000. -/
theorem abigail_savings : monthly_savings 48000 12 = 4000 := by
  sorry

end abigail_savings_l2929_292994


namespace min_circles_6x3_min_circles_5x3_l2929_292915

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a circle
structure Circle where
  radius : ℝ

-- Define a function to calculate the minimum number of circles needed to cover a rectangle
def minCircles (r : Rectangle) (c : Circle) : ℕ :=
  sorry

-- Theorem for 6 × 3 rectangle
theorem min_circles_6x3 :
  let r := Rectangle.mk 6 3
  let c := Circle.mk (Real.sqrt 2)
  minCircles r c = 6 :=
sorry

-- Theorem for 5 × 3 rectangle
theorem min_circles_5x3 :
  let r := Rectangle.mk 5 3
  let c := Circle.mk (Real.sqrt 2)
  minCircles r c = 5 :=
sorry

end min_circles_6x3_min_circles_5x3_l2929_292915


namespace choir_arrangement_theorem_l2929_292975

theorem choir_arrangement_theorem (m : ℕ) : 
  (∃ y : ℕ, m = y^2 + 11) ∧ 
  (∃ n : ℕ, m = n * (n + 5)) ∧ 
  (∀ k : ℕ, (∃ y : ℕ, k = y^2 + 11) ∧ (∃ n : ℕ, k = n * (n + 5)) → k ≤ m) → 
  m = 300 :=
by sorry

end choir_arrangement_theorem_l2929_292975


namespace complex_equation_solution_l2929_292923

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 - Complex.I → z = -1 - Complex.I := by
  sorry

end complex_equation_solution_l2929_292923


namespace sum_of_interior_angles_cyclic_polygon_l2929_292980

/-- A cyclic polygon is a polygon whose vertices all lie on a single circle. -/
structure CyclicPolygon where
  n : ℕ
  sides_ge_4 : n ≥ 4

/-- The sum of interior angles of a cyclic polygon. -/
def sum_of_interior_angles (p : CyclicPolygon) : ℝ :=
  (p.n - 2) * 180

/-- Theorem: The sum of interior angles of a cyclic polygon with n sides is (n-2) * 180°. -/
theorem sum_of_interior_angles_cyclic_polygon (p : CyclicPolygon) :
  sum_of_interior_angles p = (p.n - 2) * 180 := by
  sorry

#check sum_of_interior_angles_cyclic_polygon

end sum_of_interior_angles_cyclic_polygon_l2929_292980


namespace circle_radius_problem_l2929_292958

/-- Given a circle with radius r and a point M at distance √7 from the center,
    if a secant from M intersects the circle such that the internal part
    of the secant is r and the external part is 2r, then r = 1. -/
theorem circle_radius_problem (r : ℝ) : 
  r > 0 →  -- r is positive (implicit condition for a circle's radius)
  (∃ (M : ℝ × ℝ) (C : ℝ × ℝ), 
    Real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2) = Real.sqrt 7 ∧  -- Distance from M to center is √7
    (∃ (A B : ℝ × ℝ),
      (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2 ∧  -- A is on the circle
      (B.1 - C.1)^2 + (B.2 - C.2)^2 = r^2 ∧  -- B is on the circle
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = r ∧  -- Internal part of secant is r
      Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 2*r  -- External part of secant is 2r
    )
  ) →
  r = 1 := by
sorry

end circle_radius_problem_l2929_292958


namespace tv_sale_net_effect_l2929_292927

/-- Given a TV set with an original price P, this theorem proves the net effect on total sale value
    after applying discounts, considering sales increase and variable costs. -/
theorem tv_sale_net_effect (P : ℝ) (original_volume : ℝ) (h_pos : P > 0) (h_vol_pos : original_volume > 0) :
  let price_after_initial_reduction := P * (1 - 0.22)
  let bulk_discount := price_after_initial_reduction * 0.05
  let loyalty_discount := price_after_initial_reduction * 0.10
  let price_after_all_discounts := price_after_initial_reduction - bulk_discount - loyalty_discount
  let new_sales_volume := original_volume * 1.86
  let variable_cost_per_unit := price_after_all_discounts * 0.10
  let net_price_after_costs := price_after_all_discounts - variable_cost_per_unit
  let original_total_sale := P * original_volume
  let new_total_sale := net_price_after_costs * new_sales_volume
  let net_effect := new_total_sale - original_total_sale
  ∃ ε > 0, |net_effect / original_total_sale - 0.109862| < ε :=
by sorry

end tv_sale_net_effect_l2929_292927


namespace log_equation_solution_l2929_292931

theorem log_equation_solution (a : ℕ) : 
  (10 - 2*a > 0) ∧ (a - 2 > 0) ∧ (a - 2 ≠ 1) → a = 4 := by
  sorry

end log_equation_solution_l2929_292931


namespace triangle_tangent_ratio_l2929_292952

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if tan A * tan B = 4(tan A + tan B) * tan C, then (a^2 + b^2) / c^2 = 9 -/
theorem triangle_tangent_ratio (a b c A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) → (0 < B) → (B < π) → (0 < C) → (C < π) →
  (A + B + C = π) →
  (Real.tan A * Real.tan B = 4 * (Real.tan A + Real.tan B) * Real.tan C) →
  ((a^2 + b^2) / c^2 = 9) := by
  sorry

end triangle_tangent_ratio_l2929_292952


namespace people_liking_neither_sport_l2929_292924

/-- Given a class with the following properties:
  * There are 16 people in total
  * 5 people like both baseball and football
  * 2 people only like baseball
  * 3 people only like football
  Prove that 6 people like neither baseball nor football -/
theorem people_liking_neither_sport (total : Nat) (both : Nat) (only_baseball : Nat) (only_football : Nat)
  (h_total : total = 16)
  (h_both : both = 5)
  (h_only_baseball : only_baseball = 2)
  (h_only_football : only_football = 3) :
  total - (both + only_baseball + only_football) = 6 := by
sorry

end people_liking_neither_sport_l2929_292924


namespace tape_area_calculation_l2929_292966

theorem tape_area_calculation (width : ℝ) (length : ℝ) (num_pieces : ℕ) (overlap : ℝ) 
  (h_width : width = 9.4)
  (h_length : length = 3.7)
  (h_num_pieces : num_pieces = 15)
  (h_overlap : overlap = 0.6) :
  let single_area := width * length
  let total_area_no_overlap := num_pieces * single_area
  let overlap_area := overlap * length
  let total_overlap_area := (num_pieces - 1) * overlap_area
  let total_area := total_area_no_overlap - total_overlap_area
  total_area = 490.62 := by sorry

end tape_area_calculation_l2929_292966


namespace work_completion_l2929_292904

theorem work_completion (days_first_group : ℝ) (men_second_group : ℕ) (days_second_group : ℝ) :
  days_first_group = 25 →
  men_second_group = 20 →
  days_second_group = 18.75 →
  ∃ (men_first_group : ℕ), 
    men_first_group * days_first_group = men_second_group * days_second_group ∧
    men_first_group = 15 :=
by
  sorry

end work_completion_l2929_292904


namespace concyclicity_equivalence_l2929_292942

-- Define the types for points and complex numbers
variable (P A B C D E F G H O₁ O₂ O₃ O₄ : ℂ)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral (A B C D : ℂ) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect (A B C D P : ℂ) : Prop := sorry

-- Define midpoints
def is_midpoint (M A B : ℂ) : Prop := M = (A + B) / 2

-- Define circumcenter
def is_circumcenter (O P Q R : ℂ) : Prop := sorry

-- Define concyclicity
def are_concyclic (A B C D : ℂ) : Prop := sorry

-- State the theorem
theorem concyclicity_equivalence :
  is_convex_quadrilateral A B C D →
  diagonals_intersect A B C D P →
  is_midpoint E A B →
  is_midpoint F B C →
  is_midpoint G C D →
  is_midpoint H D A →
  is_circumcenter O₁ P H E →
  is_circumcenter O₂ P E F →
  is_circumcenter O₃ P F G →
  is_circumcenter O₄ P G H →
  (are_concyclic O₁ O₂ O₃ O₄ ↔ are_concyclic A B C D) :=
by sorry

end concyclicity_equivalence_l2929_292942


namespace trigonometric_simplification_l2929_292946

theorem trigonometric_simplification (α : ℝ) :
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α :=
by sorry

end trigonometric_simplification_l2929_292946


namespace fraction_addition_l2929_292991

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l2929_292991


namespace noah_doctor_visits_l2929_292992

/-- The number of holidays Noah took in a year -/
def total_holidays : ℕ := 36

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of times Noah visits the doctor each month -/
def doctor_visits_per_month : ℕ := total_holidays / months_in_year

theorem noah_doctor_visits :
  doctor_visits_per_month = 3 :=
sorry

end noah_doctor_visits_l2929_292992


namespace arthur_muffins_l2929_292962

theorem arthur_muffins (initial_muffins : ℕ) : 
  initial_muffins + 48 = 83 → initial_muffins = 35 := by
  sorry

end arthur_muffins_l2929_292962


namespace sum_squares_of_roots_l2929_292944

theorem sum_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 4 * x₁ - 9 = 0) →
  (3 * x₂^2 + 4 * x₂ - 9 = 0) →
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 70/9) := by
sorry

end sum_squares_of_roots_l2929_292944


namespace arun_weight_average_l2929_292934

def arun_weight_range (w : ℝ) : Prop :=
  64 < w ∧ w < 72 ∧  -- Arun's opinion
  60 < w ∧ w < 70 ∧  -- Brother's opinion
  w ≤ 67 ∧           -- Mother's opinion
  63 ≤ w ∧ w ≤ 71 ∧  -- Sister's opinion
  62 < w ∧ w ≤ 73    -- Father's opinion

theorem arun_weight_average :
  (∃ a b : ℝ, a < b ∧
    (∀ w, a < w ∧ w ≤ b ↔ arun_weight_range w) ∧
    (b - a + 1) / 2 + a = 66) :=
sorry

end arun_weight_average_l2929_292934


namespace solve_equation_l2929_292973

theorem solve_equation : ∀ x : ℝ, x + 1 = 2 → x = 1 := by
  sorry

end solve_equation_l2929_292973


namespace min_value_2a_minus_ab_l2929_292996

def is_valid (a b : ℕ) : Prop := 0 < a ∧ a < 8 ∧ 0 < b ∧ b < 8

theorem min_value_2a_minus_ab :
  ∃ (a₀ b₀ : ℕ), is_valid a₀ b₀ ∧
  (∀ (a b : ℕ), is_valid a b → (2 * a - a * b : ℤ) ≥ (2 * a₀ - a₀ * b₀ : ℤ)) ∧
  (2 * a₀ - a₀ * b₀ : ℤ) = -35 :=
sorry

end min_value_2a_minus_ab_l2929_292996


namespace square_perimeter_l2929_292947

/-- The perimeter of a square with side length 11 cm is 44 cm. -/
theorem square_perimeter : 
  ∀ (s : ℝ), s = 11 → 4 * s = 44 := by
  sorry

end square_perimeter_l2929_292947


namespace number_comparisons_l2929_292976

theorem number_comparisons :
  (π > 3.14) ∧ (-Real.sqrt 3 < -Real.sqrt 2) ∧ (2 < Real.sqrt 5) := by
  sorry

end number_comparisons_l2929_292976


namespace field_length_calculation_l2929_292938

theorem field_length_calculation (width length : ℝ) (pond_side : ℝ) : 
  length = 2 * width →
  pond_side = 8 →
  pond_side^2 = (1 / 50) * (length * width) →
  length = 80 := by
sorry

end field_length_calculation_l2929_292938


namespace concentric_circles_annulus_area_l2929_292910

theorem concentric_circles_annulus_area (r R : ℝ) (h : r > 0) (H : R > 0) (eq : π * r^2 = π * R^2 / 2) :
  let annulus_area := π * R^2 / 2 - 2 * (π * R^2 / 4 - R^2 / 2)
  annulus_area = 2 * r^2 :=
by sorry

end concentric_circles_annulus_area_l2929_292910


namespace octal_calculation_l2929_292916

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Convert a decimal number to its octal representation --/
def toOctal (n : Nat) : OctalNumber :=
  sorry

/-- Add two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtract two octal numbers --/
def octalSub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Theorem: 72₈ - 45₈ + 23₈ = 50₈ in base 8 --/
theorem octal_calculation :
  octalAdd (octalSub (toOctal 72) (toOctal 45)) (toOctal 23) = toOctal 50 := by
  sorry

end octal_calculation_l2929_292916


namespace cyclic_sum_equals_two_l2929_292905

theorem cyclic_sum_equals_two (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (1 + a + a*b) / (1 + a + a*b + a*b*c) + 
  (1 + b + b*c) / (1 + b + b*c + b*c*d) + 
  (1 + c + c*d) / (1 + c + c*d + c*d*a) + 
  (1 + d + d*a) / (1 + d + d*a + d*a*b) = 2 := by
  sorry

end cyclic_sum_equals_two_l2929_292905


namespace antonov_remaining_packs_l2929_292900

/-- Calculates the number of remaining candy packs given the initial number of candies,
    the number of candies per pack, and the number of packs given away. -/
def remaining_packs (initial_candies : ℕ) (candies_per_pack : ℕ) (packs_given : ℕ) : ℕ :=
  (initial_candies - packs_given * candies_per_pack) / candies_per_pack

/-- Proves that Antonov has 2 packs of candy remaining. -/
theorem antonov_remaining_packs :
  remaining_packs 60 20 1 = 2 := by
  sorry

end antonov_remaining_packs_l2929_292900


namespace royal_family_children_count_l2929_292985

/-- Represents the royal family -/
structure RoyalFamily where
  king_age : ℕ
  queen_age : ℕ
  num_sons : ℕ
  num_daughters : ℕ
  children_total_age : ℕ

/-- The possible numbers of children for the royal family -/
def possible_children_numbers : Set ℕ := {7, 9}

theorem royal_family_children_count (family : RoyalFamily) 
  (h1 : family.king_age = 35)
  (h2 : family.queen_age = 35)
  (h3 : family.num_sons = 3)
  (h4 : family.num_daughters ≥ 1)
  (h5 : family.children_total_age = 35)
  (h6 : family.num_sons + family.num_daughters ≤ 20)
  (h7 : ∃ (n : ℕ), n > 0 ∧ family.king_age + n + family.queen_age + n = family.children_total_age + n * (family.num_sons + family.num_daughters)) :
  (family.num_sons + family.num_daughters) ∈ possible_children_numbers :=
sorry

end royal_family_children_count_l2929_292985


namespace power_multiplication_l2929_292922

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end power_multiplication_l2929_292922


namespace slope_60_degrees_l2929_292901

/-- The slope of a line with an angle of inclination of 60° is equal to √3 -/
theorem slope_60_degrees :
  let angle_of_inclination : ℝ := 60 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 := by
  sorry

end slope_60_degrees_l2929_292901


namespace quadratic_zero_in_interval_l2929_292930

/-- Given a quadratic function f(x) = ax^2 + bx + c, prove that it has a zero in the interval (-2, 0) under certain conditions. -/
theorem quadratic_zero_in_interval
  (a b c : ℝ)
  (h1 : 2 * a + c / 2 > b)
  (h2 : c < 0) :
  ∃ x : ℝ, -2 < x ∧ x < 0 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end quadratic_zero_in_interval_l2929_292930


namespace quadratic_properties_l2929_292936

def f (x : ℝ) := -x^2 + 3*x + 1

theorem quadratic_properties :
  (∀ x y, x < y → f y < f x) ∧ 
  (3/2 = -(-3)/(2*(-1))) ∧
  (∀ x y, x < y → y < 3/2 → f x < f y) ∧
  (∀ x, f x = 0 → x < 4) := by
sorry

end quadratic_properties_l2929_292936


namespace negation_equivalence_l2929_292918

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5*x₀ + 6 > 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end negation_equivalence_l2929_292918


namespace solve_for_y_l2929_292912

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := by
  sorry

end solve_for_y_l2929_292912


namespace three_identical_digits_divisible_by_37_l2929_292906

theorem three_identical_digits_divisible_by_37 (A : ℕ) (h : A < 10) :
  ∃ k : ℕ, 111 * A = 37 * k := by
  sorry

end three_identical_digits_divisible_by_37_l2929_292906


namespace garden_perimeter_l2929_292961

/-- Given a square garden with a pond, if the pond area is 20 square meters
    and the remaining garden area is 124 square meters,
    then the perimeter of the garden is 48 meters. -/
theorem garden_perimeter (s : ℝ) : 
  s > 0 → 
  s^2 = 20 + 124 → 
  4 * s = 48 :=
by
  sorry

end garden_perimeter_l2929_292961


namespace sequence_ratio_theorem_l2929_292977

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n

/-- Theorem: Given conditions on arithmetic and geometric sequences, prove the ratio -/
theorem sequence_ratio_theorem (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  a 1 = b 1 ∧ a 3 = b 2 ∧ a 7 = b 3 →
  (b 3 + b 4) / (b 4 + b 5) = 1 / 2 := by
  sorry

end sequence_ratio_theorem_l2929_292977


namespace asteroid_speed_comparison_l2929_292972

/-- Asteroid observation and speed comparison -/
theorem asteroid_speed_comparison 
  (distance_X13 : ℝ) 
  (time_X13 : ℝ) 
  (speed_X13 : ℝ) 
  (speed_Y14 : ℝ) 
  (h1 : distance_X13 = 2000) 
  (h2 : speed_X13 = distance_X13 / time_X13) 
  (h3 : speed_Y14 = 3 * speed_X13) : 
  speed_Y14 - speed_X13 = speed_X13 := by
  sorry

end asteroid_speed_comparison_l2929_292972


namespace parabolas_common_point_l2929_292995

/-- A parabola in the family y = -x^2 + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The y-coordinate of a point on a parabola given its x-coordinate -/
def Parabola.y_coord (para : Parabola) (x : ℝ) : ℝ :=
  -x^2 + para.p * x + para.q

/-- The condition that the vertex of a parabola lies on y = x^2 -/
def vertex_on_curve (para : Parabola) : Prop :=
  ∃ a : ℝ, para.y_coord a = a^2

theorem parabolas_common_point :
  ∀ p : ℝ, ∃ para : Parabola, 
    vertex_on_curve para ∧ 
    para.p = p ∧
    para.y_coord 0 = 0 := by
  sorry

end parabolas_common_point_l2929_292995


namespace absolute_value_plus_inverse_l2929_292997

theorem absolute_value_plus_inverse : |(-2 : ℝ)| + 3⁻¹ = 7/3 := by
  sorry

end absolute_value_plus_inverse_l2929_292997


namespace friend_walking_speed_difference_l2929_292990

theorem friend_walking_speed_difference 
  (total_distance : ℝ) 
  (p_distance : ℝ) 
  (hp : total_distance = 22) 
  (hpd : p_distance = 12) : 
  let q_distance := total_distance - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 20 := by
sorry

end friend_walking_speed_difference_l2929_292990


namespace chicken_eggs_today_l2929_292907

theorem chicken_eggs_today (eggs_yesterday : ℕ) (eggs_difference : ℕ) : 
  eggs_yesterday = 10 → eggs_difference = 59 → eggs_yesterday + eggs_difference = 69 :=
by sorry

end chicken_eggs_today_l2929_292907


namespace slope_determines_m_l2929_292943

/-- Given two points A(-2, m) and B(m, 4), if the slope of line AB is -2, then m = -8 -/
theorem slope_determines_m (m : ℝ) : 
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  (4 - m) / (m - (-2)) = -2 → m = -8 := by
sorry

end slope_determines_m_l2929_292943


namespace grace_marks_calculation_l2929_292965

theorem grace_marks_calculation (num_students : ℕ) (initial_avg : ℚ) (final_avg : ℚ) 
  (h1 : num_students = 35)
  (h2 : initial_avg = 37)
  (h3 : final_avg = 40) :
  (num_students * final_avg - num_students * initial_avg) / num_students = 3 := by
  sorry

end grace_marks_calculation_l2929_292965


namespace sum_of_three_numbers_l2929_292981

theorem sum_of_three_numbers (S : Finset ℕ) (h1 : S.card = 10) (h2 : S.sum id > 144) :
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b + c ≥ 54 := by
  sorry

end sum_of_three_numbers_l2929_292981


namespace linear_equation_m_value_l2929_292988

theorem linear_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b, (4 - m) * x^(|m| - 3) - 16 = a * x + b) ∧ 
  (m - 4 ≠ 0) → 
  m = 2 := by
sorry

end linear_equation_m_value_l2929_292988


namespace problem_statement_l2929_292974

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 5) :
  b / (a + b) + c / (b + c) + a / (c + a) = 10 := by
  sorry

end problem_statement_l2929_292974


namespace father_age_twice_marika_l2929_292989

/-- The year when Marika's father's age will be twice Marika's age -/
def target_year : ℕ := 2036

/-- Marika's age in 2006 -/
def marika_age_2006 : ℕ := 10

/-- The year of reference -/
def reference_year : ℕ := 2006

/-- Father's age is five times Marika's age in 2006 -/
def father_age_2006 : ℕ := 5 * marika_age_2006

theorem father_age_twice_marika (y : ℕ) :
  y = target_year →
  father_age_2006 + (y - reference_year) = 2 * (marika_age_2006 + (y - reference_year)) :=
by sorry

end father_age_twice_marika_l2929_292989


namespace dave_tickets_left_l2929_292926

def tickets_left (won : ℕ) (lost : ℕ) (used : ℕ) : ℕ :=
  won - lost - used

theorem dave_tickets_left : tickets_left 14 2 10 = 2 := by
  sorry

end dave_tickets_left_l2929_292926


namespace p_necessary_not_sufficient_for_q_l2929_292951

theorem p_necessary_not_sufficient_for_q :
  (∀ a b : ℝ, a^2 + b^2 = 0 → a + b = 0) ∧
  (∃ a b : ℝ, a + b = 0 ∧ a^2 + b^2 ≠ 0) := by
  sorry

end p_necessary_not_sufficient_for_q_l2929_292951


namespace h_in_terms_of_c_l2929_292903

theorem h_in_terms_of_c (a b c d e f g h : ℚ) : 
  8 = (6 / 100) * a →
  6 = (8 / 100) * b →
  9 = (5 / 100) * d →
  7 = (3 / 100) * e →
  c = b / a →
  f = d / a →
  g = e / b →
  h = f + g →
  h = (803 / 20) * c := by
sorry

end h_in_terms_of_c_l2929_292903


namespace marias_blueberries_l2929_292957

/-- Proves that Maria has 8 cartons of blueberries given the problem conditions -/
theorem marias_blueberries 
  (total_needed : ℕ) 
  (strawberries : ℕ) 
  (to_buy : ℕ) 
  (h1 : total_needed = 21)
  (h2 : strawberries = 4)
  (h3 : to_buy = 9) :
  total_needed - (strawberries + to_buy) = 8 := by
  sorry

#eval 21 - (4 + 9)  -- Should output 8

end marias_blueberries_l2929_292957


namespace quadratic_equation_real_roots_l2929_292971

theorem quadratic_equation_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(1+a)*x + (3*a^2 + 4*a*b + 4*b^2 + 2) = 0) → 
  (a = 1 ∧ b = -1/2) := by
sorry

end quadratic_equation_real_roots_l2929_292971


namespace current_speed_l2929_292908

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 9.4) :
  ∃ (man_speed current_speed : ℝ),
    speed_with_current = man_speed + current_speed ∧
    speed_against_current = man_speed - current_speed ∧
    current_speed = 2.8 := by
  sorry

end current_speed_l2929_292908


namespace triangle_angle_theorem_l2929_292998

theorem triangle_angle_theorem (a : ℝ) (x : ℝ) :
  (5 < a) → (a < 35) →
  (2 * a + 20) + (3 * a - 15) + x = 180 →
  x = 175 - 5 * a ∧
  ∃ (ε : ℝ), ε > 0 ∧ 35 - ε > a ∧
  max (2 * a + 20) (max (3 * a - 15) (175 - 5 * a)) = 88 :=
by sorry

end triangle_angle_theorem_l2929_292998
